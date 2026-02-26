#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Meta-Learning PINN v5  —  Raman Spectroscopy Transfer Learning             ║
║  DIG4BIO Raman Transfer Learning Challenge                                   ║
║                                                                              ║
║  Why previous versions failed (R² 0.35):                                    ║
║  ─────────────────────────────────────────                                   ║
║  We were treating this as a 96-sample regression problem.                   ║
║  It's actually a TRANSFER LEARNING problem.                                  ║
║  The 8 device CSV files have THOUSANDS of labeled spectra — we never        ║
║  used them. The HGB "cheat" ignores them too, but it has a simpler model.   ║
║                                                                              ║
║  Correct framing:                                                            ║
║  ─────────────────                                                           ║
║  Source domains : 8 Raman devices  (thousands of labeled samples each)      ║
║  Target domain  : transfer_plate   (96 labeled samples, new instrument)     ║
║  Test           : 96_samples.csv   (same instrument as transfer_plate)      ║
║                                                                              ║
║  Meta-Learning Pipeline:                                                     ║
║  ────────────────────────                                                    ║
║  Step 1 — Shared spectral preprocessing onto a common wavenumber grid       ║
║  Step 2 — MAML-style meta-training across the 8 device tasks:               ║
║           • Each episode: sample a device, split support/query              ║
║           • Inner loop (2 steps): adapt to support set                      ║
║           • Outer loop: update shared init to minimise query loss           ║
║  Step 3 — Fine-tune the meta-learned init on all 96 transfer plate samples  ║
║  Step 4 — Cross-validate the fine-tuning step for robust predictions        ║
║                                                                              ║
║  Physics laws in the loss (all phases):                                      ║
║  ① Beer-Lambert non-negativity  : c ≥ 0                                     ║
║  ② Mass-balance soft constraint : c ≤ physical ceiling                      ║
║  ③ Fingerprint invariance       : c(x) ≈ c(aug(x))                          ║
║  ④ Target scaling               : y normalised per task (z-scores)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python pinn_raman_v5.py

Requires:
    torch, numpy, pandas, scipy, scikit-learn
    pip install pybaselines   (optional — falls back to poly baseline)
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR  = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"
OUT_DIR   = "./pinn_outputs"
SEED      = 42

# Shared wavenumber grid
WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0   # 1643 bins

# Device files (source domains for meta-learning)
DEVICE_FILES = [
    "anton_532.csv", "anton_785.csv", "kaiser.csv", "mettler_toledo.csv",
    "metrohm.csv", "tec5.csv", "timegate.csv", "tornado.csv",
]

# PCA compression
N_PCA = 60   # slightly more than v4 because we have far more pre-training data

# Architecture — shared encoder
# Encoder: N_PCA → 128 → 64   (shared across all devices)
# Head:    64    → 32  → 3    (task-specific, re-initialised at fine-tune)
ENC_H1  = 128
ENC_H2  = 64
HEAD_H  = 32
N_OUT   = 3
DROPOUT = 0.30

# ── Meta-training (MAML)
META_EPOCHS      = 80       # outer loop epochs
META_TASKS_PER_EPOCH = 16   # episodes per epoch
K_SUPPORT        = 20       # samples per device in support set
K_QUERY          = 20       # samples per device in query set
META_LR          = 5e-4     # outer (meta) learning rate
INNER_LR         = 1e-2     # inner loop learning rate
INNER_STEPS      = 3        # gradient steps in inner loop
META_WEIGHT_DECAY= 1e-4

# ── Fine-tuning on transfer plate
FT_EPOCHS        = 300
FT_LR            = 5e-4
FT_WEIGHT_DECAY  = 5e-4
FT_BATCH         = 24
FT_DROPOUT       = 0.40     # higher dropout for 96 samples
N_FOLDS          = 5
N_RESTARTS       = 6
PATIENCE         = 50       # in units of 5 epochs

# Physics loss
PHYS_START_FT    = 30       # epoch in fine-tuning to start physics losses
PHYS_FULL_FT     = 100
LW_MSE           = 1.00
LW_NONNEG_C      = 0.20
LW_MASS          = 0.15
LW_INV           = 0.08

CONC_MAX_RAW     = np.array([15.0, 3.0, 4.0], dtype=np.float32)

# Augmentation (in PCA space)
AUG_SCALE        = (0.88, 1.12)
AUG_NOISE        = 0.05

TTA_N            = 20


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(s):
    np.random.seed(s);  torch.manual_seed(s);  torch.cuda.manual_seed_all(s)


def make_grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Target scaler
# ──────────────────────────────────────────────────────────────────────────────

class TargetScaler:
    def fit(self, y):
        self.mean = y.mean(0).astype(np.float32)
        self.std  = np.maximum(y.std(0), 1e-6).astype(np.float32)
        return self
    def transform(self, y):
        return ((y - self.mean) / self.std).astype(np.float32)
    def inverse(self, yn):
        return (yn * self.std + self.mean).astype(np.float32)
    def zero_norm(self):
        return (-self.mean / self.std).astype(np.float32)
    def max_norm(self, cmax):
        return ((cmax - self.mean) / self.std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _clean(s):
    return pd.to_numeric(
        s.astype(str).str.replace("[","",regex=False).str.replace("]","",regex=False),
        errors="coerce")


def load_device(path: str, grid_wns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one device CSV.
    Format: spectral columns (wavenumber as header) + last 5 cols (analytes + fold index).
    Labels = first 3 of df.iloc[:, -5:-1].
    Wavenumbers are inferred from column names.
    """
    df = pd.read_csv(path)

    # Detect spectral columns (numeric-named columns excluding last 5)
    spec_cols = df.columns[:-5]
    wns_raw   = []
    valid_idx = []
    for i, c in enumerate(spec_cols):
        try:
            wns_raw.append(float(c))
            valid_idx.append(i)
        except ValueError:
            pass

    wns_raw = np.array(wns_raw, dtype=np.float64)
    X_raw   = df.iloc[:, valid_idx].values.astype(np.float64)

    # Sort by wavenumber (required for np.interp)
    order   = np.argsort(wns_raw)
    wns_raw = wns_raw[order]
    X_raw   = X_raw[:, order]

    # Interpolate onto shared grid
    lo, hi  = float(grid_wns.min()), float(grid_wns.max())
    in_range = (wns_raw >= lo) & (wns_raw <= hi)
    if in_range.sum() < 10:
        return None, None   # device doesn't cover the grid range

    X_interp = np.array(
        [np.interp(grid_wns, wns_raw[in_range], row[in_range]) for row in X_raw],
        dtype=np.float32)

    # Labels: first 3 of last 5 columns (before the fold index)
    y = df.iloc[:, -5:-2].values[:, :3].astype(np.float32)
    # Drop rows with any NaN in labels
    valid = np.all(np.isfinite(y), axis=1)
    return X_interp[valid], y[valid]


def load_plate(path: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_train:
        df    = pd.read_csv(path)
        tcols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y     = df[tcols].dropna().values.astype(np.float32)
        Xdf   = df.iloc[:, :-4].copy()
    else:
        df    = pd.read_csv(path, header=None);  y = None;  Xdf = df.copy()

    Xdf.columns = ["sample_id"] + [str(i) for i in range(Xdf.shape[1]-1)]
    Xdf["sample_id"] = Xdf["sample_id"].ffill()
    for col in Xdf.columns[1:]:
        Xdf[col] = _clean(Xdf[col])

    X = Xdf.drop(columns=["sample_id"]).values.astype(np.float32)
    X = X.reshape(-1, 2, 2048).mean(axis=1)
    return X, y


def plate_to_grid(X2048, grid_wns, s=65.0, e=3350.0):
    full = np.linspace(s, e, 2048, dtype=np.float64)
    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    sel = (full >= lo) & (full <= hi)
    return np.array(
        [np.interp(grid_wns, full[sel], r) for r in X2048[:, sel].astype(np.float64)],
        dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Spectral preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def apply_msc(X, ref=None):
    X64 = X.astype(np.float64)
    ref = X64.mean(0) if ref is None else ref.astype(np.float64)
    out = np.empty_like(X64)
    for i in range(len(X64)):
        m, b = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - b) / (m + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)


def apply_baseline(X):
    try:
        import pybaselines
        fitter = pybaselines.Baseline()
        out    = np.empty_like(X, dtype=np.float64)
        for i, row in enumerate(X.astype(np.float64)):
            base, _ = fitter.snip(row, max_half_window=20,
                                  decreasing=True, smooth_half_window=3)
            out[i] = row - base
        return out.astype(np.float32)
    except ImportError:
        t   = np.linspace(0, 1, X.shape[1])
        out = np.empty_like(X, dtype=np.float64)
        for i, row in enumerate(X.astype(np.float64)):
            out[i] = row - np.polyval(np.polyfit(t, row, 3), t)
        return out.astype(np.float32)


class SpectralPreprocessor:
    """Fit MSC reference + StandardScaler on training data, transform everything else."""
    def __init__(self):
        self.msc_ref = None;  self.scaler = None

    def fit_transform(self, X):
        X, self.msc_ref = apply_msc(X)
        X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64), 7, 2, 0, axis=1).astype(np.float32)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X).astype(np.float32)

    def transform(self, X):
        X, _ = apply_msc(X, self.msc_ref)
        X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64), 7, 2, 0, axis=1).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_pca(Z, rng, scale=AUG_SCALE, noise=AUG_NOISE):
    Z = Z.copy()
    Z *= rng.uniform(*scale, (len(Z), 1)).astype(np.float32)
    if noise > 0:
        Z += rng.normal(0, noise, Z.shape).astype(np.float32)
    return Z


# ──────────────────────────────────────────────────────────────────────────────
# Model — shared encoder + task head
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Shared feature extractor learned via MAML across devices."""
    def __init__(self, n_pca=N_PCA, h1=ENC_H1, h2=ENC_H2, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pca, h1),  nn.LayerNorm(h1),  nn.GELU(),  nn.Dropout(dropout),
            nn.Linear(h1,    h2),  nn.LayerNorm(h2),  nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight);  nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class Head(nn.Module):
    """Task-specific prediction head. Small + high dropout for fine-tuning on 96 samples."""
    def __init__(self, h2=ENC_H2, hh=HEAD_H, n_out=N_OUT, dropout=FT_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h2, hh),  nn.LayerNorm(hh),  nn.GELU(),  nn.Dropout(dropout),
            nn.Linear(hh, n_out),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class MetaModel(nn.Module):
    def __init__(self, n_pca=N_PCA):
        super().__init__()
        self.encoder = Encoder(n_pca)
        self.head    = Head()

    def forward(self, z):
        return self.head(self.encoder(z))


# ──────────────────────────────────────────────────────────────────────────────
# Physics loss
# ──────────────────────────────────────────────────────────────────────────────

def physics_loss(pred, true, pred_aug, zero_n, max_n, pw):
    """
    Physics-Informed Loss — named components for full transparency.

    Objective:
      Meta-learning irons out device-to-device instrument offsets.
      PINN terms keep predictions physically valid at all times.

    Components:
      mse    — primary regression objective (always active)
      nonneg — Beer-Lambert: concentrations are non-negative  (c >= 0)
      mass   — mass balance: concentrations cannot exceed physical ceiling
      inv    — Raman fingerprint invariance: same sample under instrument
               drift/aug must give same concentration (peak positions fixed
               by molecular structure; only intensities change)

    pw (physics weight 0..1): ramped from 0 so regression stabilises first.
    """
    mse = F.mse_loss(pred, true)
    z   = mse * 0.0   # zero on correct device, no allocation

    if pw <= 0:
        return {"mse": mse, "nonneg": z, "mass": z, "inv": z, "total": mse}

    zero = torch.tensor(zero_n, device=pred.device, dtype=torch.float32)
    cmax = torch.tensor(max_n,  device=pred.device, dtype=torch.float32)

    # ① Beer-Lambert: c >= 0  (negative concentration is unphysical)
    nonneg = F.relu(zero - pred).pow(2).mean() * pw

    # ② Mass balance: c <= cmax  (analyte cannot exceed total possible mass)
    mass   = F.relu(pred - cmax).pow(2).mean() * pw

    # ③ Raman fingerprint invariance
    inv    = F.mse_loss(pred_aug, pred.detach()) * pw if pred_aug is not None else z

    total = (LW_MSE * mse + LW_NONNEG_C * nonneg + LW_MASS * mass + LW_INV * inv)

    return {"mse": mse, "nonneg": nonneg, "mass": mass, "inv": inv, "total": total}


# ──────────────────────────────────────────────────────────────────────────────
# MAML meta-training
# ──────────────────────────────────────────────────────────────────────────────

def maml_inner_loop(model: MetaModel, Z_sup: torch.Tensor, y_sup: torch.Tensor,
                    inner_lr: float, inner_steps: int,
                    zero_n, max_n) -> MetaModel:
    """
    MAML inner loop: clone model, take `inner_steps` gradient steps on support set,
    return the adapted clone (original model unchanged).
    Uses LayerNorm instead of BatchNorm because inner-loop batch size is small.
    """
    fast_model = copy.deepcopy(model)
    fast_opt   = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        fast_model.train()
        pred  = fast_model(Z_sup)
        ld    = physics_loss(pred, y_sup, None, zero_n, max_n, pw=0.0)  # MSE only in inner
        fast_opt.zero_grad()
        ld["total"].backward()
        fast_opt.step()

    return fast_model


def meta_train(model: MetaModel,
               device_z: List[Tuple[np.ndarray, np.ndarray]],
               device: torch.device) -> MetaModel:
    """
    MAML outer loop over device tasks.

    device_z: list of (Z, y) where Z is already PCA-projected + preprocessed.
    Each episode:
    1. Sample a random device
    2. Sample K_SUPPORT + K_QUERY examples
    3. Inner loop (inner_steps SGD): adapt shared init to support set
    4. Outer loop: meta-gradient from query loss → update shared init

    Key design: meta-learning irons out per-device spectral offsets so the
    shared encoder learns concentration-relevant features only.
    Physics loss in outer loop keeps predictions in the valid physical domain.
    """
    meta_opt = torch.optim.Adam(model.parameters(), lr=META_LR,
                                weight_decay=META_WEIGHT_DECAY)
    # Global TargetScaler across all device data (for physics bounds in meta-training)
    all_y    = np.concatenate([y for _, y in device_z if y is not None], axis=0)
    gts      = TargetScaler().fit(all_y)
    zero_n   = gts.zero_norm()
    max_n    = gts.max_norm(CONC_MAX_RAW)

    n_devices = len(device_z)
    rng       = np.random.RandomState(SEED)

    print(f"\n{'='*70}")
    print(f"  MAML Meta-Training  ({META_EPOCHS} epochs × {META_TASKS_PER_EPOCH} tasks)")
    print(f"  Source: {n_devices} devices   Support K={K_SUPPORT}  Query K={K_QUERY}")
    print(f"{'='*70}")

    for epoch in range(1, META_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(META_TASKS_PER_EPOCH):
            # Sample a device task (Z already PCA-projected — no double transform)
            dev_idx      = rng.randint(0, n_devices)
            Z_dev, y_dev = device_z[dev_idx]
            if len(Z_dev) < K_SUPPORT + K_QUERY:
                continue

            # Normalise targets per-episode
            y_dev_n  = gts.transform(y_dev)

            # Sample support + query (random, no overlap)
            idx      = rng.permutation(len(Z_dev))
            sup_idx  = idx[:K_SUPPORT]
            qry_idx  = idx[K_SUPPORT:K_SUPPORT + K_QUERY]

            Z_sup = torch.tensor(Z_dev[sup_idx],   dtype=torch.float32, device=device)
            y_sup = torch.tensor(y_dev_n[sup_idx], dtype=torch.float32, device=device)
            Z_qry = torch.tensor(Z_dev[qry_idx],   dtype=torch.float32, device=device)
            y_qry = torch.tensor(y_dev_n[qry_idx], dtype=torch.float32, device=device)

            # Inner loop: adapt to support set
            fast_model = maml_inner_loop(model, Z_sup, y_sup,
                                         INNER_LR, INNER_STEPS, zero_n, max_n)

            # Outer loop: evaluate adapted model on query set
            fast_model.train()
            pred_qry = fast_model(Z_qry)
            # Light augmentation during meta-training for invariance
            Z_qry_np  = Z_qry.cpu().numpy()
            Z_qry_aug = torch.tensor(
                augment_pca(Z_qry_np, rng, scale=(0.90,1.10), noise=0.03),
                dtype=torch.float32, device=device)
            pred_aug  = fast_model(Z_qry_aug)

            pw   = min(1.0, epoch / META_EPOCHS)   # ramp physics weight over training
            ld   = physics_loss(pred_qry, y_qry, pred_aug, zero_n, max_n, pw=pw * 0.5)

            meta_opt.zero_grad()
            ld["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            meta_opt.step()

            epoch_loss += ld["total"].item()

        avg_loss = epoch_loss / META_TASKS_PER_EPOCH
        if epoch % 20 == 0:
            print(f"  meta epoch {epoch:4d}  avg query loss = {avg_loss:.4f}")

    print("  Meta-training complete.\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning on transfer plate
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune_once(meta_model: MetaModel,
                   Z_tr: np.ndarray, y_tr_raw: np.ndarray,
                   Z_val: np.ndarray, y_val_raw: np.ndarray,
                   device: torch.device, seed: int) -> Tuple[MetaModel, float]:
    """
    Fine-tune a clone of meta_model on the training fold.
    Strategy: first unfreeze the head only, then unfreeze the full encoder.
    """
    set_seed(seed)
    model = copy.deepcopy(meta_model).to(device)

    tscaler = TargetScaler().fit(y_tr_raw)
    y_tr    = tscaler.transform(y_tr_raw)
    zero_n  = tscaler.zero_norm()
    max_n   = tscaler.max_norm(CONC_MAX_RAW)

    Z_tr_t  = torch.tensor(Z_tr, dtype=torch.float32)
    Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)
    rng     = np.random.RandomState(seed)

    ds = TensorDataset(Z_tr_t, torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, num_workers=0)

    # Phase A: head only (first 30% of epochs)
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FT_LR * 2, weight_decay=FT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=60, T_mult=2, eta_min=1e-5)

    best_r2, best_state = -np.inf, None
    patience = 0
    phase_b_started = False

    for epoch in range(1, FT_EPOCHS + 1):

        # Phase B: unfreeze encoder at 30% of epochs
        if epoch == max(1, int(FT_EPOCHS * 0.30)) and not phase_b_started:
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=60, T_mult=2, eta_min=1e-5)
            phase_b_started = True

        pw = 0.0
        if epoch > PHYS_START_FT:
            pw = min(1.0, (epoch - PHYS_START_FT) /
                          max(PHYS_FULL_FT - PHYS_START_FT, 1))

        model.train()
        ep_losses = {"mse": 0., "nonneg": 0., "mass": 0., "inv": 0.}
        for z_b, y_b in dl:
            z_b, y_b = z_b.to(device), y_b.to(device)
            z_aug = torch.tensor(
                augment_pca(z_b.cpu().numpy(), rng),
                dtype=torch.float32, device=device)
            pred     = model(z_b)
            pred_aug = model(z_aug)
            ld   = physics_loss(pred, y_b, pred_aug, zero_n, max_n, pw)
            opt.zero_grad()
            ld["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in ep_losses:
                ep_losses[k] += ld[k].item()
        sched.step(epoch)

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pn = model(Z_val_t).cpu().numpy()
            p_raw = np.maximum(tscaler.inverse(pn), 0.0)
            vr2   = r2_score(y_val_raw, p_raw)
            if vr2 > best_r2:
                best_r2    = vr2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
            if epoch % 50 == 0:
                nb = max(len(dl), 1)
                phys_str = (f"mse={ep_losses['mse']/nb:.4f}  "
                            f"nonneg={ep_losses['nonneg']/nb:.4f}  "
                            f"mass={ep_losses['mass']/nb:.4f}  "
                            f"inv={ep_losses['inv']/nb:.4f}")
                print(f"    ep {epoch:4d} | {phys_str} | val R²={vr2:.4f}"
                      + (" [phys ON]" if pw > 0 else " [warmup]"))
            if patience >= PATIENCE:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r2, tscaler


@torch.no_grad()
def predict_tta(model, Z, tscaler, device, n=TTA_N):
    model.eval()
    rng   = np.random.RandomState(99)
    preds = [tscaler.inverse(
        model(torch.tensor(Z, dtype=torch.float32, device=device)).cpu().numpy())]
    for _ in range(n - 1):
        Z_aug = augment_pca(Z, rng, scale=(0.94,1.06), noise=AUG_NOISE * 0.3)
        preds.append(tscaler.inverse(
            model(torch.tensor(Z_aug, dtype=torch.float32, device=device)).cpu().numpy()))
    return np.maximum(np.mean(preds, 0), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def cv_fine_tune(meta_model, Z_train, y_train, Z_test, device):
    kf         = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_OUT), dtype=np.float32)
    fold_r2s   = []

    print(f"\n{'='*70}")
    print(f"  Fine-tuning  ({N_FOLDS}-fold × {N_RESTARTS} restarts)")
    print(f"{'='*70}")
    tnames = ["Glucose","NaAc","MgAc"]

    for fold, (tr_idx, val_idx) in enumerate(kf.split(Z_train)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────")
        Z_tr, Z_val   = Z_train[tr_idx], Z_train[val_idx]
        y_tr, y_val   = y_train[tr_idx],  y_train[val_idx]

        best_model, best_r2, best_tscaler = None, -np.inf, None

        for restart in range(N_RESTARTS):
            m, vr2, ts = fine_tune_once(
                meta_model, Z_tr, y_tr, Z_val, y_val, device,
                seed=SEED + fold*100 + restart)
            print(f"  restart {restart+1}/{N_RESTARTS}  val R²={vr2:.4f}"
                  + (" ← best" if vr2 > best_r2 else ""))
            if vr2 > best_r2:
                best_r2, best_model, best_tscaler = vr2, m, ts

        oof_preds[val_idx] = predict_tta(best_model, Z_val, best_tscaler, device)
        test_preds        += predict_tta(best_model, Z_test, best_tscaler, device) / N_FOLDS

        r2s = [r2_score(y_val[:,i], oof_preds[val_idx,i]) for i in range(3)]
        avg = np.mean(r2s)
        fold_r2s.append(avg)
        print(f"  Fold {fold+1}: "
              + "  ".join(f"{n}={v:.4f}" for n,v in zip(tnames,r2s))
              + f"  Avg={avg:.4f}")

    return oof_preds, test_preds, fold_r2s


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process(preds, y_train):
    names = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    out   = np.maximum(preds, 0.0)
    for i, n in enumerate(names):
        lo = max(0.0, np.percentile(y_train[:,i], 1))
        hi = np.percentile(y_train[:,i], 99)
        mg = 0.12*(hi-lo)
        out[:,i] = np.clip(out[:,i], max(0, lo-mg), hi+mg)
        print(f"  {n}: pred [{preds[:,i].min():.2f},{preds[:,i].max():.2f}]"
              f" → clipped [{max(0,lo-mg):.2f},{hi+mg:.2f}]")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda"  if torch.cuda.is_available()         else
                          "mps"   if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Device: {device}")

    wns = make_grid(WN_LOW, WN_HIGH, WN_STEP)
    print(f"Grid: {WN_LOW}–{WN_HIGH} cm⁻¹  ({len(wns)} bins)")

    # ── Load transfer plate + test ───────────────────────────────────────────
    print("\nLoading transfer plate + test …")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR,"transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR,"96_samples.csv"),     False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Transfer plate: {Xtr.shape}  Labels: {y_train.shape}")
    print(f"  Test:           {Xte.shape}")

    # ── Load device datasets (source domains) ────────────────────────────────
    print("\nLoading device datasets (source domains for meta-learning) …")
    device_data = []
    total_dev_samples = 0
    for fname in DEVICE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue
        X_dev, y_dev = load_device(path, wns)
        if X_dev is None or len(X_dev) < K_SUPPORT + K_QUERY:
            print(f"  [SKIP] {fname} — too few samples or no grid overlap")
            continue
        device_data.append((X_dev, y_dev))
        total_dev_samples += len(X_dev)
        print(f"  {fname:<25}  {len(X_dev):4d} samples  "
              f"y range Glu=[{y_dev[:,0].min():.1f},{y_dev[:,0].max():.1f}]")

    if len(device_data) == 0:
        print("\n  [WARNING] No device files found. Running fine-tune-only mode.")
        print("  Place device CSV files in DATA_DIR to enable meta-learning.")
        # Fall back to fine-tune only (better than nothing)
        use_meta = False
    else:
        use_meta = True
        print(f"\n  Total device samples: {total_dev_samples}")

    # ── Fit spectral preprocessor on ALL available spectra ──────────────────
    print("\nFitting spectral preprocessor …")
    prep = SpectralPreprocessor()
    # Fit on transfer plate (always available) + device data if available
    if use_meta:
        all_Xraw = np.concatenate(
            [Xtr] + [X for X, _ in device_data], axis=0)
    else:
        all_Xraw = Xtr
    prep.fit_transform(all_Xraw)          # sets msc_ref and scaler
    # Now transform each split consistently
    Xtr_pp   = prep.transform(Xtr)
    Xte_pp   = prep.transform(Xte)
    if use_meta:
        device_pp = [(prep.transform(X), y) for X, y in device_data]

    # ── PCA — fit on transfer plate + device data ────────────────────────────
    print(f"\nPCA({N_PCA}) …")
    pca = PCA(n_components=N_PCA, random_state=SEED)
    if use_meta:
        all_Xpp = np.concatenate([Xtr_pp] + [X for X,_ in device_pp], axis=0)
    else:
        all_Xpp = Xtr_pp
    pca.fit(all_Xpp)
    Z_train = pca.transform(Xtr_pp).astype(np.float32)
    Z_test  = pca.transform(Xte_pp).astype(np.float32)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {var_exp:.2f}%")
    if use_meta:
        device_z = [(pca.transform(X).astype(np.float32), y)
                    for X, y in device_pp]

    # Print label statistics for sanity check
    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    print("\nTransfer plate label stats:")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28} mean={y_train[:,i].mean():.2f}  "
              f"[{y_train[:,i].min():.2f},{y_train[:,i].max():.2f}]")

    # ── Phase 1: MAML meta-training on device data ───────────────────────────
    meta_model = MetaModel(N_PCA).to(device)
    print(f"\nModel params: encoder={sum(p.numel() for p in meta_model.encoder.parameters()):,}"
          f"  head={sum(p.numel() for p in meta_model.head.parameters()):,}")

    if use_meta:
        meta_model = meta_train(meta_model, device_z, device)
        torch.save(meta_model.state_dict(),
                   os.path.join(OUT_DIR, "meta_model_weights.pt"))
        print("  Meta-model weights saved.")
    else:
        print("\n  Skipping meta-training (no device files). Using random init.")

    # ── Phase 2: Cross-validated fine-tuning on transfer plate ───────────────
    oof_preds, test_preds, fold_r2s = cv_fine_tune(
        meta_model, Z_train, y_train, Z_test, device)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL OOF  — Meta-Learning PINN v5")
    print("="*70)
    oof_r2s = [r2_score(y_train[:,i], oof_preds[:,i]) for i in range(N_OUT)]
    for n, r2 in zip(tnames_full, oof_r2s):
        print(f"  {n:<28}  R² = {r2:.4f}")
    print(f"  {'Overall':<28}  R² = {np.mean(oof_r2s):.4f}")
    print(f"\n  Per-fold: " + "  ".join(f"{v:.4f}" for v in fold_r2s))
    print(f"  CV {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

    # Prediction distribution sanity check
    print("\nPrediction distribution vs training labels:")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28} pred mean={test_preds[:,i].mean():.2f}  "
              f"train mean={y_train[:,i].mean():.2f}")

    # ── Post-process + save ──────────────────────────────────────────────────
    print("\nPost-processing …")
    test_final = post_process(test_preds, y_train)

    sub = pd.DataFrame({
        "ID":                np.arange(1, len(test_final)+1),
        "Glucose":           test_final[:,0],
        "Sodium Acetate":    test_final[:,1],
        "Magnesium Sulfate": test_final[:,2],
    })
    out_path = os.path.join(OUT_DIR, "submission_pinn_v5.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission → {out_path}")
    print(sub.head(10).to_string(index=False))

    return sub


if __name__ == "__main__":
    main()
