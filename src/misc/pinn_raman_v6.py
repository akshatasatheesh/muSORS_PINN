#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v6  —  Fixed Meta-Learning + Per-Target Weighting           ║
║                                                                              ║
║  Critical fixes over v5:                                                     ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. MAML was completely broken: copy.deepcopy() severed the gradient graph   ║
║     so meta_opt.step() updated model with ZERO gradients. The rising meta   ║
║     loss (0.98→1.14) was pure weight decay degradation.                     ║
║     FIX: Replace with Reptile (correct, simple, no higher-order grads).     ║
║                                                                              ║
║  2. Physics losses (nonneg, mass) were always 0.0000 because predictions    ║
║     are in z-score space where violations can't occur.                       ║
║     FIX: Remove dead terms. Keep invariance only.                           ║
║                                                                              ║
║  3. No per-target loss weighting — Glucose (var ~10x NaAc) dominated MSE.  ║
║     FIX: Inverse-variance weighting so NaAc gets proportional signal.       ║
║                                                                              ║
║  4. PCA(60) may compress away NaAc-discriminative spectral bands.           ║
║     FIX: USE_FULL_SPECTRUM option to bypass PCA during fine-tuning.         ║
║                                                                              ║
║  5. TTA augmentation in PCA space added noise without helping.              ║
║     FIX: Reduced TTA or disabled.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
OUT_DIR   = "./v6_outputs"
SEED      = 42

# Shared wavenumber grid
WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0   # 1643 bins

DEVICE_FILES = [
    "anton_532.csv", "anton_785.csv", "kaiser.csv", "mettler_toledo.csv",
    "metrohm.csv", "tec5.csv", "timegate.csv", "tornado.csv",
]

# ══════════════════════════════════════════════════════════════════════════════
# KEY SWITCH: Use full spectrum (1643 features) instead of PCA(60)?
# Setting this True bypasses PCA for the fine-tune stage.
# The encoder becomes a small 1D network on the full preprocessed spectrum.
# ══════════════════════════════════════════════════════════════════════════════
USE_FULL_SPECTRUM = True    # <-- Set False to fall back to PCA(60) like v5

N_PCA = 60  # only used if USE_FULL_SPECTRUM = False

# Architecture
ENC_H1  = 256
ENC_H2  = 64
HEAD_H  = 32
N_OUT   = 3
DROPOUT = 0.25

# ── Reptile meta-training (replaces broken MAML)
META_EPOCHS          = 120
META_TASKS_PER_EPOCH = 16
K_SUPPORT            = 40       # more samples per task for stability
REPTILE_OUTER_LR     = 0.3     # interpolation rate toward adapted weights
INNER_LR             = 3e-3
INNER_STEPS          = 8       # more inner steps = better per-task fit
META_WEIGHT_DECAY    = 0.0     # no weight decay on meta — Reptile handles it

# ── Fine-tuning on transfer plate
FT_EPOCHS       = 400
FT_LR           = 3e-4
FT_WEIGHT_DECAY = 3e-4
FT_BATCH        = 24
FT_DROPOUT      = 0.35
N_FOLDS         = 5
N_RESTARTS      = 6
PATIENCE        = 60       # in units of 5 epochs

# Invariance regularisation (the only physics term that was actually working)
LW_INV          = 0.05

# Augmentation — gentler than v5
AUG_NOISE       = 0.02
AUG_SCALE       = (0.94, 1.06)

# TTA
TTA_N           = 12

CONC_MAX_RAW = np.array([15.0, 3.0, 4.0], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Target scaler  (unchanged from v5)
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


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (unchanged from v5)
# ──────────────────────────────────────────────────────────────────────────────

def _clean(s):
    return pd.to_numeric(
        s.astype(str).str.replace("[","",regex=False).str.replace("]","",regex=False),
        errors="coerce")


def load_device(path: str, grid_wns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    spec_cols = df.columns[:-5]
    wns_raw, valid_idx = [], []
    for i, c in enumerate(spec_cols):
        try:
            wns_raw.append(float(c)); valid_idx.append(i)
        except ValueError:
            pass

    wns_raw = np.array(wns_raw, dtype=np.float64)
    X_raw   = df.iloc[:, valid_idx].values.astype(np.float64)
    order   = np.argsort(wns_raw)
    wns_raw = wns_raw[order]
    X_raw   = X_raw[:, order]

    lo, hi  = float(grid_wns.min()), float(grid_wns.max())
    in_range = (wns_raw >= lo) & (wns_raw <= hi)
    if in_range.sum() < 10:
        return None, None

    X_interp = np.array(
        [np.interp(grid_wns, wns_raw[in_range], row[in_range]) for row in X_raw],
        dtype=np.float32)

    y = df.iloc[:, -5:-2].values[:, :3].astype(np.float32)
    valid = np.all(np.isfinite(y), axis=1)
    return X_interp[valid], y[valid]


def load_plate(path: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_train:
        df    = pd.read_csv(path)
        tcols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y     = df[tcols].dropna().values.astype(np.float32)
        Xdf   = df.iloc[:, :-4].copy()
    else:
        df = pd.read_csv(path, header=None); y = None; Xdf = df.copy()

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
# Spectral preprocessing  (unchanged from v5)
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
        out = np.empty_like(X, dtype=np.float64)
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
    def __init__(self):
        self.msc_ref = None; self.scaler = None

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
# Augmentation — gentler, works in whatever feature space we use
# ──────────────────────────────────────────────────────────────────────────────

def augment(Z, rng, scale=AUG_SCALE, noise=AUG_NOISE):
    Z = Z.copy()
    Z *= rng.uniform(*scale, (len(Z), 1)).astype(np.float32)
    if noise > 0:
        Z += rng.normal(0, noise, Z.shape).astype(np.float32)
    return Z


# ──────────────────────────────────────────────────────────────────────────────
# Model — flexible input dim (works for PCA or full spectrum)
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Shared feature extractor. Input dim is flexible."""
    def __init__(self, n_in, h1=ENC_H1, h2=ENC_H2, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1,   h2), nn.LayerNorm(h2), nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class Head(nn.Module):
    """Prediction head with per-target output."""
    def __init__(self, h2=ENC_H2, hh=HEAD_H, n_out=N_OUT, dropout=FT_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h2, hh), nn.LayerNorm(hh), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hh, n_out),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class MetaModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.encoder = Encoder(n_in)
        self.head    = Head()

    def forward(self, z):
        return self.head(self.encoder(z))


# ──────────────────────────────────────────────────────────────────────────────
# Per-target weighted MSE  (FIX #3 — NaAc gets proportional gradient signal)
# ──────────────────────────────────────────────────────────────────────────────

def weighted_mse(pred, true, target_weights):
    """
    MSE weighted per target so each analyte contributes equally.
    target_weights: tensor of shape (N_OUT,) — inverse variance weights.
    """
    per_target = ((pred - true) ** 2).mean(dim=0)  # (N_OUT,)
    return (per_target * target_weights).mean()


def compute_target_weights(y_train):
    """
    Inverse-variance weights so each target contributes ~equally to the loss.
    Computed on raw (un-normalized) labels, then used on z-scored predictions.
    Since z-scoring already normalizes variance to 1, these weights handle
    any residual imbalance from the training fold.
    """
    # After z-scoring, variances should be ~1, but the *predictability* differs.
    # We use inverse raw variance as a prior to upweight NaAc.
    var = np.maximum(y_train.var(0), 1e-8)
    inv_var = 1.0 / var
    # Normalize so mean weight = 1
    w = inv_var / inv_var.mean()
    print(f"  Target weights: Glu={w[0]:.3f}  NaAc={w[1]:.3f}  MgAc={w[2]:.3f}")
    return w.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Loss function — simplified, only terms that actually work
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(pred, true, pred_aug, target_weights_t, pw):
    """
    Simplified loss:
      - Weighted MSE (primary) — per-target balanced
      - Invariance (secondary) — augmented input should give same output
    
    Removed nonneg and mass terms because they were always 0.0000 in v5.
    (Predictions in z-score space never violate raw concentration bounds.)
    """
    mse = weighted_mse(pred, true, target_weights_t)
    
    inv = torch.tensor(0.0, device=pred.device)
    if pred_aug is not None and pw > 0:
        inv = F.mse_loss(pred_aug, pred.detach()) * pw

    total = mse + LW_INV * inv
    return {"mse": mse, "inv": inv, "total": total}


# ──────────────────────────────────────────────────────────────────────────────
# Reptile meta-training  (FIX #1 — replaces broken MAML)
# ──────────────────────────────────────────────────────────────────────────────
#
# Why Reptile instead of MAML:
#   v5's MAML used copy.deepcopy() which severs the computational graph.
#   The meta-optimizer was updating the original model with ZERO gradients.
#   The rising meta loss (0.98→1.14) was just weight decay degrading params.
#
#   Reptile is correct by construction: no second-order gradients needed.
#   After K inner steps on a task, move the meta-weights toward the adapted
#   weights by interpolation: θ ← θ + ε(θ' - θ).
#
# ──────────────────────────────────────────────────────────────────────────────

def reptile_inner_loop(model, Z_sup, y_sup, inner_lr, inner_steps):
    """
    Train a CLONE of the model on one task's support set.
    Return the adapted state_dict (no gradient connection needed).
    """
    clone = copy.deepcopy(model)
    opt   = torch.optim.SGD(clone.parameters(), lr=inner_lr, momentum=0.9)
    
    clone.train()
    for _ in range(inner_steps):
        pred = clone(Z_sup)
        loss = F.mse_loss(pred, y_sup)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return clone.state_dict(), loss.item()


def meta_train(model, device_z, device):
    """
    Reptile meta-training across device tasks.
    
    For each episode:
      1. Sample a device, sample K_SUPPORT examples
      2. Clone model, train clone for INNER_STEPS on the support set
      3. Interpolate: θ ← θ + REPTILE_OUTER_LR * (θ' - θ)
    
    This correctly finds an initialization that can quickly adapt
    to any device — which is exactly what we need for transfer to
    the multiplexer.
    """
    # Global target scaler for normalizing labels across devices
    all_y = np.concatenate([y for _, y in device_z], axis=0)
    gts   = TargetScaler().fit(all_y)
    
    n_devices = len(device_z)
    rng = np.random.RandomState(SEED)
    
    print(f"\n{'='*70}")
    print(f"  Reptile Meta-Training  ({META_EPOCHS} epochs × {META_TASKS_PER_EPOCH} tasks)")
    print(f"  Source: {n_devices} devices   K={K_SUPPORT}  inner_steps={INNER_STEPS}")
    print(f"  outer_lr={REPTILE_OUTER_LR}  inner_lr={INNER_LR}")
    print(f"{'='*70}")
    
    for epoch in range(1, META_EPOCHS + 1):
        epoch_loss = 0.0
        
        # Decay outer LR over training (cosine schedule)
        frac = epoch / META_EPOCHS
        outer_lr = REPTILE_OUTER_LR * (0.5 * (1 + np.cos(np.pi * frac)))
        
        for _ in range(META_TASKS_PER_EPOCH):
            # Sample a device
            dev_idx = rng.randint(0, n_devices)
            Z_dev, y_dev = device_z[dev_idx]
            
            if len(Z_dev) < K_SUPPORT:
                continue
            
            # Normalize targets
            y_dev_n = gts.transform(y_dev)
            
            # Sample support set
            idx     = rng.permutation(len(Z_dev))[:K_SUPPORT]
            Z_sup   = torch.tensor(Z_dev[idx],   dtype=torch.float32, device=device)
            y_sup   = torch.tensor(y_dev_n[idx], dtype=torch.float32, device=device)
            
            # Inner loop: adapt clone to this device's support set
            adapted_state, task_loss = reptile_inner_loop(
                model, Z_sup, y_sup, INNER_LR, INNER_STEPS)
            
            # Reptile update: θ ← θ + outer_lr * (θ_adapted - θ)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    adapted_param = adapted_state[name].to(device)
                    param.data += outer_lr * (adapted_param - param.data)
            
            epoch_loss += task_loss
        
        avg_loss = epoch_loss / META_TASKS_PER_EPOCH
        if epoch % 20 == 0:
            print(f"  epoch {epoch:4d}  avg task loss = {avg_loss:.4f}  outer_lr = {outer_lr:.4f}")
    
    print("  Meta-training complete.\n")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning on transfer plate
# ──────────────────────────────────────────────────────────────────────────────

def fine_tune_once(meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw,
                   device, target_weights, seed):
    """Fine-tune a clone of meta_model on one CV fold."""
    set_seed(seed)
    model = copy.deepcopy(meta_model).to(device)
    
    tscaler = TargetScaler().fit(y_tr_raw)
    y_tr    = tscaler.transform(y_tr_raw)
    
    # Target weights in z-score space: since std normalization makes all
    # target variances ~1, we apply the raw inverse-variance weights as
    # a prior to ensure NaAc isn't drowned out.
    tw_t = torch.tensor(target_weights, dtype=torch.float32, device=device)
    
    Z_tr_t  = torch.tensor(Z_tr, dtype=torch.float32)
    Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)
    rng     = np.random.RandomState(seed)
    
    ds = TensorDataset(Z_tr_t, torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, num_workers=0)
    
    # Phase A: head only (first 20% of epochs)
    phase_switch = max(1, int(FT_EPOCHS * 0.20))
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FT_LR * 3, weight_decay=FT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=80, T_mult=2, eta_min=1e-6)
    
    best_r2, best_state = -np.inf, None
    patience_count = 0
    phase_b_started = False
    
    for epoch in range(1, FT_EPOCHS + 1):
        # Phase B: unfreeze encoder
        if epoch == phase_switch and not phase_b_started:
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                model.parameters(), lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=80, T_mult=2, eta_min=1e-6)
            phase_b_started = True
        
        # Physics weight ramp for invariance
        pw = 0.0
        if epoch > 50:
            pw = min(1.0, (epoch - 50) / 100)
        
        model.train()
        ep_mse, ep_inv = 0., 0.
        for z_b, y_b in dl:
            z_b, y_b = z_b.to(device), y_b.to(device)
            
            pred = model(z_b)
            
            # Augmented prediction for invariance
            pred_aug = None
            if pw > 0:
                z_aug = torch.tensor(
                    augment(z_b.cpu().numpy(), rng),
                    dtype=torch.float32, device=device)
                pred_aug = model(z_aug)
            
            ld = compute_loss(pred, y_b, pred_aug, tw_t, pw)
            opt.zero_grad()
            ld["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_mse += ld["mse"].item()
            ep_inv += ld["inv"].item()
        
        sched.step(epoch)
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pn = model(Z_val_t).cpu().numpy()
            p_raw = np.maximum(tscaler.inverse(pn), 0.0)
            vr2 = r2_score(y_val_raw, p_raw)
            if vr2 > best_r2:
                best_r2    = vr2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            
            if epoch % 50 == 0:
                nb = max(len(dl), 1)
                print(f"    ep {epoch:4d} | mse={ep_mse/nb:.4f}  inv={ep_inv/nb:.4f}"
                      f" | val R²={vr2:.4f}"
                      + (" [inv ON]" if pw > 0 else " [warmup]"))
            
            if patience_count >= PATIENCE:
                break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r2, tscaler


@torch.no_grad()
def predict_tta(model, Z, tscaler, device, n=TTA_N):
    """Test-time augmentation with gentle noise."""
    model.eval()
    rng   = np.random.RandomState(99)
    Z_t   = torch.tensor(Z, dtype=torch.float32, device=device)
    preds = [tscaler.inverse(model(Z_t).cpu().numpy())]
    for _ in range(n - 1):
        # Very gentle augmentation
        Z_aug = augment(Z, rng, scale=(0.97, 1.03), noise=AUG_NOISE * 0.3)
        Z_aug_t = torch.tensor(Z_aug, dtype=torch.float32, device=device)
        preds.append(tscaler.inverse(model(Z_aug_t).cpu().numpy()))
    return np.maximum(np.mean(preds, 0), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validated fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def cv_fine_tune(meta_model, Z_train, y_train, Z_test, device, target_weights):
    kf        = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_OUT), dtype=np.float32)
    fold_r2s  = []
    tnames    = ["Glucose", "NaAc", "MgAc"]
    
    print(f"\n{'='*70}")
    print(f"  Fine-tuning  ({N_FOLDS}-fold × {N_RESTARTS} restarts)")
    print(f"{'='*70}")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Z_train)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────")
        Z_tr, Z_val  = Z_train[tr_idx], Z_train[val_idx]
        y_tr, y_val  = y_train[tr_idx], y_train[val_idx]
        
        best_model, best_r2, best_tscaler = None, -np.inf, None
        
        for restart in range(N_RESTARTS):
            m, vr2, ts = fine_tune_once(
                meta_model, Z_tr, y_tr, Z_val, y_val, device,
                target_weights, seed=SEED + fold*100 + restart)
            print(f"  restart {restart+1}/{N_RESTARTS}  val R²={vr2:.4f}"
                  + (" ← best" if vr2 > best_r2 else ""))
            if vr2 > best_r2:
                best_r2, best_model, best_tscaler = vr2, m, ts
        
        oof_preds[val_idx] = predict_tta(best_model, Z_val, best_tscaler, device)
        test_preds += predict_tta(best_model, Z_test, best_tscaler, device) / N_FOLDS
        
        r2s = [r2_score(y_val[:, i], oof_preds[val_idx, i]) for i in range(3)]
        avg = np.mean(r2s)
        fold_r2s.append(avg)
        print(f"  Fold {fold+1}: "
              + "  ".join(f"{n}={v:.4f}" for n, v in zip(tnames, r2s))
              + f"  Avg={avg:.4f}")
    
    return oof_preds, test_preds, fold_r2s


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process(preds, y_train):
    names = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    out   = np.maximum(preds, 0.0)
    for i, n in enumerate(names):
        lo = max(0.0, np.percentile(y_train[:, i], 1))
        hi = np.percentile(y_train[:, i], 99)
        mg = 0.15 * (hi - lo)   # slightly wider margin than v5
        out[:, i] = np.clip(out[:, i], max(0, lo - mg), hi + mg)
        print(f"  {n}: pred [{preds[:,i].min():.2f},{preds[:,i].max():.2f}]"
              f" → clipped [{max(0,lo-mg):.2f},{hi+mg:.2f}]")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    device = torch.device("cuda"  if torch.cuda.is_available()       else
                          "mps"   if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Device: {device}")
    
    wns = make_grid(WN_LOW, WN_HIGH, WN_STEP)
    N_BINS = len(wns)
    print(f"Grid: {WN_LOW}–{WN_HIGH} cm⁻¹  ({N_BINS} bins)")
    
    # ── Load transfer plate + test
    print("\nLoading transfer plate + test …")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"),     False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Transfer plate: {Xtr.shape}  Labels: {y_train.shape}")
    print(f"  Test:           {Xte.shape}")
    
    # ── Load devices
    print("\nLoading device datasets …")
    device_data = []
    total_dev = 0
    for fname in DEVICE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found"); continue
        X_dev, y_dev = load_device(path, wns)
        if X_dev is None or len(X_dev) < K_SUPPORT:
            print(f"  [SKIP] {fname} — too few samples"); continue
        device_data.append((X_dev, y_dev))
        total_dev += len(X_dev)
        print(f"  {fname:<25}  {len(X_dev):4d} samples  "
              f"y range Glu=[{y_dev[:,0].min():.1f},{y_dev[:,0].max():.1f}]")
    
    use_meta = len(device_data) > 0
    if use_meta:
        print(f"\n  Total device samples: {total_dev}")
    else:
        print("\n  [WARNING] No device files found. Fine-tune only.")
    
    # ── Spectral preprocessing
    print("\nFitting spectral preprocessor …")
    prep = SpectralPreprocessor()
    if use_meta:
        all_Xraw = np.concatenate([Xtr] + [X for X, _ in device_data], axis=0)
    else:
        all_Xraw = Xtr
    prep.fit_transform(all_Xraw)
    Xtr_pp = prep.transform(Xtr)
    Xte_pp = prep.transform(Xte)
    if use_meta:
        device_pp = [(prep.transform(X), y) for X, y in device_data]
    
    # ── Feature preparation: PCA or full spectrum
    if USE_FULL_SPECTRUM:
        print(f"\nUsing FULL SPECTRUM ({N_BINS} features) — no PCA compression")
        # Just use the preprocessed spectra directly
        # Apply a per-feature StandardScaler for numerical stability
        feat_scaler = StandardScaler()
        if use_meta:
            all_pp = np.concatenate([Xtr_pp] + [X for X, _ in device_pp], axis=0)
        else:
            all_pp = Xtr_pp
        feat_scaler.fit(all_pp)
        
        Z_train = feat_scaler.transform(Xtr_pp).astype(np.float32)
        Z_test  = feat_scaler.transform(Xte_pp).astype(np.float32)
        n_features = N_BINS
        
        if use_meta:
            device_z = [(feat_scaler.transform(X).astype(np.float32), y)
                        for X, y in device_pp]
    else:
        print(f"\nPCA({N_PCA}) …")
        pca = PCA(n_components=N_PCA, random_state=SEED)
        if use_meta:
            all_pp = np.concatenate([Xtr_pp] + [X for X, _ in device_pp], axis=0)
        else:
            all_pp = Xtr_pp
        pca.fit(all_pp)
        Z_train = pca.transform(Xtr_pp).astype(np.float32)
        Z_test  = pca.transform(Xte_pp).astype(np.float32)
        n_features = N_PCA
        var_exp = pca.explained_variance_ratio_.sum() * 100
        print(f"  Variance explained: {var_exp:.2f}%")
        
        if use_meta:
            device_z = [(pca.transform(X).astype(np.float32), y)
                        for X, y in device_pp]
    
    print(f"\n  Feature dimension: {n_features}")
    
    # Label stats
    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    print("\nTransfer plate label stats:")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28} mean={y_train[:,i].mean():.2f}  "
              f"[{y_train[:,i].min():.2f},{y_train[:,i].max():.2f}]")
    
    # ── Compute per-target loss weights  (FIX #3)
    print("\nComputing per-target loss weights (inverse variance):")
    target_weights = compute_target_weights(y_train)
    
    # ── Build model
    meta_model = MetaModel(n_features).to(device)
    print(f"\nModel params: encoder={sum(p.numel() for p in meta_model.encoder.parameters()):,}"
          f"  head={sum(p.numel() for p in meta_model.head.parameters()):,}")
    
    # ── Reptile meta-training
    if use_meta:
        meta_model = meta_train(meta_model, device_z, device)
        torch.save(meta_model.state_dict(),
                   os.path.join(OUT_DIR, "meta_model_weights.pt"))
        print("  Meta-model weights saved.")
    else:
        print("\n  Skipping meta-training. Using random init.")
    
    # ── Cross-validated fine-tuning
    oof_preds, test_preds, fold_r2s = cv_fine_tune(
        meta_model, Z_train, y_train, Z_test, device, target_weights)
    
    # ── Summary
    print("\n" + "="*70)
    print("  FINAL OOF  — Raman Transfer v6")
    print("="*70)
    oof_r2s = [r2_score(y_train[:, i], oof_preds[:, i]) for i in range(N_OUT)]
    for n, r2 in zip(tnames_full, oof_r2s):
        print(f"  {n:<28}  R² = {r2:.4f}")
    print(f"  {'Overall':<28}  R² = {np.mean(oof_r2s):.4f}")
    print(f"\n  Per-fold: " + "  ".join(f"{v:.4f}" for v in fold_r2s))
    print(f"  CV {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    
    # Prediction distribution
    print("\nPrediction distribution vs training labels:")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28} pred mean={test_preds[:,i].mean():.2f}  "
              f"train mean={y_train[:,i].mean():.2f}")
        print(f"  {'':28} pred range=[{test_preds[:,i].min():.2f},{test_preds[:,i].max():.2f}]"
              f"  train range=[{y_train[:,i].min():.2f},{y_train[:,i].max():.2f}]")
    
    # ── Post-process + save
    print("\nPost-processing …")
    test_final = post_process(test_preds, y_train)
    
    sub = pd.DataFrame({
        "ID":                np.arange(1, len(test_final)+1),
        "Glucose":           test_final[:, 0],
        "Sodium Acetate":    test_final[:, 1],
        "Magnesium Sulfate": test_final[:, 2],
    })
    out_path = os.path.join(OUT_DIR, "submission_v6.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission → {out_path}")
    print(sub.head(10).to_string(index=False))
    
    return sub


if __name__ == "__main__":
    main()
