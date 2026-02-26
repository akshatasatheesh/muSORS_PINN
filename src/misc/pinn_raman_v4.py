#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Physics-Informed Neural Network v4  — Raman Spectroscopy                   ║
║  DIG4BIO Raman Transfer Learning Challenge                                   ║
║                                                                              ║
║  Core lesson from v1/v2/v3 failure:                                          ║
║  ────────────────────────────────────                                        ║
║  v3 had 225,473 parameters for 77 training samples (1:3000 ratio).           ║
║  No physics constraint can fix that. The model memorised noise.              ║
║                                                                              ║
║  v4 design principle: RIGHT-SIZE FIRST, physics second.                      ║
║  ─────────────────────────────────────────────────────                       ║
║  Step 1 — PCA(50): 1643 dims → 50 components  (captures >99% variance)      ║
║           Params drop from 225K to 5,344.  Sample:param = 77:5344 = 1:69    ║
║                                                                              ║
║  Step 2 — Tiny MLP on PCA scores:                                            ║
║           50 → 64 → 32 → 3    (all with dropout 0.5 + BatchNorm)            ║
║                                                                              ║
║  Step 3 — Physics as LIGHT regularisation (not a second objective):          ║
║           • Non-negativity penalty  (c ≥ 0,  Beer-Lambert)                  ║
║           • Mass balance penalty    (c ≤ physical ceiling)                  ║
║           • Fingerprint invariance  (augmentation-stability loss)            ║
║           NO Beer-Lambert decoder — pure_spectra (3×1643=4929 params)       ║
║           removed entirely. Physics encoded in loss only.                    ║
║                                                                              ║
║  Step 4 — 10 random restarts per fold, keep best val R².                    ║
║           Avoids bad local minima without extra data.                        ║
║                                                                              ║
║  Step 5 — Final blend: 0.35×PINN + 0.65×HGB (best known baseline).         ║
║           The blend file is written automatically if HGB csv is present.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"
OUT_DIR    = "./pinn_outputs"
HGB_CSV    = "submission_pp_hgb_7_2_0.csv"   # best baseline for blending
SEED       = 42

WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0   # 1643 bins

# PCA — key dimensionality reduction
N_PCA      = 50    # 50 components capture >99% of spectral variance

# Architecture — intentionally tiny for 96 samples
HIDDEN1    = 64
HIDDEN2    = 32
N_TARGETS  = 3
DROPOUT    = 0.50   # high dropout because sample count is tiny

# Training
N_FOLDS    = 5
EPOCHS     = 400
BATCH_SIZE = 24    # smaller → more updates per epoch
LR         = 1e-3
WEIGHT_DECAY = 1e-3
N_RESTARTS = 8     # random restarts per fold; keeps best by val R²

# Physics loss (ramp in after PHYS_START epochs)
PHYS_START = 40
PHYS_FULL  = 120
LW_MSE        = 1.00
LW_NONNEG_C   = 0.20   # c ≥ 0   (Beer-Lambert non-negativity)
LW_MASS       = 0.15   # c ≤ cmax (mass balance)
LW_INV        = 0.08   # augmentation invariance (fingerprint stability)

# Physical concentration ceilings (g/L)
CONC_MAX_RAW = np.array([15.0, 3.0, 4.0], dtype=np.float32)

# Blend weights with HGB baseline
BLEND_PINN = 0.35
BLEND_HGB  = 0.65

# Augmentation (applied in PCA space — cheaper and equally valid)
AUG_SCALE     = (0.88, 1.12)
AUG_NOISE_STD = 0.05    # noise in PCA score space

TTA_N = 20


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Target scaler — normalise y to z-scores during training
# ──────────────────────────────────────────────────────────────────────────────

class TargetScaler:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, y):
        self.mean = y.mean(0).astype(np.float32)
        self.std  = np.maximum(y.std(0), 1e-6).astype(np.float32)
        return self

    def transform(self, y):
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y_norm):
        return (y_norm * self.std + self.mean).astype(np.float32)

    def zero_norm(self):
        """0 g/L in normalised space — used for non-negativity penalty."""
        return (-self.mean / self.std).astype(np.float32)

    def max_norm(self, cmax):
        """Physical ceiling in normalised space."""
        return ((cmax - self.mean) / self.std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _clean(s):
    return pd.to_numeric(
        s.astype(str).str.replace("[","",regex=False).str.replace("]","",regex=False),
        errors="coerce")


def load_plate(path, is_train):
    if is_train:
        df    = pd.read_csv(path)
        tcols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y     = df[tcols].dropna().values.astype(np.float32)
        Xdf   = df.iloc[:, :-4].copy()
    else:
        df    = pd.read_csv(path, header=None)
        y     = None
        Xdf   = df.copy()

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
    sel  = (full >= lo) & (full <= hi)
    return np.array(
        [np.interp(grid_wns, full[sel], row) for row in X2048[:, sel].astype(np.float64)],
        dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Spectral preprocessing (fit once on full training set)
# ──────────────────────────────────────────────────────────────────────────────

def apply_msc(X, ref=None):
    X64 = X.astype(np.float64)
    ref = X64.mean(0) if ref is None else ref.astype(np.float64)
    out = np.empty_like(X64)
    for i in range(len(X64)):
        m, b = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - b) / (m + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)


def apply_snip(X):
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
        t = np.linspace(0, 1, X.shape[1])
        out = np.empty_like(X, dtype=np.float64)
        for i, row in enumerate(X.astype(np.float64)):
            out[i] = row - np.polyval(np.polyfit(t, row, 3), t)
        return out.astype(np.float32)


class SpectralPreprocessor:
    """MSC → SNIP → SavGol → StandardScaler  (fit on training data only)."""
    def __init__(self):
        self.msc_ref = None
        self.scaler  = None

    def fit_transform(self, X):
        X, self.msc_ref = apply_msc(X)
        X = apply_snip(X)
        X = savgol_filter(X.astype(np.float64), 7, 2, 0, axis=1).astype(np.float32)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X).astype(np.float32)

    def transform(self, X):
        X, _ = apply_msc(X, ref=self.msc_ref)
        X = apply_snip(X)
        X = savgol_filter(X.astype(np.float64), 7, 2, 0, axis=1).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation in PCA score space (fast, avoids wavenumber interpolation loop)
# ──────────────────────────────────────────────────────────────────────────────

def augment_pca(Z: np.ndarray, rng: np.random.RandomState,
                scale=(0.88, 1.12), noise_std=0.05) -> np.ndarray:
    """
    Augment in PCA-score space.
    Physics motivation:
    - Scale: Beer-Lambert intensity variation (path length)
    - Noise: instrument noise propagated through PCA projection
    """
    Z = Z.copy()
    Z *= rng.uniform(*scale, (len(Z), 1)).astype(np.float32)
    if noise_std > 0:
        Z += rng.normal(0, noise_std, Z.shape).astype(np.float32)
    return Z


# ──────────────────────────────────────────────────────────────────────────────
# Model — 3-layer MLP on PCA scores
# ──────────────────────────────────────────────────────────────────────────────

class TinyPINN(nn.Module):
    """
    3-layer MLP operating on PCA scores.

    Parameter count: 50×64 + 64×32 + 32×3 = 5,344
    For 77 training samples → ratio 1:69  (vs 1:3000 in v3)

    No Beer-Lambert decoder — physics encoded in loss only.
    Output: raw logits (target-scaled space).
    """

    def __init__(self, n_pca=50, h1=64, h2=32, n_out=3, dropout=0.50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pca, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),

            nn.Linear(h2, n_out),
        )
        # Initialise final layer with small weights — start near y_mean (=0 in normed space)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z)


# ──────────────────────────────────────────────────────────────────────────────
# Physics loss
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """
    Physics laws encoded as soft penalties:
    ① MSE regression (always)
    ② Non-negativity: c ≥ 0  (Beer-Lambert — no negative concentration)
    ③ Mass balance:   c ≤ cmax  (conservation of total analyte)
    ④ Invariance:     c(x) ≈ c(aug(x))  (Raman fingerprint peak positions fixed)

    All physics terms normalised to have comparable gradient magnitude to MSE.
    """

    def __init__(self, zero_norm: np.ndarray, max_norm: np.ndarray,
                 phys_start=40, phys_full=120):
        super().__init__()
        self.register_buffer("zero_n", torch.tensor(zero_norm, dtype=torch.float32))
        self.register_buffer("max_n",  torch.tensor(max_norm,  dtype=torch.float32))
        self.phys_start = phys_start
        self.phys_full  = phys_full

    def ramp(self, epoch):
        if epoch < self.phys_start:
            return 0.0
        return min(1.0, (epoch - self.phys_start) /
                        max(self.phys_full - self.phys_start, 1))

    def forward(self, pred, true, pred_aug, epoch):
        pw   = self.ramp(epoch)
        zero = self.zero_n.to(pred.device)
        cmax = self.max_n.to(pred.device)

        mse  = F.mse_loss(pred, true)

        if pw == 0.0:
            return {"mse": mse, "nonneg": mse*0, "mass": mse*0,
                    "inv": mse*0, "total": mse}

        nonneg = F.relu(zero - pred).pow(2).mean() * pw
        mass   = F.relu(pred - cmax).pow(2).mean() * pw
        inv    = F.mse_loss(pred_aug, pred.detach()) * pw

        total = (LW_MSE * mse + LW_NONNEG_C * nonneg +
                 LW_MASS * mass + LW_INV * inv)

        return {"mse": mse, "nonneg": nonneg, "mass": mass,
                "inv": inv, "total": total}


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_model(Z_tr, y_tr_norm, Z_val, y_val_raw,
                    tscaler, criterion, device, seed):
    """Train a single TinyPINN and return (model, best_val_r2)."""
    set_seed(seed)
    model = TinyPINN(N_PCA, HIDDEN1, HIDDEN2, N_TARGETS, DROPOUT).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=80, T_mult=2, eta_min=1e-5)

    ds = TensorDataset(torch.tensor(Z_tr), torch.tensor(y_tr_norm))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)
    rng     = np.random.RandomState(seed)

    best_r2, best_state = -np.inf, None
    patience = 0
    MAX_PAT  = 60   # patience in units of 5 epochs

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for z_b, y_b in dl:
            z_b, y_b = z_b.to(device), y_b.to(device)
            z_aug = torch.tensor(
                augment_pca(z_b.cpu().numpy(), rng,
                            scale=AUG_SCALE, noise_std=AUG_NOISE_STD),
                dtype=torch.float32, device=device)

            pred     = model(z_b)
            pred_aug = model(z_aug)
            loss_d   = criterion(pred, y_b, pred_aug, epoch)

            opt.zero_grad()
            loss_d["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step(epoch)

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p_norm = model(Z_val_t).cpu().numpy()
            p_raw = tscaler.inverse(p_norm)
            p_raw = np.maximum(p_raw, 0.0)
            vr2   = r2_score(y_val_raw, p_raw)

            if vr2 > best_r2:
                best_r2    = vr2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1

            if patience >= MAX_PAT:
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_r2


@torch.no_grad()
def predict_tta(model, Z: np.ndarray, tscaler, device, n=TTA_N):
    """TTA in PCA-score space, return g/L predictions."""
    model.eval()
    rng   = np.random.RandomState(99)
    preds = []

    p, _ = model(torch.tensor(Z, dtype=torch.float32, device=device)), None
    p    = model(torch.tensor(Z, dtype=torch.float32, device=device))
    preds.append(tscaler.inverse(p.cpu().numpy()))

    for _ in range(n - 1):
        Z_aug = augment_pca(Z, rng, scale=(0.94, 1.06), noise_std=AUG_NOISE_STD * 0.3)
        p = model(torch.tensor(Z_aug, dtype=torch.float32, device=device))
        preds.append(tscaler.inverse(p.cpu().numpy()))

    return np.maximum(np.mean(preds, 0), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process(preds, y_train):
    names = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    out   = np.maximum(preds, 0.0)
    for i, n in enumerate(names):
        lo = max(0.0, np.percentile(y_train[:, i], 1))
        hi = np.percentile(y_train[:, i], 99)
        mg = 0.12 * (hi - lo)
        out[:, i] = np.clip(out[:, i], max(0, lo - mg), hi + mg)
        print(f"  {n}: pred [{preds[:,i].min():.2f},{preds[:,i].max():.2f}] "
              f"→ clipped [{max(0,lo-mg):.2f},{hi+mg:.2f}]")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def train_cv(Z_all, y_train, Z_test, tnames_full, device):
    """
    For each fold:
      1. Fit TargetScaler on train split
      2. Run N_RESTARTS independent random-seed training runs
      3. Keep the run with best val R²
      4. Collect OOF + test predictions
    """
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_TARGETS), dtype=np.float32)
    fold_r2s   = []

    print(f"\n{'='*70}")
    print(f"  PINN v4  |  PCA({N_PCA}) → MLP({HIDDEN1}→{HIDDEN2}→3)")
    print(f"  {N_FOLDS}-fold CV  ×  {N_RESTARTS} restarts  |  device={device}")
    print(f"{'='*70}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(Z_all)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────────────────")

        Z_tr, Z_val   = Z_all[tr_idx], Z_all[val_idx]
        y_tr, y_val   = y_train[tr_idx], y_train[val_idx]

        tscaler = TargetScaler().fit(y_tr)
        y_tr_n  = tscaler.transform(y_tr)

        criterion = PhysicsLoss(
            zero_norm  = tscaler.zero_norm(),
            max_norm   = tscaler.max_norm(CONC_MAX_RAW),
            phys_start = PHYS_START,
            phys_full  = PHYS_FULL,
        ).to(device)

        # Run N_RESTARTS and keep the best
        best_model, best_r2 = None, -np.inf
        for restart in range(N_RESTARTS):
            m, vr2 = train_one_model(
                Z_tr, y_tr_n, Z_val, y_val,
                tscaler, criterion, device,
                seed=SEED + fold * 100 + restart)
            print(f"  restart {restart+1}/{N_RESTARTS}  val R²={vr2:.4f}"
                  + (" ← best" if vr2 > best_r2 else ""))
            if vr2 > best_r2:
                best_r2, best_model = vr2, m

        # OOF + test predictions using TTA
        oof_preds[val_idx] = predict_tta(best_model, Z_val, tscaler, device)
        test_preds        += predict_tta(best_model, Z_test, tscaler, device) / N_FOLDS

        r2s = [r2_score(y_val[:, i], oof_preds[val_idx, i]) for i in range(3)]
        avg = np.mean(r2s)
        fold_r2s.append(avg)
        short = ["Glu", "NaAc", "MgAc"]
        print(f"  Fold {fold+1} OOF: "
              + "  ".join(f"{n}={v:.4f}" for n, v in zip(short, r2s))
              + f"  →  Avg={avg:.4f}")

    return oof_preds, test_preds, fold_r2s


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

    # ── Load ────────────────────────────────────────────────────────────────
    print("\nLoading …")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"),     False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Train {Xtr.shape}  Labels {y_train.shape}")
    print(f"  Test  {Xte.shape}")

    # ── Spectral preprocessing ───────────────────────────────────────────────
    print("\nSpectral preprocessing …")
    prep = SpectralPreprocessor()
    Xtr_pp = prep.fit_transform(Xtr)
    Xte_pp = prep.transform(Xte)

    # ── PCA — the critical dimensionality reduction step ────────────────────
    print(f"\nPCA({N_PCA}) compression …")
    pca      = PCA(n_components=N_PCA, random_state=SEED)
    Z_train  = pca.fit_transform(Xtr_pp).astype(np.float32)
    Z_test   = pca.transform(Xte_pp).astype(np.float32)
    var_exp  = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {var_exp:.2f}%")
    print(f"  Z_train {Z_train.shape}   Z_test {Z_test.shape}")
    print(f"  Model params: {N_PCA*HIDDEN1 + HIDDEN1*HIDDEN2 + HIDDEN2*N_TARGETS:,}  "
          f"(for ~{int(len(y_train)*0.8)} train samples per fold)")

    # Print label stats
    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    print("\nLabel statistics:")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28}  mean={y_train[:,i].mean():.2f}  "
              f"std={y_train[:,i].std():.2f}  "
              f"[{y_train[:,i].min():.2f}, {y_train[:,i].max():.2f}]")

    # ── Cross-validated training ─────────────────────────────────────────────
    oof_preds, test_preds, fold_r2s = train_cv(
        Z_train, y_train, Z_test, tnames_full, device)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FINAL OOF  — PINN v4")
    print("="*70)
    oof_r2s = [r2_score(y_train[:, i], oof_preds[:, i]) for i in range(N_TARGETS)]
    for n, r2 in zip(tnames_full, oof_r2s):
        print(f"  {n:<28}  R² = {r2:.4f}")
    print(f"  {'Overall':<28}  R² = {np.mean(oof_r2s):.4f}")
    print(f"\n  Per-fold:  " + "  ".join(f"{v:.4f}" for v in fold_r2s))
    print(f"  CV mean ± std:  {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

    # ── Post-process ─────────────────────────────────────────────────────────
    print("\nPost-processing …")
    test_final = post_process(test_preds, y_train)

    print("\nPrediction distribution (should match training label range):")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28}  mean={test_final[:,i].mean():.2f}  "
              f"[{test_final[:,i].min():.2f}, {test_final[:,i].max():.2f}]")

    # ── Pure PINN submission ─────────────────────────────────────────────────
    sub_pinn = pd.DataFrame({
        "ID":                np.arange(1, len(test_final)+1),
        "Glucose":           test_final[:, 0],
        "Sodium Acetate":    test_final[:, 1],
        "Magnesium Sulfate": test_final[:, 2],
    })
    pinn_path = os.path.join(OUT_DIR, "submission_pinn_v4.csv")
    sub_pinn.to_csv(pinn_path, index=False)
    print(f"\nPINN submission → {pinn_path}")

    # ── Blend with HGB baseline ──────────────────────────────────────────────
    hgb_path = os.path.join(DATA_DIR, HGB_CSV)
    if not os.path.exists(hgb_path):
        # Also try current directory
        hgb_path = HGB_CSV
    if os.path.exists(hgb_path):
        print(f"\nBlending {BLEND_PINN:.0%} PINN + {BLEND_HGB:.0%} HGB …")
        sub_hgb = pd.read_csv(hgb_path).set_index("ID").reindex(sub_pinn["ID"]).reset_index()
        cols    = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]
        sub_blend = sub_pinn.copy()
        sub_blend[cols] = (sub_pinn[cols] * BLEND_PINN +
                           sub_hgb[cols]  * BLEND_HGB)
        blend_path = os.path.join(OUT_DIR, "submission_blend_v4.csv")
        sub_blend.to_csv(blend_path, index=False)
        print(f"Blend submission → {blend_path}")
        print(sub_blend.head(5).to_string(index=False))
    else:
        print(f"\n[INFO] HGB baseline not found at '{hgb_path}' — skipping blend.")
        print(f"       Copy {HGB_CSV} to {DATA_DIR} to enable blending.")

    print(sub_pinn.head(5).to_string(index=False))
    return sub_pinn


if __name__ == "__main__":
    main()
