#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Physics-Informed Neural Network v3  — Raman Spectroscopy                   ║
║  DIG4BIO Raman Transfer Learning Challenge                                   ║
║                                                                              ║
║  FIXES vs v2 (R² = -0.3):                                                   ║
║  ──────────────────────────                                                  ║
║  ROOT CAUSE: model predicted mean=2.77 g/L Glucose vs actual mean=6.94 g/L  ║
║                                                                              ║
║  Fix 1 — TARGET SCALING (most impactful)                                    ║
║    Normalise y to mean=0, std=1 during training.                            ║
║    Inverse-transform predictions at inference.                               ║
║    Without this, Softplus + physics penalties pull predictions to ~0.        ║
║                                                                              ║
║  Fix 2 — REMOVE SOFTPLUS FROM HEAD                                          ║
║    Softplus is great for strict non-negativity but saturates at large values ║
║    (Glucose 4-12 g/L → logit must be ~4-12 → very hard to learn).           ║
║    Replaced by: raw linear head + target scaling handles the range.          ║
║    Non-negativity enforced by: (a) physics penalty (b) post-process clip.   ║
║                                                                              ║
║  Fix 3 — INIT HEAD BIAS TO TRAINING MEAN                                    ║
║    Head predicts deviation from mean; training starts near ground truth.    ║
║                                                                              ║
║  Fix 4 — PHYSICS CONSTRAINTS IN NORMALISED SPACE                           ║
║    CONC_MAX in g/L → converted to normalised units.                         ║
║    Reconstruction loss normalised by spectrum std → stable gradients.       ║
║                                                                              ║
║  Fix 5 — DISABLE BEER-LAMBERT UNTIL REGRESSION IS STABLE                   ║
║    Beer-Lambert decoder only activates at epoch 80.                         ║
║    pure_spectra frozen for first 80 epochs.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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

DATA_DIR  = "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"
OUT_DIR   = "./pinn_outputs"
SEED      = 42

WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0   # 1643 bins

# Architecture
HIDDEN_DIM   = 128
N_TARGETS    = 3
DROPOUT      = 0.30

# Training
N_FOLDS      = 5
EPOCHS       = 600
BATCH_SIZE   = 32
LR           = 8e-4
WEIGHT_DECAY = 5e-4

# Physics loss schedule (all in NORMALISED target space)
PHYS_START   = 80    # epoch to begin introducing physics
PHYS_FULL    = 200   # epoch at which physics weight reaches 1.0

# Physics loss weights — kept modest; regression is primary
LW_MSE        = 1.00
LW_BEER       = 0.04   # Beer-Lambert reconstruction
LW_NONNEG_C   = 0.15   # concentration non-negativity
LW_NONNEG_S   = 0.01   # reconstructed spectrum non-negativity
LW_SMOOTH     = 0.01   # spectral smoothness
LW_INV        = 0.06   # fingerprint invariance
LW_MASS       = 0.08   # mass balance (upper bound)
LW_PURE_SM    = 0.01   # pure component spectra smoothness

# Augmentation
AUG_SCALE     = (0.88, 1.12)
AUG_BASE_AMP  = 0.03
AUG_MAX_SHIFT = 1.5
AUG_NOISE_STD = 0.003

# Physical concentration ceilings (g/L) — will be normalised per fold
CONC_MAX_RAW  = np.array([15.0, 3.0, 4.0], dtype=np.float32)

TTA_N = 16

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(lo, hi, step) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Target Scaler — KEY FIX
# Normalise y to zero mean / unit std per target so the network never has to
# learn to output raw g/L values (0..12 range).  Inverse-transform at the end.
# ──────────────────────────────────────────────────────────────────────────────

class TargetScaler:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean = y.mean(axis=0).astype(np.float32)
        self.std  = np.maximum(y.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y_norm: np.ndarray) -> np.ndarray:
        return (y_norm * self.std + self.mean).astype(np.float32)

    def conc_max_normalised(self, conc_max_raw: np.ndarray) -> np.ndarray:
        """Convert g/L ceiling to normalised space for mass-balance penalty."""
        return ((conc_max_raw - self.mean) / self.std).astype(np.float32)

    def conc_zero_normalised(self) -> np.ndarray:
        """0 g/L in normalised space (for non-negativity penalty)."""
        return (-self.mean / self.std).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False),
        errors="coerce")


def load_plate(path: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_train:
        df    = pd.read_csv(path)
        tcols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y     = df[tcols].dropna().values.astype(np.float32)
        Xdf   = df.iloc[:, :-4].copy()
    else:
        df    = pd.read_csv(path, header=None)
        y     = None
        Xdf   = df.copy()

    Xdf.columns = ["sample_id"] + [str(i) for i in range(Xdf.shape[1] - 1)]
    Xdf["sample_id"] = Xdf["sample_id"].ffill()
    for col in Xdf.columns[1:]:
        Xdf[col] = _clean(Xdf[col])

    X = Xdf.drop(columns=["sample_id"]).values.astype(np.float32)
    X = X.reshape(-1, 2, 2048).mean(axis=1)
    return X, y


def plate_to_grid(X2048: np.ndarray, grid_wns: np.ndarray,
                  s=65.0, e=3350.0) -> np.ndarray:
    full = np.linspace(s, e, 2048, dtype=np.float64)
    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    sel  = (full >= lo) & (full <= hi)
    wsel = full[sel]
    Xsel = X2048[:, sel].astype(np.float64)
    return np.array([np.interp(grid_wns, wsel, r) for r in Xsel], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def apply_msc(X: np.ndarray, ref=None):
    X64 = X.astype(np.float64)
    ref = X64.mean(0) if ref is None else ref.astype(np.float64)
    out = np.empty_like(X64)
    for i in range(len(X64)):
        m, b = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - b) / (m + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)


def apply_snip(X: np.ndarray) -> np.ndarray:
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


def preprocess(X_raw: np.ndarray,
               msc_ref=None, scaler=None, fit=False):
    X, msc_ref = apply_msc(X_raw, ref=msc_ref)
    X = apply_snip(X)
    X = savgol_filter(X.astype(np.float64), window_length=7,
                      polyorder=2, deriv=0, axis=1).astype(np.float32)
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    return X, msc_ref, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment(X: np.ndarray, wns: np.ndarray, rng: np.random.RandomState,
            scale=(0.88, 1.12), amp=0.03, shift=1.5, noise=0.003) -> np.ndarray:
    X   = X.astype(np.float32).copy()
    wns = wns.astype(np.float32)
    N, D = X.shape

    X *= rng.uniform(*scale, (N, 1)).astype(np.float32)

    if amp > 0:
        t = np.linspace(0, 1, D, dtype=np.float32)
        for i in range(N):
            c = rng.normal(0, 1, 3)
            base = c[0] + c[1]*t + c[2]*t**2
            base = (base - base.mean()) / (base.std() + 1e-8)
            a = rng.uniform(-amp, amp) * float(np.abs(X[i]).max() + 1e-8)
            X[i] += a * base

    if shift > 0:
        sh  = rng.uniform(-shift, shift, N).astype(np.float32)
        out = np.empty_like(X)
        for i in range(N):
            out[i] = np.interp((wns - sh[i]).astype(np.float64),
                               wns.astype(np.float64), X[i].astype(np.float64))
        X = out.astype(np.float32)

    if noise > 0:
        X += rng.normal(0, noise, X.shape).astype(np.float32)

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Model — shallow MLP, regression-first design
# ──────────────────────────────────────────────────────────────────────────────

class RamanPINN(nn.Module):
    """
    Shallow MLP with Beer-Lambert physics decoder.

    KEY CHANGE vs v2:
    - Output is RAW linear (no Softplus at head).
      Targets are normalised → model predicts in z-score space.
      Non-negativity enforced by physics penalty + post-process clip.
    - Head bias initialised to 0 in z-score space (= training mean in g/L).
    - pure_spectra frozen until epoch PHYS_START (unfrozen by training loop).
    """

    def __init__(self, input_dim: int, hidden_dim=128, n_targets=3,
                 dropout=0.30, pure_spectra_init: Optional[np.ndarray] = None):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
        )

        # Raw linear output — no Softplus
        # Bias initialised to zero → predicts training mean in normalised space
        self.head = nn.Linear(hidden_dim // 4, n_targets)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

        # Beer-Lambert decoder
        if pure_spectra_init is not None:
            init = torch.tensor(
                np.tile(pure_spectra_init[None, :], (n_targets, 1)),
                dtype=torch.float32
            ) + 0.05 * torch.randn(n_targets, input_dim)
        else:
            init = torch.abs(torch.randn(n_targets, input_dim)) * 0.1
        self.pure_spectra = nn.Parameter(init)

        self._init_encoder()

    def _init_encoder(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_concentrations(self, x: torch.Tensor) -> torch.Tensor:
        """Raw linear output in normalised target space."""
        return self.head(self.encoder(x))

    def reconstruct_spectrum(self, c: torch.Tensor) -> torch.Tensor:
        """Beer-Lambert: x̂ = c · E  (in normalised concentration space)."""
        E = F.softplus(self.pure_spectra)   # keep pure spectra non-negative
        return c @ E

    def forward(self, x):
        c   = self.predict_concentrations(x)
        x_r = self.reconstruct_spectrum(c)
        return c, x_r


# ──────────────────────────────────────────────────────────────────────────────
# Physics-Informed Loss  (operates in normalised target space)
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    All physics constraints expressed in NORMALISED target space.
    conc_zero_norm : what 0 g/L looks like in z-score space (the non-neg threshold)
    conc_max_norm  : what CONC_MAX_RAW looks like in z-score space
    """

    def __init__(self, conc_zero_norm: np.ndarray, conc_max_norm: np.ndarray,
                 phys_start=80, phys_full=200):
        super().__init__()
        self.register_buffer("conc_zero", torch.tensor(conc_zero_norm, dtype=torch.float32))
        self.register_buffer("conc_max",  torch.tensor(conc_max_norm,  dtype=torch.float32))
        self.phys_start = phys_start
        self.phys_full  = phys_full

    def _ramp(self, epoch: int) -> float:
        if epoch < self.phys_start:
            return 0.0
        return min(1.0, (epoch - self.phys_start) /
                        max(self.phys_full - self.phys_start, 1))

    def _tv2(self, x): return (x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]).pow(2).mean()
    def _tv2r(self, x): return (x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]).pow(2).mean()

    def forward(self, pred_c, true_c, x_in, x_recon, pure_spec,
                epoch, pred_c_aug=None):
        pw = self._ramp(epoch)
        z  = torch.zeros(1, device=pred_c.device).squeeze()

        losses = {"mse": F.mse_loss(pred_c, true_c)}

        if pw <= 0:
            for k in ["beer", "nonneg_c", "nonneg_s", "smooth",
                      "mass", "pure_sm", "inv"]:
                losses[k] = z
        else:
            # Normalise reconstruction loss by input variance for stable scale
            spec_var = x_in.var().detach().clamp(min=1e-6)
            losses["beer"] = F.mse_loss(x_recon, x_in) / spec_var * pw

            # Non-negativity: penalise predictions below 0 g/L in normed space
            losses["nonneg_c"] = F.relu(self.conc_zero.to(pred_c.device) - pred_c).pow(2).mean() * pw
            losses["nonneg_s"] = F.relu(-x_recon).pow(2).mean() * pw

            losses["smooth"] = self._tv2(x_recon) * pw

            # Mass balance: penalise c > CONC_MAX in normed space
            losses["mass"]   = F.relu(pred_c - self.conc_max.to(pred_c.device)).pow(2).mean() * pw

            # Pure component smoothness
            E = F.softplus(pure_spec)
            losses["pure_sm"] = self._tv2r(E) * pw

            # Fingerprint invariance
            losses["inv"] = F.mse_loss(pred_c_aug, pred_c.detach()) * pw \
                            if pred_c_aug is not None else z

        losses["total"] = (
            LW_MSE     * losses["mse"]
          + LW_BEER    * losses["beer"]
          + LW_NONNEG_C* losses["nonneg_c"]
          + LW_NONNEG_S* losses["nonneg_s"]
          + LW_SMOOTH  * losses["smooth"]
          + LW_MASS    * losses["mass"]
          + LW_PURE_SM * losses["pure_sm"]
          + LW_INV     * losses["inv"]
        )
        return losses


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch, device, wns):
    model.train()
    totals: Dict[str, float] = {}
    rng = np.random.RandomState(epoch + SEED)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_np  = x.cpu().numpy()
        x_aug = torch.tensor(
            augment(x_np, wns, rng,
                    scale=AUG_SCALE, amp=AUG_BASE_AMP,
                    shift=AUG_MAX_SHIFT, noise=AUG_NOISE_STD),
            dtype=torch.float32, device=device)

        pred_c, x_recon   = model(x)
        pred_c_aug, _     = model(x_aug)

        losses = criterion(pred_c, y, x, x_recon,
                           model.pure_spectra, epoch, pred_c_aug)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    return {k: v/n for k, v in totals.items()}


@torch.no_grad()
def predict_tta(model, X_pp: np.ndarray, wns: np.ndarray,
                tscaler: TargetScaler, device, n=16) -> np.ndarray:
    """TTA in normalised space, inverse-transform to g/L."""
    model.eval()
    rng   = np.random.RandomState(0)
    preds = []

    c, _ = model(torch.tensor(X_pp, dtype=torch.float32, device=device))
    preds.append(tscaler.inverse(c.cpu().numpy()))

    for _ in range(n - 1):
        X_aug = augment(X_pp, wns, rng,
                        scale=(0.94, 1.06), amp=AUG_BASE_AMP*0.4,
                        shift=AUG_MAX_SHIFT*0.4, noise=AUG_NOISE_STD*0.4)
        c, _ = model(torch.tensor(X_aug, dtype=torch.float32, device=device))
        preds.append(tscaler.inverse(c.cpu().numpy()))

    return np.mean(preds, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process(preds: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    names = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    out   = np.maximum(preds, 0.0)
    for i, name in enumerate(names):
        lo = max(0.0, np.percentile(y_train[:, i], 1))
        hi = np.percentile(y_train[:, i], 99)
        mg = 0.12 * (hi - lo)
        lo = max(0.0, lo - mg)
        hi = hi + mg
        out[:, i] = np.clip(out[:, i], lo, hi)
        print(f"  {name}: [{lo:.3f}, {hi:.3f}]  (pred range [{preds[:,i].min():.3f}, {preds[:,i].max():.3f}])")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def train_cv(X_pp, y_train, X_test_pp, wns, device):
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((X_test_pp.shape[0], N_TARGETS), dtype=np.float32)
    fold_r2s   = []

    mean_spec = X_pp.mean(axis=0).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"  PINN v3   ({N_FOLDS}-fold, {EPOCHS} ep,  device={device})")
    print(f"{'='*70}")
    tnames = ["Glucose", "NaAc", "MgAc"]

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_pp)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────────────────")
        X_tr, X_val = X_pp[tr_idx], X_pp[val_idx]
        y_tr_raw, y_val_raw = y_train[tr_idx], y_train[val_idx]

        # Fit target scaler on training fold only
        tscaler = TargetScaler().fit(y_tr_raw)
        y_tr    = tscaler.transform(y_tr_raw)

        # Physics bounds in normalised space
        conc_zero_norm = tscaler.conc_zero_normalised()
        conc_max_norm  = tscaler.conc_max_normalised(CONC_MAX_RAW)

        model = RamanPINN(
            input_dim=X_tr.shape[1], hidden_dim=HIDDEN_DIM,
            n_targets=N_TARGETS, dropout=DROPOUT,
            pure_spectra_init=mean_spec,
        ).to(device)

        # Two param groups from the start so the scheduler never sees a size change:
        #   group 0 — encoder + head   (lr = LR,       active immediately)
        #   group 1 — pure_spectra     (lr = 0,        activated at PHYS_START)
        encoder_params = [p for n, p in model.named_parameters()
                          if "pure_spectra" not in n]
        optimizer = torch.optim.AdamW([
            {"params": encoder_params,          "lr": LR,    "weight_decay": WEIGHT_DECAY},
            {"params": [model.pure_spectra],    "lr": 0.0,   "weight_decay": 0.0},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=1e-5)
        criterion = PhysicsInformedLoss(
            conc_zero_norm=conc_zero_norm,
            conc_max_norm=conc_max_norm,
            phys_start=PHYS_START,
            phys_full=PHYS_FULL,
        ).to(device)

        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

        best_r2, best_state = -np.inf, None
        patience = 0
        PATIENCE = 100

        for epoch in range(1, EPOCHS + 1):

            # Activate pure_spectra learning when physics losses kick in
            if epoch == PHYS_START:
                optimizer.param_groups[1]["lr"] = LR * 0.1
                print(f"  -> pure_spectra LR activated at epoch {epoch}")

            tr_losses = train_epoch(model, dl, optimizer, criterion,
                                    epoch, device, wns)
            scheduler.step(epoch)

            if epoch % 10 == 0 or epoch == EPOCHS:
                model.eval()
                with torch.no_grad():
                    val_c_norm, _ = model(X_val_t)
                # Inverse-transform to g/L for R² evaluation
                val_pred = tscaler.inverse(val_c_norm.cpu().numpy())
                val_pred = np.maximum(val_pred, 0.0)
                val_r2   = r2_score(y_val_raw, val_pred)

                if val_r2 > best_r2:
                    best_r2    = val_r2
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if epoch % 100 == 0:
                    r2s    = [r2_score(y_val_raw[:, i], val_pred[:, i]) for i in range(3)]
                    r2_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(tnames, r2s))
                    phys   = " [phys ON]" if epoch >= PHYS_START else " [warmup]"
                    print(f"  ep {epoch:4d} | mse {tr_losses['mse']:.4f} | "
                          f"{r2_str} | best={best_r2:.4f}{phys}")

                if patience >= PATIENCE // 10:
                    print(f"  → Early stop ep {epoch}  best R²={best_r2:.4f}")
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        oof_preds[val_idx] = predict_tta(model, X_val, wns, tscaler, device, TTA_N)
        test_preds        += predict_tta(model, X_test_pp, wns, tscaler, device, TTA_N) / N_FOLDS

        r2s = [r2_score(y_val_raw[:, i], oof_preds[val_idx, i]) for i in range(3)]
        avg = np.mean(r2s)
        fold_r2s.append(avg)
        print(f"  Fold {fold+1} OOF:  "
              + "  ".join(f"{n}={v:.4f}" for n, v in zip(tnames, r2s))
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

    print("\nLoading …")
    X_tr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    X_te_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"),     False)
    X_train_raw = plate_to_grid(X_tr_raw, wns)
    X_test_raw  = plate_to_grid(X_te_raw, wns)
    print(f"  Train {X_train_raw.shape}   Labels {y_train.shape}")
    print(f"  Test  {X_test_raw.shape}")

    print("\nPreprocessing …")
    X_train_pp, msc_ref, scaler = preprocess(X_train_raw, fit=True)
    X_test_pp,  _,       _      = preprocess(X_test_raw,
                                             msc_ref=msc_ref, scaler=scaler, fit=False)
    print(f"  Train PP {X_train_pp.shape}")

    # Print training label stats so we can verify predictions are in range
    print("\nTraining label statistics:")
    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28}  mean={y_train[:,i].mean():.2f}  "
              f"std={y_train[:,i].std():.2f}  "
              f"[{y_train[:,i].min():.2f}, {y_train[:,i].max():.2f}]")

    oof_preds, test_preds, fold_r2s = train_cv(
        X_train_pp, y_train, X_test_pp, wns, device)

    # ── Summary
    print("\n" + "="*70)
    print("  FINAL OOF RESULTS — PINN v3")
    print("="*70)
    oof_r2s = [r2_score(y_train[:, i], oof_preds[:, i]) for i in range(N_TARGETS)]
    for name, r2 in zip(tnames_full, oof_r2s):
        print(f"  {name:<28}  R² = {r2:.4f}")
    print(f"  {'Overall':<28}  R² = {np.mean(oof_r2s):.4f}")
    print(f"\n  Fold R²s: " + "  ".join(f"{v:.4f}" for v in fold_r2s))
    print(f"  CV {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

    # Compare prediction distributions to HGB baseline
    print("\nPrediction distribution check (should match training data range):")
    for i, n in enumerate(tnames_full):
        print(f"  {n:<28}  pred mean={test_preds[:,i].mean():.2f}  "
              f"std={test_preds[:,i].std():.2f}  "
              f"[{test_preds[:,i].min():.2f}, {test_preds[:,i].max():.2f}]")

    print("\nPost-processing …")
    test_final = post_process(test_preds, y_train)

    submission = pd.DataFrame({
        "ID":                np.arange(1, len(test_final) + 1),
        "Glucose":           test_final[:, 0],
        "Sodium Acetate":    test_final[:, 1],
        "Magnesium Sulfate": test_final[:, 2],
    })
    out_path = os.path.join(OUT_DIR, "submission_pinn_v3.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission → {out_path}")
    print(submission.head(10).to_string(index=False))

    return submission


if __name__ == "__main__":
    main()
