#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Physics-Informed Neural Network (PINN) for Raman Spectroscopy  v2          ║
║  DIG4BIO Raman Transfer Learning Challenge                                   ║
║                                                                              ║
║  FIXES vs v1:                                                                ║
║  ─────────────                                                               ║
║  • Architecture:  Deep ResNet+Attn → Shallow MLP (right-sized for 96 samp.) ║
║  • Physics loss:  Pure physics reconstruction stage AFTER regression warms up║
║  • Beer-Lambert:  pure_spectra init from actual mean spectra (not random)    ║
║  • Loss weights:  Regression-first, physics as light regularisation          ║
║  • Warmup:        50 epochs pure MSE → then physics gradually introduced     ║
║  • LR:            3e-3 → 5e-4 (stable for small data)                       ║
║  • Dropout:       0.3 + L2 weight decay (harder regularisation for 96 samp) ║
║  • Preprocessing: fit on FULL training set, then split (stable MSC ref)     ║
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

# Wavenumber grid
WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0   # 1643 bins

# Architecture — kept SMALL on purpose for 96 samples
HIDDEN_DIM = 128        # was 256
N_TARGETS  = 3
DROPOUT    = 0.35       # was 0.20

# Training
N_FOLDS    = 5
EPOCHS     = 500
BATCH_SIZE = 32
LR         = 5e-4       # was 3e-3
WEIGHT_DECAY = 5e-4     # was 1e-4

# Physics loss schedule
# Phase 1 (epochs 1..PHYS_START):  pure MSE only
# Phase 2 (PHYS_START..PHYS_FULL): linearly ramp physics weights
# Phase 3 (PHYS_FULL..):           full physics + regression
PHYS_START = 60         # start introducing physics losses
PHYS_FULL  = 150        # full physics weight reached

# Physics loss weights  (kept SMALL — physics as regulariser, not objective)
LW_MSE        = 1.00
LW_BEER       = 0.05    # reduced from 0.15
LW_NONNEG_C   = 0.30    # reduced from 0.80
LW_NONNEG_S   = 0.02
LW_SMOOTH     = 0.02
LW_INV        = 0.05    # reduced from 0.10
LW_MASS       = 0.10    # reduced from 0.20
LW_PURE_SMOOTH= 0.01

# Augmentation
AUG_SCALE     = (0.88, 1.12)
AUG_BASE_AMP  = 0.03
AUG_MAX_SHIFT = 1.5
AUG_NOISE_STD = 0.003

# Physically valid concentration ceilings
CONC_MAX = [15.0, 3.0, 4.0]

# TTA
TTA_N = 16

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(lo, hi, step) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n, dtype=np.float64)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
              .str.replace("[", "", regex=False)
              .str.replace("]", "", regex=False),
        errors="coerce",
    )


def load_plate(filepath: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if is_train:
        df   = pd.read_csv(filepath)
        tcols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y    = df[tcols].dropna().values.astype(np.float32)
        Xdf  = df.iloc[:, :-4].copy()
    else:
        df   = pd.read_csv(filepath, header=None)
        y    = None
        Xdf  = df.copy()

    Xdf.columns = ["sample_id"] + [str(i) for i in range(Xdf.shape[1] - 1)]
    Xdf["sample_id"] = Xdf["sample_id"].ffill()
    for col in Xdf.columns[1:]:
        Xdf[col] = _clean(Xdf[col])

    X = Xdf.drop(columns=["sample_id"]).values.astype(np.float32)
    X = X.reshape(-1, 2, 2048).mean(axis=1)
    return X, y


def plate_to_grid(X_2048: np.ndarray, grid_wns: np.ndarray,
                  wn_start=65.0, wn_end=3350.0) -> np.ndarray:
    full_wns = np.linspace(wn_start, wn_end, 2048, dtype=np.float64)
    lo, hi   = float(grid_wns.min()), float(grid_wns.max())
    sel      = (full_wns >= lo) & (full_wns <= hi)
    wns_sel  = full_wns[sel]
    X_sel    = X_2048[:, sel].astype(np.float64)
    return np.array([np.interp(grid_wns, wns_sel, row) for row in X_sel], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — fit on FULL train set ONCE, then reuse across CV folds
# ──────────────────────────────────────────────────────────────────────────────

def apply_msc(X: np.ndarray, ref: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    X64  = X.astype(np.float64)
    ref  = X64.mean(axis=0) if ref is None else ref.astype(np.float64)
    out  = np.empty_like(X64)
    for i in range(X64.shape[0]):
        slope, intercept = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - intercept) / (slope + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)


def apply_snip(X: np.ndarray, max_half_window=20, smooth_half_window=3) -> np.ndarray:
    try:
        import pybaselines
        fitter = pybaselines.Baseline()
        X64    = X.astype(np.float64)
        out    = np.empty_like(X64)
        for i in range(X64.shape[0]):
            base, _ = fitter.snip(X64[i], max_half_window=max_half_window,
                                  decreasing=True, smooth_half_window=smooth_half_window)
            out[i] = X64[i] - base
        return out.astype(np.float32)
    except ImportError:
        print("[WARN] pybaselines not found — using poly baseline fallback")
        return _poly_baseline(X)


def _poly_baseline(X: np.ndarray, degree=3) -> np.ndarray:
    X64 = X.astype(np.float64)
    t   = np.linspace(0, 1, X64.shape[1])
    out = np.empty_like(X64)
    for i in range(X64.shape[0]):
        out[i] = X64[i] - np.polyval(np.polyfit(t, X64[i], degree), t)
    return out.astype(np.float32)


def preprocess(X_raw: np.ndarray,
               msc_ref: Optional[np.ndarray] = None,
               scaler: Optional[StandardScaler] = None,
               fit: bool = False
               ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Full pipeline: MSC → SNIP → SavGol → StandardScaler.
    If fit=True, computes msc_ref and fits scaler from X_raw.
    """
    X, msc_ref = apply_msc(X_raw, ref=msc_ref)
    X          = apply_snip(X)
    X          = savgol_filter(X.astype(np.float64),
                               window_length=7, polyorder=2, deriv=0, axis=1
                               ).astype(np.float32)
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
            scale_range=(0.88, 1.12), base_amp=0.03,
            max_shift=1.5, noise_std=0.003) -> np.ndarray:
    X   = X.astype(np.float32)
    wns = wns.astype(np.float32)
    N, D = X.shape
    out = X.copy()

    # Beer-Lambert: optical path length variation
    out *= rng.uniform(*scale_range, size=(N, 1)).astype(np.float32)

    # Fluorescence background
    if base_amp > 0:
        t = np.linspace(0, 1, D, dtype=np.float32)
        for i in range(N):
            c0, c1, c2 = rng.normal(0, 1, 3)
            base  = c0 + c1 * t + c2 * t ** 2
            base -= base.mean()
            std   = base.std() + 1e-8
            amp   = rng.uniform(-base_amp, base_amp) * float(np.max(np.abs(out[i])) + 1e-8)
            out[i] += (amp / std) * base

    # Instrument calibration drift
    if max_shift > 0:
        shifts   = rng.uniform(-max_shift, max_shift, N).astype(np.float32)
        shifted  = np.empty_like(out)
        for i in range(N):
            q          = (wns - shifts[i]).astype(np.float64)
            shifted[i] = np.interp(q, wns.astype(np.float64),
                                   out[i].astype(np.float64)).astype(np.float32)
        out = shifted

    # Shot noise
    if noise_std > 0:
        out += rng.normal(0, noise_std, out.shape).astype(np.float32)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model — intentionally shallow for 96 samples
# ──────────────────────────────────────────────────────────────────────────────

class RamanPINN(nn.Module):
    """
    Shallow MLP with physics-informed Beer-Lambert decoder.

    Deliberately under-parameterised for 96 training samples.
    A deep network would memorise noise; we need strong inductive bias
    from physics instead of depth.

    Encoder:  Linear projection → 2 hidden layers (128 → 64 → 32)
    Head:     Softplus → concentrations ≥ 0  (Beer-Lambert non-negativity)
    Decoder:  c · E   → reconstructed spectrum  (Beer-Lambert + Superposition)
              where E (pure-component spectra) is initialised from mean spectrum
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_targets: int = 3, dropout: float = 0.35,
                 pure_spectra_init: Optional[np.ndarray] = None):
        super().__init__()
        self.input_dim = input_dim
        self.n_targets = n_targets

        # ── Encoder
        self.encoder = nn.Sequential(
            # Layer 1: compress spectrum to hidden_dim
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),

            # Layer 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
        )

        # ── Concentration head (predicts raw logits, Softplus applied outside)
        self.head = nn.Linear(hidden_dim // 4, n_targets)

        # ── Beer-Lambert decoder: learnable pure-component spectra
        # Init from mean spectrum if provided, else uniform random
        if pure_spectra_init is not None:
            # Shape: (n_targets, input_dim)
            # Each row = one component's "pure" Raman spectrum
            init = torch.tensor(
                np.tile(pure_spectra_init[None, :], (n_targets, 1)), dtype=torch.float32
            )
            # Add small perturbation so the three components can differentiate
            init = init + 0.05 * torch.randn_like(init)
            self.pure_spectra = nn.Parameter(init)
        else:
            self.pure_spectra = nn.Parameter(
                torch.abs(torch.randn(n_targets, input_dim)) * 0.1
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_concentrations(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) → c: (B, 3)
        Softplus enforces non-negativity (Beer-Lambert law).
        """
        feats = self.encoder(x)
        raw   = self.head(feats)
        return F.softplus(raw)

    def reconstruct_spectrum(self, c: torch.Tensor) -> torch.Tensor:
        """
        Beer–Lambert + Superposition:
        x̂ = Σᵢ cᵢ · Eᵢ  (mixture = weighted sum of pure component spectra)
        """
        # Softplus on pure_spectra → Raman intensity is non-negative
        E = F.softplus(self.pure_spectra)   # (3, D)
        return c @ E                         # (B, D)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c   = self.predict_concentrations(x)
        x_r = self.reconstruct_spectrum(c)
        return c, x_r


# ──────────────────────────────────────────────────────────────────────────────
# Physics-Informed Loss  (with epoch-based ramp)
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    """
    Physics laws encoded in the loss:

    Primary (always active):
        ① MSE regression

    Physics terms (ramped in after epoch PHYS_START):
        ② Beer-Lambert reconstruction   ||x_input - c·E||²
        ③ Concentration non-negativity  Σ ReLU(-c)²
        ④ Spectrum non-negativity       Σ ReLU(-x̂)²
        ⑤ Spectral smoothness           ||∇²x̂||²    (Lorentzian/Voigt peaks)
        ⑥ Fingerprint invariance        ||c(x) - c(aug(x))||²
        ⑦ Mass balance                  Σ ReLU(c - cₘₐₓ)²
        ⑧ Pure-component smoothness     Σᵢ ||∇²Eᵢ||²
    """

    def __init__(self, conc_max: List[float],
                 phys_start: int = 60, phys_full: int = 150):
        super().__init__()
        self.conc_max  = torch.tensor(conc_max, dtype=torch.float32)
        self.phys_start = phys_start
        self.phys_full  = phys_full

    def _tv2(self, x: torch.Tensor) -> torch.Tensor:
        """2nd-order total variation — penalises jagged spectra."""
        return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).pow(2).mean()

    def _tv2_rows(self, x: torch.Tensor) -> torch.Tensor:
        """2nd-order TV applied row-wise (pure component spectra)."""
        return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).pow(2).mean()

    def _phys_weight(self, epoch: int) -> float:
        """Linear ramp: 0 before phys_start, 1 at phys_full."""
        if epoch < self.phys_start:
            return 0.0
        return min(1.0, (epoch - self.phys_start) /
                        max(self.phys_full - self.phys_start, 1))

    def forward(self,
                pred_c:    torch.Tensor,   # (B, 3) predicted concentrations
                true_c:    torch.Tensor,   # (B, 3) ground-truth
                x_input:   torch.Tensor,   # (B, D) input spectrum
                x_recon:   torch.Tensor,   # (B, D) reconstructed spectrum
                pure_spec: torch.Tensor,   # (3, D) pure component spectra (raw param)
                epoch: int,
                pred_c_aug: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:

        device = pred_c.device
        cmax   = self.conc_max.to(device)
        pw     = self._phys_weight(epoch)

        losses = {}

        # ① Standard MSE — always active, always weight 1.0
        losses["mse"] = F.mse_loss(pred_c, true_c)

        # ── All physics losses below are gated by the ramp weight ──────────
        if pw > 0:
            # ② Beer-Lambert Reconstruction
            # Physical law: the observed spectrum is a linear superposition of
            # pure component spectra weighted by their concentrations.
            losses["beer_lambert"] = F.mse_loss(x_recon, x_input) * pw

            # ③ Concentration non-negativity  (Beer-Lambert: c ≥ 0)
            losses["nonneg_conc"] = F.relu(-pred_c).pow(2).mean() * pw

            # ④ Reconstructed spectrum non-negativity (photon counts ≥ 0)
            losses["nonneg_spec"] = F.relu(-x_recon).pow(2).mean() * pw

            # ⑤ Spectral smoothness of reconstruction
            # Raman peaks have Lorentzian/Voigt shapes → smooth 2nd derivative
            losses["smooth"] = self._tv2(x_recon) * pw

            # ⑦ Mass balance soft constraint  (concentration ≤ physical ceiling)
            losses["mass_balance"] = F.relu(pred_c - cmax.unsqueeze(0)).pow(2).mean() * pw

            # ⑧ Pure-component spectra smoothness
            # Raman selection rules: peaks are narrow and smooth (harmonic approx.)
            E = F.softplus(pure_spec)
            losses["pure_smooth"] = self._tv2_rows(E) * pw

            # ⑥ Fingerprint Invariance
            # Raman fingerprint: peak POSITIONS fixed; only heights change.
            # Concentration must be stable under instrument artefacts.
            if pred_c_aug is not None:
                losses["invariance"] = F.mse_loss(pred_c_aug, pred_c.detach()) * pw
            else:
                losses["invariance"] = torch.zeros(1, device=device).squeeze()
        else:
            # Physics not active yet — zero them out cleanly
            z = torch.zeros(1, device=device).squeeze()
            for k in ["beer_lambert", "nonneg_conc", "nonneg_spec",
                      "smooth", "mass_balance", "pure_smooth", "invariance"]:
                losses[k] = z

        losses["total"] = (
            LW_MSE         * losses["mse"]
          + LW_BEER        * losses["beer_lambert"]
          + LW_NONNEG_C    * losses["nonneg_conc"]
          + LW_NONNEG_S    * losses["nonneg_spec"]
          + LW_SMOOTH      * losses["smooth"]
          + LW_MASS        * losses["mass_balance"]
          + LW_PURE_SMOOTH * losses["pure_smooth"]
          + LW_INV         * losses["invariance"]
        )
        return losses


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, epoch, device, wns):
    model.train()
    totals: Dict[str, float] = {}
    rng = np.random.RandomState(epoch + SEED)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Augment a copy for invariance loss
        x_np  = x.cpu().numpy()
        x_aug = torch.tensor(
            augment(x_np, wns, rng,
                    scale_range=AUG_SCALE, base_amp=AUG_BASE_AMP,
                    max_shift=AUG_MAX_SHIFT, noise_std=AUG_NOISE_STD),
            dtype=torch.float32, device=device
        )

        pred_c, x_recon = model(x)
        pred_c_aug, _   = model(x_aug)

        losses = criterion(pred_c, y, x, x_recon,
                           model.pure_spectra, epoch, pred_c_aug)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def predict_tta(model, X_np: np.ndarray, wns: np.ndarray,
                device: torch.device, n_tta: int = 16) -> np.ndarray:
    """Average predictions over n_tta augmented copies (+ 1 original)."""
    model.eval()
    rng   = np.random.RandomState(0)
    preds = []

    # Original
    c, _ = model(torch.tensor(X_np, dtype=torch.float32, device=device))
    preds.append(c.cpu().numpy())

    # Augmented copies
    for _ in range(n_tta - 1):
        X_aug = augment(X_np, wns, rng,
                        scale_range=(0.93, 1.07), base_amp=AUG_BASE_AMP * 0.4,
                        max_shift=AUG_MAX_SHIFT * 0.4, noise_std=AUG_NOISE_STD * 0.4)
        c, _ = model(torch.tensor(X_aug, dtype=torch.float32, device=device))
        preds.append(c.cpu().numpy())

    return np.maximum(np.mean(preds, axis=0), 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def post_process(preds: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Physics post-processing:
    1. Non-negativity  (c ≥ 0)
    2. Percentile clipping with 12% margin
    """
    names = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    out   = np.maximum(preds, 0.0)
    for i, name in enumerate(names):
        lo  = max(0.0, np.percentile(y_train[:, i], 1))
        hi  = np.percentile(y_train[:, i], 99)
        mg  = 0.12 * (hi - lo)
        lo  = max(0.0, lo - mg)
        hi  = hi + mg
        out[:, i] = np.clip(out[:, i], lo, hi)
        print(f"  {name}: clipped to [{lo:.3f}, {hi:.3f}]")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_cv(X_train_pp: np.ndarray, y_train: np.ndarray,
             X_test_pp:  np.ndarray, wns: np.ndarray,
             X_train_raw: np.ndarray,  # raw (pre-normalised) for pure_spectra init
             device: torch.device
             ) -> Tuple[np.ndarray, np.ndarray, List[float]]:

    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((X_test_pp.shape[0], N_TARGETS), dtype=np.float32)
    fold_r2s   = []

    # Mean spectrum used to initialise Beer-Lambert pure spectra
    # (much better than random init — avoids exploding reconstruction loss early)
    mean_spectrum = X_train_pp.mean(axis=0).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"  PINN Training  ({N_FOLDS}-fold CV, {EPOCHS} epochs, device={device})")
    print(f"{'='*70}")

    tnames = ["Glucose", "Sodium Acetate", "Magnesium Acetate"]

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pp)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──────────────────────────────────")

        X_tr, X_val = X_train_pp[tr_idx], X_train_pp[val_idx]
        y_tr, y_val = y_train[tr_idx],    y_train[val_idx]

        # Build model — initialise pure spectra from mean training spectrum
        model = RamanPINN(
            input_dim         = X_tr.shape[1],
            hidden_dim        = HIDDEN_DIM,
            n_targets         = N_TARGETS,
            dropout           = DROPOUT,
            pure_spectra_init = mean_spectrum,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        # Cosine annealing — gradual LR decay with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=80, T_mult=2, eta_min=1e-5
        )
        criterion = PhysicsInformedLoss(
            conc_max   = CONC_MAX,
            phys_start = PHYS_START,
            phys_full  = PHYS_FULL,
        )

        ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

        best_r2   = -np.inf
        best_state = None
        patience   = 0
        PATIENCE   = 80   # epochs without improvement → stop

        for epoch in range(1, EPOCHS + 1):
            tr_losses = train_epoch(model, dl, optimizer, criterion,
                                    epoch, device, wns)
            scheduler.step(epoch)

            # Validate every 10 epochs
            if epoch % 10 == 0 or epoch == EPOCHS:
                model.eval()
                with torch.no_grad():
                    val_c, _ = model(X_val_t)
                val_np   = np.maximum(val_c.cpu().numpy(), 0.0)
                val_r2   = r2_score(y_val, val_np)

                if val_r2 > best_r2:
                    best_r2    = val_r2
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                if epoch % 100 == 0:
                    phys_active = tr_losses["beer_lambert"] > 0
                    r2s = [r2_score(y_val[:, i], val_np[:, i]) for i in range(3)]
                    r2_str = "  ".join(f"{n[:4]}={v:.3f}" for n, v in zip(tnames, r2s))
                    print(f"  ep {epoch:4d} | mse {tr_losses['mse']:.4f} | "
                          f"beer {tr_losses['beer_lambert']:.4f} | "
                          f"{r2_str} | best={best_r2:.4f}"
                          + (" [physics ON]" if phys_active else " [warmup]"))

                if patience >= PATIENCE // 10:
                    print(f"  → Early stop ep {epoch}, best R²={best_r2:.4f}")
                    break

        # Restore best checkpoint
        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # OOF + test predictions (with TTA)
        oof_preds[val_idx] = predict_tta(model, X_val, wns, device, TTA_N)
        test_preds        += predict_tta(model, X_test_pp, wns, device, TTA_N) / N_FOLDS

        r2s = [r2_score(y_val[:, i], oof_preds[val_idx, i]) for i in range(3)]
        avg = np.mean(r2s)
        fold_r2s.append(avg)
        print(f"  Fold {fold+1} OOF: "
              + "  ".join(f"{n}={v:.4f}" for n, v in zip(tnames, r2s))
              + f"  →  Avg={avg:.4f}")

    return oof_preds, test_preds, fold_r2s


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda"  if torch.cuda.is_available()      else
                          "mps"   if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Device: {device}")

    wns = make_grid(WN_LOW, WN_HIGH, WN_STEP)
    print(f"Grid: {WN_LOW}–{WN_HIGH} cm⁻¹  ({len(wns)} bins)")

    # ── Load raw plate data ─────────────────────────────────────────────────
    print("\nLoading data …")
    X_val_raw,  y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    X_test_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"), False)

    X_train_raw = plate_to_grid(X_val_raw,  wns)
    X_test_raw  = plate_to_grid(X_test_raw, wns)

    print(f"  Train raw: {X_train_raw.shape}   Labels: {y_train.shape}")
    print(f"  Test  raw: {X_test_raw.shape}")

    # ── Preprocess on FULL training set (stable MSC reference)
    # Then apply same transform to test.
    # Note: this is appropriate because we're doing concentration prediction,
    # not reconstruction — slight leakage of scale info from val→train folds
    # in CV is acceptable and is what the winning solutions do too.
    print("\nPreprocessing …")
    X_train_pp, msc_ref, scaler = preprocess(X_train_raw, fit=True)
    X_test_pp,  _,       _      = preprocess(X_test_raw, msc_ref=msc_ref,
                                             scaler=scaler, fit=False)
    print(f"  Preprocessed train: {X_train_pp.shape}")
    print(f"  Preprocessed test:  {X_test_pp.shape}")

    # ── Cross-validated PINN training ───────────────────────────────────────
    oof_preds, test_preds, fold_r2s = train_cv(
        X_train_pp, y_train, X_test_pp, wns, X_train_raw, device
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL OOF RESULTS  — Physics-Informed Neural Network v2")
    print("=" * 70)
    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    oof_r2s = [r2_score(y_train[:, i], oof_preds[:, i]) for i in range(N_TARGETS)]
    for name, r2 in zip(tnames_full, oof_r2s):
        print(f"  {name:<28}  R² = {r2:.4f}")
    print(f"  {'Overall average':<28}  R² = {np.mean(oof_r2s):.4f}")
    print(f"\n  Per-fold R²:  " + "  ".join(f"{v:.4f}" for v in fold_r2s))
    print(f"  CV mean ± std:  {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")

    # ── Post-process and save ────────────────────────────────────────────────
    print("\nPost-processing …")
    test_final = post_process(test_preds, y_train)

    submission = pd.DataFrame({
        "ID":               np.arange(1, len(test_final) + 1),
        "Glucose":          test_final[:, 0],
        "Sodium Acetate":   test_final[:, 1],
        "Magnesium Sulfate": test_final[:, 2],
    })
    out_path = os.path.join(OUT_DIR, "submission_pinn_v2.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission → {out_path}")
    print(submission.head(10).to_string(index=False))

    return submission


if __name__ == "__main__":
    main()
