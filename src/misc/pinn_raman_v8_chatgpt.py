#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v9 (ChatGPT) —  SSL Shape Encoder + Scale-aware Head        ║
║                                                                              ║
║  Core innovation (A + B):                                                    ║
║                                                                              ║
║  A) SELF-SUPERVISED PRETRAINING (SimCLR / NT-Xent)                            ║
║     - Pretrain a 1D CNN encoder on *unlabeled* Raman spectra using two       ║
║       physics-motivated augmented "views" of the same spectrum.              ║
║     - This learns instrument-invariant "shape" representations before we     ║
║       ever look at concentration labels (key when labeled transfer set is    ║
║       tiny).                                                                 ║
║                                                                              ║
║  B) SEPARATE PHYSICAL SPACE FROM ML SPACE                                     ║
║     - Physical preprocessing (MSC → baseline removal → SavGol) is applied     ║
║       in real spectral units, then we split information into:                 ║
║         • SHAPE: per-sample max-abs normalized spectrum  (x_shape)           ║
║         • SCALE: global intensity features (amp_max, amp_area, …)            ║
║     - The encoder only sees SHAPE. The regression head sees [embedding,      ║
║       SCALE]. This prevents the encoder from overfitting on scale artifacts  ║
║       while still letting the predictor use intensity information.           ║
║                                                                              ║
║  Why v9 can generalize better than “PINN-style reconstruction losses”:       ║
║     - Beer–Lambert linear mixing is not a great fit for fluorescence +       ║
║       baseline-removed, instrument-shifted, normalized Raman spectra.        ║
║     - In practice, learning a robust representation first (SSL) and then     ║
║       doing a small supervised fit is often a stronger inductive bias for    ║
║       transfer learning with tiny target labels.                              ║
║                                                                              ║
║  Outputs:                                                                    ║
║     <out_dir>/submission_v9_ssl.csv                                          ║
║     <out_dir>/encoder_ssl.pt                                                ║
║     <out_dir>/head_fold*.pt                                                 ║
║                                                                              ║
║  Kaggle run example:                                                         ║
║    python pinn_raman_v9_ssl_chatgpt.py \                                     ║
║      --data_dir "/kaggle/input/dig-4-bio-raman-transfer-learning-challenge"  ║
║      --out_dir "./v9_outputs" \                                              ║
║      --ssl_include_test 1 \                                                  ║
║      --ssl_epochs 250 --ssl_batch 512 --ssl_lr 1e-3 \                        ║
║      --head_folds 5 --head_epochs 800 --head_lr 3e-4                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:
  - numpy, pandas, scipy, scikit-learn
  - torch
  - optional: pybaselines (for SNIP baseline); else poly baseline fallback

Notes:
  - This script is intentionally "single file" and competition-oriented.
  - If you want *strict* non-transductive training, set --ssl_include_test 0
    (so SSL does not touch 96_samples.csv).
"""

from __future__ import annotations

import argparse
import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Repro / utils
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n, dtype=np.float64)).astype(np.float32)


def safe_float_cols(cols: List[str]) -> np.ndarray:
    out: List[float] = []
    for c in cols:
        try:
            out.append(float(str(c).strip()))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

DEVICE_FILES = [
    "anton_532.csv",
    "anton_785.csv",
    "kaiser.csv",
    "mettler_toledo.csv",
    "metrohm.csv",
    "tec5.csv",
    "timegate.csv",
    "tornado.csv",
]


def load_device_csv(path: str, grid_wns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Device CSVs (competition):
      - spectral columns first (wavenumber column names)
      - last 5 columns contain labels/metadata; we take the first 3 of those.
    Returns:
      X_grid: (N, D) float32 spectra interpolated to grid_wns
      y:      (N, 3) float32 labels
    """
    df = pd.read_csv(path)

    spec_cols = list(df.columns[:-5])
    wns = safe_float_cols(spec_cols)
    mask = np.isfinite(wns)
    wns = wns[mask]

    spec = df.iloc[:, :-5].loc[:, np.array(mask)].to_numpy(dtype=np.float64)

    # sort by wavenumber to keep interp stable
    order = np.argsort(wns)
    wns = wns[order]
    spec = spec[:, order]

    X_grid = np.vstack([
        np.interp(grid_wns.astype(np.float64), wns.astype(np.float64), row).astype(np.float32)
        for row in spec
    ])

    y = df.iloc[:, -5:-2].to_numpy(dtype=np.float32)  # 3 analytes
    return X_grid.astype(np.float32), y.astype(np.float32)


def _clean_brackets_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_plate_file(path: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    transfer_plate.csv (train): 2 replicates per sample, 2048 points each.
    96_samples.csv (test):      2 replicates per sample, 2048 points each.

    Returns:
      X_2048: (N_samples, 2048) averaged across replicates
      y:      (N_samples, 3) averaged across replicates (train only) or None
    """
    if is_train:
        df = pd.read_csv(path)
        target_cols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        # forward fill targets across replicate rows then average
        y_df = df.loc[:, target_cols].copy().ffill()
        y_all = y_df.to_numpy(dtype=np.float32)
    else:
        df = pd.read_csv(path, header=None)
        y_all = None

    # sample_id is col0 with NaNs on replicate rows
    sample_id = df.iloc[:, 0].ffill()

    # spectral columns are next 2048 columns
    spec_df = df.iloc[:, 1 : 1 + 2048].copy()
    spec_df = spec_df.apply(_clean_brackets_to_float)
    X_all = spec_df.to_numpy(dtype=np.float64)

    if X_all.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates). Got {X_all.shape[0]} rows in {path}")

    X_2048 = X_all.reshape(-1, 2, 2048).mean(axis=1).astype(np.float32)

    if is_train:
        y = y_all.reshape(-1, 2, 3).mean(axis=1).astype(np.float32)
    else:
        y = None

    _ = sample_id  # unused, but kept for clarity
    return X_2048, y


def plate_to_grid(
    X_2048: np.ndarray,
    grid_wns: np.ndarray,
    wn_start: float = 65.0,
    wn_end: float = 3350.0,
) -> np.ndarray:
    """Interpolate 2048-point plate spectra onto the shared wavenumber grid."""
    plate_wns = np.linspace(float(wn_start), float(wn_end), 2048, dtype=np.float64)

    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    m = (plate_wns >= lo) & (plate_wns <= hi)
    plate_wns_sel = plate_wns[m]
    X_sel = X_2048[:, m].astype(np.float64)

    X_grid = np.vstack([
        np.interp(grid_wns.astype(np.float64), plate_wns_sel, row).astype(np.float32)
        for row in X_sel
    ])
    return X_grid.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Physical preprocessing (MSC + baseline + SavGol) + Shape/Scale split
# ─────────────────────────────────────────────────────────────────────────────

def apply_msc(X: np.ndarray, ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Multiplicative Scatter Correction."""
    X64 = X.astype(np.float64, copy=False)
    if ref is None:
        ref = X64.mean(axis=0)
    ref64 = ref.astype(np.float64, copy=False)

    out = np.empty_like(X64, dtype=np.float64)
    for i in range(X64.shape[0]):
        slope, intercept = np.polyfit(ref64, X64[i], 1)
        out[i] = (X64[i] - intercept) / (slope + 1e-12)
    return out.astype(np.float32), ref64.astype(np.float32)


def baseline_snip(X: np.ndarray, max_half_window: int = 20, smooth_half_window: int = 3) -> np.ndarray:
    """SNIP baseline removal (pybaselines). Falls back to polynomial detrend if unavailable."""
    try:
        import pybaselines  # type: ignore
        fitter = pybaselines.Baseline()
        X64 = X.astype(np.float64, copy=False)
        out = np.empty_like(X64, dtype=np.float64)
        for i in range(X64.shape[0]):
            base, _ = fitter.snip(
                X64[i],
                max_half_window=max_half_window,
                decreasing=True,
                smooth_half_window=smooth_half_window,
            )
            out[i] = X64[i] - base
        return out.astype(np.float32)
    except Exception:
        # polynomial baseline fallback
        X64 = X.astype(np.float64, copy=False)
        n, d = X64.shape
        t = np.linspace(-1.0, 1.0, d, dtype=np.float64)
        out = np.empty_like(X64, dtype=np.float64)
        for i in range(n):
            coef = np.polyfit(t, X64[i], deg=3)
            out[i] = X64[i] - np.polyval(coef, t)
        return out.astype(np.float32)


@dataclass
class PhysPreprocessConfig:
    use_msc: bool = True
    baseline: str = "snip"          # "snip" | "poly" | "none"
    snip_halfwin: int = 20
    savgol_window: int = 7
    savgol_polyorder: int = 2
    savgol_deriv: int = 0
    # shape normalization (split from scale)
    shape_norm: str = "maxabs"      # "maxabs" | "none"


@dataclass
class PhysPreprocessState:
    cfg: PhysPreprocessConfig
    msc_ref: Optional[np.ndarray] = None


class PhysPreprocessor:
    """
    Produces:
      - x_shape: (N, D) per-sample normalized spectra (shape only)
      - scale_feats: (N, 2) [log_maxabs, log_area_abs] intensity features
    """
    def __init__(self, cfg: PhysPreprocessConfig):
        self.cfg = cfg
        self.state = PhysPreprocessState(cfg=cfg, msc_ref=None)

    def fit(self, X_ref: np.ndarray) -> "PhysPreprocessor":
        if self.cfg.use_msc:
            _, ref = apply_msc(X_ref, ref=None)
            self.state.msc_ref = ref.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        Xp = X.astype(np.float32, copy=False)

        # 1) MSC
        if cfg.use_msc:
            if self.state.msc_ref is None:
                raise ValueError("PhysPreprocessor.fit() must be called before transform() when use_msc=True")
            Xp, _ = apply_msc(Xp, ref=self.state.msc_ref)

        # 2) baseline
        if cfg.baseline == "snip":
            Xp = baseline_snip(Xp, max_half_window=cfg.snip_halfwin, smooth_half_window=3)
        elif cfg.baseline == "poly":
            Xp = baseline_snip(Xp, max_half_window=cfg.snip_halfwin, smooth_half_window=3)  # poly fallback path
        elif cfg.baseline == "none":
            pass
        else:
            raise ValueError(f"Unknown baseline: {cfg.baseline}")

        # 3) SavGol smoothing
        Xp = savgol_filter(
            Xp.astype(np.float64, copy=False),
            window_length=int(cfg.savgol_window),
            polyorder=int(cfg.savgol_polyorder),
            deriv=int(cfg.savgol_deriv),
            axis=1,
            mode="mirror",
        ).astype(np.float32)

        # 4) scale features (computed before shape normalization)
        max_abs = np.max(np.abs(Xp), axis=1) + 1e-12
        area_abs = np.mean(np.abs(Xp), axis=1) + 1e-12
        scale_feats = np.stack([np.log(max_abs), np.log(area_abs)], axis=1).astype(np.float32)

        # 5) shape normalization
        if cfg.shape_norm == "maxabs":
            X_shape = (Xp / max_abs[:, None]).astype(np.float32)
        elif cfg.shape_norm == "none":
            X_shape = Xp.astype(np.float32)
        else:
            raise ValueError(f"Unknown shape_norm: {cfg.shape_norm}")

        return X_shape, scale_feats


# ─────────────────────────────────────────────────────────────────────────────
# Torch augmentations (vectorized) for SSL and invariance
# ─────────────────────────────────────────────────────────────────────────────

def _shift_spectra_1d(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """
    Differentiable-ish linear interpolation shift.
    x: (B, L), shifts: (B,) in bins (can be fractional).
    Output y[b, i] = x[b] evaluated at q = i - shifts[b].
    """
    B, L = x.shape
    device = x.device
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)
    q = pos - shifts.view(B, 1)  # (B, L)

    q0 = torch.floor(q).to(torch.long)
    q1 = q0 + 1

    q0c = torch.clamp(q0, 0, L - 1)
    q1c = torch.clamp(q1, 0, L - 1)

    w1 = (q - q0.to(torch.float32)).clamp(0.0, 1.0)
    w0 = 1.0 - w1

    x0 = torch.gather(x, 1, q0c)
    x1 = torch.gather(x, 1, q1c)
    return w0 * x0 + w1 * x1


def augment_batch(
    x: torch.Tensor,
    *,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    baseline_amp: float = 0.04,
    baseline_degree: int = 2,
    max_shift_bins: float = 2.0,
    noise_std: float = 0.01,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Physics-motivated augmentations on SHAPE spectra:
      - scale (laser power / focus / coupling)
      - smooth polynomial baseline (residual background)
      - small wavenumber shift (calibration drift)
      - additive noise (shot/electronics)
      - optional dropout (bad pixels)
    Then re-normalize to maxabs=1 to keep the encoder stable.
    """
    B, L = x.shape
    out = x

    # 1) scale
    if scale_range is not None:
        lo, hi = float(scale_range[0]), float(scale_range[1])
        scales = torch.empty((B, 1), device=x.device, dtype=x.dtype).uniform_(lo, hi)
        out = out * scales

    # 2) baseline drift (unit-variance polynomial on [-1,1])
    if baseline_amp and baseline_amp > 0.0:
        t = torch.linspace(-1.0, 1.0, L, device=x.device, dtype=x.dtype)  # (L,)
        # precompute powers: (deg+1, L)
        powers = torch.stack([t ** k for k in range(int(baseline_degree) + 1)], dim=0)
        coeffs = torch.randn((B, int(baseline_degree) + 1), device=x.device, dtype=x.dtype)
        base = coeffs @ powers  # (B, L)
        base = base - base.mean(dim=1, keepdim=True)
        base = base / (base.std(dim=1, keepdim=True) + 1e-8)
        out = out + float(baseline_amp) * base

    # 3) wavenumber shift
    if max_shift_bins and max_shift_bins > 0.0:
        shifts = torch.empty((B,), device=x.device, dtype=x.dtype).uniform_(-max_shift_bins, max_shift_bins)
        out = _shift_spectra_1d(out, shifts)

    # 4) additive noise
    if noise_std and noise_std > 0.0:
        out = out + torch.randn_like(out) * float(noise_std)

    # 5) dropout mask
    if dropout_p and dropout_p > 0.0:
        m = (torch.rand_like(out) > float(dropout_p)).to(out.dtype)
        out = out * m

    # 6) renorm (maxabs)
    out = out / (out.abs().amax(dim=1, keepdim=True) + 1e-8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SSL encoder (1D CNN) + SimCLR loss
# ─────────────────────────────────────────────────────────────────────────────

class RamanEncoder1D(nn.Module):
    """
    Small 1D CNN encoder for 1643-bin Raman spectra.
    Input: (B, D)  -> internally (B,1,D)
    Output: (B, emb_dim)
    """
    def __init__(self, input_dim: int, emb_dim: int = 128, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)

        self.stem = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(width // 2),
            nn.GELU(),
            nn.Conv1d(width // 2, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )

        def block(dil: int) -> nn.Module:
            pad = (5 - 1) * dil // 2
            return nn.Sequential(
                nn.Conv1d(width, width, kernel_size=5, padding=pad, dilation=dil),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(width, width, kernel_size=5, padding=pad, dilation=dil),
                nn.BatchNorm1d(width),
            )

        self.blocks = nn.ModuleList([block(1), block(2), block(4), block(8)])
        self.act = nn.GELU()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        h = x.unsqueeze(1)  # (B,1,D)
        h = self.stem(h)    # (B,C,D)
        for blk in self.blocks:
            h = self.act(h + blk(h))
        h = self.pool(h).squeeze(-1)  # (B,C)
        return self.fc(h)             # (B,emb_dim)


class ProjectionHead(nn.Module):
    """SimCLR projection head."""
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent / SimCLR loss.
    z1, z2: (B, d)
    """
    B = z1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2B,d)

    logits = (z @ z.t()) / float(temperature)  # (2B,2B)

    # mask self-sim
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)

    # positives: i<->i+B
    pos = torch.cat([torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)], dim=0)
    loss = F.cross_entropy(logits, pos)
    return loss


class SSLDataset(Dataset):
    """Holds x_shape spectra only; augmentations happen in the training loop (vectorized)."""
    def __init__(self, X_shape: np.ndarray):
        self.X = torch.tensor(X_shape, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Supervised head on frozen encoder (scale-aware)
# ─────────────────────────────────────────────────────────────────────────────

class ScaleAwareHead(nn.Module):
    """
    Input: [embedding, scale_feats]
    Output: 3 targets (non-negative via Softplus at the end)
    """
    def __init__(self, emb_dim: int, scale_dim: int = 2, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        in_dim = int(emb_dim + scale_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, 3),
        )
        self.out_act = nn.Softplus()

    def forward(self, h: torch.Tensor, scale_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, scale_feats], dim=1)
        y = self.net(x)
        return self.out_act(y)


def compute_target_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-variance weights (normalized to mean 1)."""
    var = np.var(y, axis=0) + 1e-8
    w = (1.0 / var).astype(np.float32)
    w = w / np.mean(w)
    return w


@torch.no_grad()
def encode_all(encoder: nn.Module, X_shape: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    encoder.eval()
    out: List[np.ndarray] = []
    Xt = torch.tensor(X_shape, dtype=torch.float32)
    dl = DataLoader(Xt, batch_size=batch_size, shuffle=False, num_workers=0)
    for xb in dl:
        hb = encoder(xb.to(device)).cpu().numpy()
        out.append(hb)
    return np.concatenate(out, axis=0).astype(np.float32)


def train_head_cv(
    encoder: nn.Module,
    X_shape: np.ndarray,
    scale_feats: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    folds: int = 5,
    epochs: int = 800,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    lambda_inv: float = 0.05,
    inv_aug_cfg: Dict[str, float] = None,
    seed: int = 42,
    out_dir: str = ".",
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Train head with CV, encoder frozen.
    Returns:
      oof_pred:  (N,3)
      test_pred: placeholder (caller computes separately)
      fold_scores: list of avg R2 per fold
    """
    if inv_aug_cfg is None:
        inv_aug_cfg = dict(
            scale_lo=0.98, scale_hi=1.02,
            baseline_amp=0.02,
            baseline_degree=2,
            max_shift_bins=1.0,
            noise_std=0.005,
            dropout_p=0.0,
        )

    kf = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    oof = np.zeros_like(y, dtype=np.float32)
    fold_scores: List[float] = []

    # freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Precompute embeddings for speed (base)
    H_all = encode_all(encoder, X_shape, device=device, batch_size=1024)  # (N,emb)
    emb_dim = H_all.shape[1]

    w = compute_target_weights(y)
    w_t = torch.tensor(w.reshape(1, 3), dtype=torch.float32, device=device)

    print("\n" + "=" * 70)
    print(f"  Head fine-tuning (frozen encoder) — {folds}-fold CV")
    print("=" * 70)
    print("  Target weights:", w)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(H_all), start=1):
        print(f"\n── Fold {fold}/{folds} ──────────────────────────────────────────────")

        H_tr = torch.tensor(H_all[tr_idx], dtype=torch.float32)
        S_tr = torch.tensor(scale_feats[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)

        H_va = torch.tensor(H_all[va_idx], dtype=torch.float32)
        S_va = torch.tensor(scale_feats[va_idx], dtype=torch.float32)
        y_va = y[va_idx].astype(np.float32)

        head = ScaleAwareHead(emb_dim=emb_dim, scale_dim=scale_feats.shape[1], hidden=128, dropout=0.25).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-6)

        # early stopping
        best_score = -1e9
        best_state = None
        patience = 0
        max_patience = max(60, epochs // 10)

        # dataset indices for batching
        rng = np.random.RandomState(seed + fold * 1000)
        n_tr = len(tr_idx)

        for ep in range(1, epochs + 1):
            head.train()
            # shuffle
            perm = rng.permutation(n_tr)
            # mini-batches
            for i0 in range(0, n_tr, batch_size):
                ii = perm[i0 : i0 + batch_size]
                hb = H_tr[ii].to(device)
                sb = S_tr[ii].to(device)
                yb = y_tr[ii].to(device)

                # supervised loss
                pred = head(hb, sb)
                loss_sup = torch.mean(((pred - yb) ** 2) * w_t)

                # invariance: build augmented spectra -> encode -> head -> match
                loss_inv = torch.tensor(0.0, device=device)
                if lambda_inv and lambda_inv > 0.0:
                    # pick the corresponding raw spectra for these indices
                    x_raw = torch.tensor(X_shape[tr_idx[ii]], dtype=torch.float32, device=device)
                    x_aug = augment_batch(
                        x_raw,
                        scale_range=(inv_aug_cfg["scale_lo"], inv_aug_cfg["scale_hi"]),
                        baseline_amp=inv_aug_cfg["baseline_amp"],
                        baseline_degree=int(inv_aug_cfg["baseline_degree"]),
                        max_shift_bins=inv_aug_cfg["max_shift_bins"],
                        noise_std=inv_aug_cfg["noise_std"],
                        dropout_p=inv_aug_cfg["dropout_p"],
                    )
                    with torch.no_grad():
                        h_aug = encoder(x_aug)
                    pred_aug = head(h_aug, sb)
                    loss_inv = F.mse_loss(pred_aug, pred.detach())

                loss = loss_sup + float(lambda_inv) * loss_inv

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=2.0)
                opt.step()

            sched.step()

            # validate every 10 epochs (or last)
            if ep % 10 == 0 or ep == epochs:
                head.eval()
                with torch.no_grad():
                    pv = head(H_va.to(device), S_va.to(device)).cpu().numpy().astype(np.float32)
                # per-target R2
                r2s = [r2_score(y_va[:, j], pv[:, j]) for j in range(3)]
                score = float(np.mean(r2s))
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    patience = 0
                else:
                    patience += 10

                if ep % 50 == 0 or ep == epochs:
                    print(f"  ep {ep:4d} | val R2: " + "  ".join(f"{r:.3f}" for r in r2s) + f"  avg={score:.3f}")

                if patience >= max_patience:
                    break

        if best_state is not None:
            head.load_state_dict(best_state)

        # Save fold head
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"head_state": head.state_dict()}, os.path.join(out_dir, f"head_fold{fold}.pt"))

        # OOF
        head.eval()
        with torch.no_grad():
            pv = head(H_va.to(device), S_va.to(device)).cpu().numpy().astype(np.float32)
        oof[va_idx] = pv

        r2s = [r2_score(y_va[:, j], pv[:, j]) for j in range(3)]
        fold_scores.append(float(np.mean(r2s)))
        print("  Fold R2:", "  ".join(f"{r:.4f}" for r in r2s), f" avg={np.mean(r2s):.4f}")

    return oof, fold_scores


def fit_ridge_calibration(oof: np.ndarray, y: np.ndarray) -> List[Ridge]:
    """Per-target Ridge to calibrate/shrink predictions (often helps small transfer sets)."""
    models = []
    for j in range(3):
        r = Ridge(alpha=0.1, random_state=0).fit(oof[:, j:j+1], y[:, j])
        models.append(r)
    return models


def apply_ridge_calibration(pred: np.ndarray, models: List[Ridge]) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.float32)
    for j, r in enumerate(models):
        out[:, j] = r.predict(pred[:, j:j+1]).astype(np.float32)
    return out


def post_process(pred: np.ndarray, y_train: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Clip to plausible range based on training distribution; enforce non-negativity."""
    out = np.maximum(pred, 0.0).astype(np.float32)
    for j in range(3):
        lo = float(np.percentile(y_train[:, j], p_lo))
        hi = float(np.percentile(y_train[:, j], p_hi))
        margin = 0.12 * (hi - lo + 1e-8)
        lo = max(0.0, lo - margin)
        hi = hi + margin
        out[:, j] = np.clip(out[:, j], lo, hi)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Competition data directory")
    ap.add_argument("--out_dir", default="./v9_outputs")
    ap.add_argument("--seed", type=int, default=42)

    # grid
    ap.add_argument("--wn_low", type=float, default=300.0)
    ap.add_argument("--wn_high", type=float, default=1942.0)
    ap.add_argument("--wn_step", type=float, default=1.0)

    # preprocess
    ap.add_argument("--baseline", choices=["snip", "poly", "none"], default="snip")
    ap.add_argument("--snip_halfwin", type=int, default=20)
    ap.add_argument("--savgol_window", type=int, default=7)
    ap.add_argument("--savgol_polyorder", type=int, default=2)
    ap.add_argument("--shape_norm", choices=["maxabs", "none"], default="maxabs")

    # SSL
    ap.add_argument("--ssl_include_test", type=int, choices=[0, 1], default=1)
    ap.add_argument("--ssl_epochs", type=int, default=250)
    ap.add_argument("--ssl_batch", type=int, default=512)
    ap.add_argument("--ssl_lr", type=float, default=1e-3)
    ap.add_argument("--ssl_weight_decay", type=float, default=1e-4)
    ap.add_argument("--ssl_temp", type=float, default=0.20)

    # SSL augmentation (stronger)
    ap.add_argument("--ssl_scale_lo", type=float, default=0.90)
    ap.add_argument("--ssl_scale_hi", type=float, default=1.10)
    ap.add_argument("--ssl_baseline_amp", type=float, default=0.05)
    ap.add_argument("--ssl_baseline_deg", type=int, default=2)
    ap.add_argument("--ssl_max_shift", type=float, default=2.0)
    ap.add_argument("--ssl_noise_std", type=float, default=0.01)
    ap.add_argument("--ssl_dropout_p", type=float, default=0.00)

    # Head
    ap.add_argument("--head_folds", type=int, default=5)
    ap.add_argument("--head_epochs", type=int, default=800)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--head_weight_decay", type=float, default=1e-4)
    ap.add_argument("--head_batch", type=int, default=32)
    ap.add_argument("--head_lambda_inv", type=float, default=0.05)

    args = ap.parse_args()

    set_seed(int(args.seed))
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device:", device)

    grid_wns = make_grid(float(args.wn_low), float(args.wn_high), float(args.wn_step))
    print(f"Wavenumber grid: {grid_wns[0]:.1f} .. {grid_wns[-1]:.1f}  (D={len(grid_wns)})")

    # 1) Load data
    print("\n[1] Loading data ...")
    # transfer (labeled)
    X_tr_2048, y_tr = load_plate_file(os.path.join(args.data_dir, "transfer_plate.csv"), is_train=True)
    X_te_2048, _ = load_plate_file(os.path.join(args.data_dir, "96_samples.csv"), is_train=False)
    X_tr = plate_to_grid(X_tr_2048, grid_wns)
    X_te = plate_to_grid(X_te_2048, grid_wns)
    assert y_tr is not None
    y_tr = y_tr.astype(np.float32)

    print("  Transfer train:", X_tr.shape, y_tr.shape)
    print("  Test:", X_te.shape)

    # device data (labeled, but used primarily for SSL)
    X_dev_list: List[np.ndarray] = []
    y_dev_list: List[np.ndarray] = []

    for fn in DEVICE_FILES:
        p = os.path.join(args.data_dir, fn)
        if not os.path.exists(p):
            continue
        Xd, yd = load_device_csv(p, grid_wns)
        X_dev_list.append(Xd)
        y_dev_list.append(yd)
        print(f"  Device {fn:<18}  X={Xd.shape}")

    if len(X_dev_list) == 0:
        print("  [WARN] No device files found. SSL will use transfer (and optionally test) only.")

    # 2) Physical preprocessing + shape/scale split
    print("\n[2] Physical preprocessing + shape/scale split ...")
    cfg = PhysPreprocessConfig(
        use_msc=True,
        baseline=str(args.baseline),
        snip_halfwin=int(args.snip_halfwin),
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        savgol_deriv=0,
        shape_norm=str(args.shape_norm),
    )
    pre = PhysPreprocessor(cfg)

    # Fit MSC ref on all "train-side" spectra (transfer + devices).
    X_ref = np.concatenate([X_tr] + X_dev_list, axis=0) if len(X_dev_list) else X_tr
    pre.fit(X_ref)

    X_tr_shape, S_tr = pre.transform(X_tr)
    X_te_shape, S_te = pre.transform(X_te)

    X_dev_shape_all: Optional[np.ndarray] = None
    if len(X_dev_list):
        X_dev_all = np.concatenate(X_dev_list, axis=0)
        X_dev_shape_all, _S_dev = pre.transform(X_dev_all)

    print("  Transfer shape:", X_tr_shape.shape, "scale_feats:", S_tr.shape)
    print("  Test shape:", X_te_shape.shape, "scale_feats:", S_te.shape)

    # 3) SSL pretraining
    print("\n[3] SSL pretraining (SimCLR) ...")
    X_ssl_parts = [X_tr_shape]
    if X_dev_shape_all is not None:
        X_ssl_parts.append(X_dev_shape_all)
    if int(args.ssl_include_test) == 1:
        X_ssl_parts.append(X_te_shape)
    X_ssl = np.concatenate(X_ssl_parts, axis=0).astype(np.float32)
    print("  SSL pool:", X_ssl.shape, " (include_test =", int(args.ssl_include_test), ")")

    ssl_ds = SSLDataset(X_ssl)
    ssl_dl = DataLoader(ssl_ds, batch_size=int(args.ssl_batch), shuffle=True, drop_last=True, num_workers=0)

    encoder = RamanEncoder1D(input_dim=X_ssl.shape[1], emb_dim=128, width=128, dropout=0.10).to(device)
    proj = ProjectionHead(in_dim=128, proj_dim=128).to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=float(args.ssl_lr),
        weight_decay=float(args.ssl_weight_decay),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.ssl_epochs), 1), eta_min=1e-5)

    ssl_aug_cfg = dict(
        scale_range=(float(args.ssl_scale_lo), float(args.ssl_scale_hi)),
        baseline_amp=float(args.ssl_baseline_amp),
        baseline_degree=int(args.ssl_baseline_deg),
        max_shift_bins=float(args.ssl_max_shift),
        noise_std=float(args.ssl_noise_std),
        dropout_p=float(args.ssl_dropout_p),
    )

    encoder.train()
    proj.train()

    for ep in range(1, int(args.ssl_epochs) + 1):
        losses = []
        for xb in ssl_dl:
            xb = xb.to(device)

            x1 = augment_batch(xb, **ssl_aug_cfg)
            x2 = augment_batch(xb, **ssl_aug_cfg)

            h1 = encoder(x1)
            h2 = encoder(x2)

            z1 = proj(h1)
            z2 = proj(h2)

            loss = nt_xent_loss(z1, z2, temperature=float(args.ssl_temp))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(proj.parameters()), max_norm=5.0)
            opt.step()

            losses.append(float(loss.item()))

        sched.step()

        if ep == 1 or ep % 25 == 0 or ep == int(args.ssl_epochs):
            print(f"  ssl ep {ep:4d}/{int(args.ssl_epochs)}  loss={np.mean(losses):.4f}")

    # save encoder
    enc_path = os.path.join(args.out_dir, "encoder_ssl.pt")
    torch.save({"encoder_state": encoder.state_dict(), "pre_cfg": cfg.__dict__, "msc_ref": pre.state.msc_ref}, enc_path)
    print("  Saved:", enc_path)

    # 4) Head training (frozen encoder) with CV
    print("\n[4] Supervised head training ...")
    inv_aug_cfg = dict(
        scale_lo=0.98, scale_hi=1.02,
        baseline_amp=0.02,
        baseline_degree=2,
        max_shift_bins=1.0,
        noise_std=0.005,
        dropout_p=0.0,
    )

    oof, fold_scores = train_head_cv(
        encoder=encoder,
        X_shape=X_tr_shape,
        scale_feats=S_tr,
        y=y_tr,
        device=device,
        folds=int(args.head_folds),
        epochs=int(args.head_epochs),
        lr=float(args.head_lr),
        weight_decay=float(args.head_weight_decay),
        batch_size=int(args.head_batch),
        lambda_inv=float(args.head_lambda_inv),
        inv_aug_cfg=inv_aug_cfg,
        seed=int(args.seed),
        out_dir=str(args.out_dir),
    )

    # report OOF
    oof_r2s = [r2_score(y_tr[:, j], oof[:, j]) for j in range(3)]
    print("\n" + "=" * 70)
    print("  OOF R2 (head + frozen SSL encoder)")
    print("=" * 70)
    print(f"  Glucose         R2 = {oof_r2s[0]:.4f}")
    print(f"  Sodium Acetate  R2 = {oof_r2s[1]:.4f}")
    print(f"  Magnesium Acet. R2 = {oof_r2s[2]:.4f}")
    print(f"  Overall avg     R2 = {np.mean(oof_r2s):.4f}")
    print(f"  Fold avgs:      " + "  ".join(f"{s:.4f}" for s in fold_scores))

    # Optional: calibrate (often stabilizes sodium acetate)
    cal_models = fit_ridge_calibration(oof, y_tr)

    # 5) Predict test using fold heads (average)
    print("\n[5] Predicting test (fold ensemble) ...")
    # precompute embeddings for transfer/test once
    H_tr = encode_all(encoder, X_tr_shape, device=device, batch_size=1024)
    H_te = encode_all(encoder, X_te_shape, device=device, batch_size=1024)

    # Load fold heads and predict
    preds_te = np.zeros((X_te_shape.shape[0], 3), dtype=np.float32)
    for fold in range(1, int(args.head_folds) + 1):
        p = os.path.join(args.out_dir, f"head_fold{fold}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        state = torch.load(p, map_location="cpu")
        head = ScaleAwareHead(emb_dim=H_te.shape[1], scale_dim=S_te.shape[1], hidden=128, dropout=0.25).to(device)
        head.load_state_dict(state["head_state"])
        head.eval()
        with torch.no_grad():
            pt = head(
                torch.tensor(H_te, dtype=torch.float32, device=device),
                torch.tensor(S_te, dtype=torch.float32, device=device),
            ).cpu().numpy().astype(np.float32)
        preds_te += pt / float(args.head_folds)

    # Calibrate + postprocess
    preds_te_cal = apply_ridge_calibration(preds_te, cal_models)
    preds_te_pp = post_process(preds_te_cal, y_tr)

    # 6) Write submission
    sub_path = os.path.join(args.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    out = sub.copy()
    # expected columns: ID + [Glucose, Sodium Acetate, Magnesium Sulfate]
    out.loc[:, ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]] = preds_te_pp[: len(out)]
    out_file = os.path.join(args.out_dir, "submission_v9_ssl.csv")
    out.to_csv(out_file, index=False)
    print("  Wrote:", out_file, "shape=", out.shape)

    # quick stats
    print("\nPrediction stats (post-processed):")
    print(out[["Glucose", "Sodium Acetate", "Magnesium Sulfate"]].describe())


if __name__ == "__main__":
    main()
