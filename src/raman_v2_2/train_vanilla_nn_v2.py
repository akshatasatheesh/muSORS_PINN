#!/usr/bin/env python3
"""
DIG4BIO Raman Transfer Learning - train_vanilla_nn_v2.py (standalone, improved)

What this version fixes/improves (high impact):
- Uses a consistent shared grid: [wn_floor .. wn_ceil] INCLUSIVE (default 300..1942 => 1643 bins)
- Loads device training CSVs + preserves device_id
- Loads transfer_plate.csv / 96_samples.csv in the Kaggle “plate format” (2 repeats x 2048), averages repeats,
  then interpolates onto the same shared grid
- Preprocess: MSC -> baseline (poly or SNIP if pybaselines installed) -> SavGol -> scaling
- Adds per-sample max normalization (recommended) via --scaling per_sample_max
- Adds device-conditioning (one-hot) + optional test-time “average over devices” ensemble for val/test
- Adds safer invariance loss (warmup + detach) so it doesn’t explode
- Adds optional target scaling (TargetScaler) and/or target-weighted MSE when not scaling y
- Writes <out_dir>/submission.csv

Requirements:
- numpy, pandas, torch, scipy
- optional: pybaselines (only if you choose --baseline snip)

Run example:
python train_vanilla_nn_v2.py \
  --data_dir "/path/to/dig-4-bio-raman-transfer-learning-challenge" \
  --out_dir ./outputs_v2_3 \
  --epochs 200 --batch_size 64 --lr 3e-4 \
  --use_msc --baseline snip --use_savgol --savgol_window 7 --savgol_polyorder 2 \
  --scaling per_sample_max \
  --lambda_inv 0.01 --inv_warmup 10 --inv_max_shift 0.2 --inv_baseline_amp 0.005 \
  --test_device_ensemble 1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy.signal import savgol_filter


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_shared_grid(wn_floor: float, wn_ceil: float, wn_step: float) -> np.ndarray:
    # inclusive grid (important)
    # Example: 300..1942 step 1 => 1643 points
    n = int(round((wn_ceil - wn_floor) / wn_step)) + 1
    grid = wn_floor + wn_step * np.arange(n, dtype=np.float64)
    return grid.astype(np.float32)


def safe_float_cols(cols) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(float(c))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


# ----------------------------
# Data Loading
# ----------------------------

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


@dataclass
class TrainDeviceDataset:
    X: np.ndarray                 # (N, D)
    y: np.ndarray                 # (N, 3)
    device_id: np.ndarray         # (N,)
    target_names: List[str]


def load_device_training(data_dir: str, grid_wns: np.ndarray) -> TrainDeviceDataset:
    """
    Device CSVs: spectral columns then last 5 columns:
      - 4 labels? (we take the 3 analytes only)
      - CV fold index in last column
    We follow common structure: labels are df.iloc[:, -5:-1], then take first 3.
    """
    Xs = []
    ys = []
    devs = []

    for dev_idx, fname in enumerate(DEVICE_FILES):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)

        # spectral columns exclude last 5 columns
        spec_cols = df.columns[:-5]
        wns = safe_float_cols(spec_cols)
        mask = np.isfinite(wns)

        wns = wns[mask]
        spec_vals = df.iloc[:, :-5].loc[:, np.array(mask)].values.astype(np.float64)

        # sort by wavenumber (np.interp expects xp increasing)
        order = np.argsort(wns)
        wns_sorted = wns[order]
        spec_sorted = spec_vals[:, order]

        # interpolate each spectrum onto shared grid
        X_interp = np.array(
            [np.interp(grid_wns, xp=wns_sorted, fp=row).astype(np.float32) for row in spec_sorted],
            dtype=np.float32,
        )

        # labels: df.iloc[:, -5:-1] => 4 columns; take first 3
        y_all = df.iloc[:, -5:-1].values.astype(np.float32)
        y = y_all[:, :3].astype(np.float32)

        Xs.append(X_interp)
        ys.append(y)
        devs.append(np.full((len(X_interp),), dev_idx, dtype=np.int64))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    device_id = np.concatenate(devs, axis=0)
    target_names = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]
    return TrainDeviceDataset(X=X, y=y, device_id=device_id, target_names=target_names)


def load_plate_file(path: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    transfer_plate.csv: has targets and “plate format” spectra (2 repeats x 2048) with a sample_id column.
    96_samples.csv: similar, but no targets; often headerless.

    We parse exactly like the Kaggle notebook pattern:
    - read CSV
    - set first col as sample_id, forward fill, clean brackets in spectral cells
    - reshape (-1, 2, 2048) and mean over repeats => (n_samples, 2048)
    """
    if is_train:
        df = pd.read_csv(path)
        target_cols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
        y = df[target_cols].dropna().values.astype(np.float32)
        Xdf = df.iloc[:, :-4].copy()
    else:
        df = pd.read_csv(path, header=None)
        y = None
        Xdf = df.copy()

    # rename cols: first is sample_id
    Xdf.columns = ["sample_id"] + [str(i) for i in range(Xdf.shape[1] - 1)]

    # forward fill sample_id
    Xdf["sample_id"] = Xdf["sample_id"].ffill()

    # clean sample_id
    if is_train:
        Xdf["sample_id"] = Xdf["sample_id"].astype(str).str.strip()
    else:
        # typically "sample 1" etc
        Xdf["sample_id"] = (
            Xdf["sample_id"].astype(str).str.strip().str.replace("sample", "", regex=False).astype(int)
        )

    # clean spectral strings
    spectral_cols = Xdf.columns[1:]
    for col in spectral_cols:
        Xdf[col] = Xdf[col].astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
        Xdf[col] = pd.to_numeric(Xdf[col], errors="coerce")

    X = Xdf.drop(columns=["sample_id"]).values.astype(np.float32)

    # reshape 2 repeats
    if X.shape[1] != 2048:
        # if shape differs, fail loudly with helpful message
        raise ValueError(
            f"Expected 2048 spectral points per repeat, got {X.shape[1]} columns. "
            f"File: {path}. You may need to adjust parsing."
        )
    X = X.reshape(-1, 2, 2048).mean(axis=1).astype(np.float32)
    return X, y


def interpolate_plate_to_grid(X_plate_2048: np.ndarray, grid_wns: np.ndarray,
                              plate_wn_start: float = 65.0, plate_wn_end: float = 3350.0) -> np.ndarray:
    """
    Plate spectra are sampled over 2048 points linearly from plate_wn_start..plate_wn_end.
    We select the region overlapping grid range and interpolate to grid.
    """
    spectral_values = np.linspace(plate_wn_start, plate_wn_end, 2048, dtype=np.float64)

    lo = float(grid_wns.min())
    hi = float(grid_wns.max())
    sel = (spectral_values >= lo) & (spectral_values <= hi)

    wns = spectral_values[sel]
    X_sel = X_plate_2048[:, sel].astype(np.float64)

    X_interp = np.array([np.interp(grid_wns, xp=wns, fp=row) for row in X_sel], dtype=np.float32)
    return X_interp


# ----------------------------
# Preprocessing
# ----------------------------

def msc(spectra: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Multiplicative Scatter Correction
    spectra: (N, D)
    """
    X = spectra.astype(np.float64, copy=False)
    ref = X.mean(axis=0) if reference is None else reference.astype(np.float64, copy=False)

    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        slope, intercept = np.polyfit(ref, X[i], 1)
        out[i] = (X[i] - intercept) / (slope + 1e-10)
    return out.astype(np.float32)


def baseline_poly(spectra: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Simple polynomial baseline subtract per spectrum.
    """
    X = spectra.astype(np.float64, copy=False)
    n, d = X.shape
    x = np.linspace(0, 1, d, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)

    for i in range(n):
        coef = np.polyfit(x, X[i], degree)
        base = np.polyval(coef, x)
        out[i] = X[i] - base
    return out.astype(np.float32)


def baseline_snip_pybaselines(spectra: np.ndarray,
                             max_half_window: int = 20,
                             smooth_half_window: int = 3) -> np.ndarray:
    """
    SNIP baseline using pybaselines if installed.
    """
    try:
        import pybaselines
    except Exception as e:
        raise ImportError(
            "pybaselines not installed. Install with: pip install pybaselines "
            "or use --baseline poly/none."
        ) from e

    fitter = pybaselines.Baseline()
    X = spectra.astype(np.float64, copy=False)
    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        baseline, _params = fitter.snip(
            X[i],
            max_half_window=max_half_window,
            decreasing=True,
            smooth_half_window=smooth_half_window,
        )
        out[i] = X[i] - baseline
    return out.astype(np.float32)


def savgol(spectra: np.ndarray, window: int = 7, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
    X = spectra.astype(np.float64, copy=False)
    out = savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1)
    return out.astype(np.float32)


def standard_scale_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.std(axis=0, dtype=np.float64)
    sd = np.maximum(sd, 1e-8)
    return mu.astype(np.float32), sd.astype(np.float32)


def standard_scale_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def per_sample_max_norm(X: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(X), axis=1, keepdims=True) + 1e-8
    return (X / denom).astype(np.float32)


@dataclass
class PreprocessConfig:
    use_msc: bool
    baseline: str               # none|poly|snip
    baseline_degree: int
    snip_max_half_window: int
    snip_smooth_half_window: int
    use_savgol: bool
    savgol_window: int
    savgol_polyorder: int
    savgol_deriv: int
    scaling: str                # none|standard|per_sample_max


@dataclass
class PreprocessState:
    cfg: PreprocessConfig
    # for standard scaling
    mu: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    # for MSC reference
    msc_ref: Optional[np.ndarray] = None


def preprocess_fit(X_train_raw: np.ndarray, cfg: PreprocessConfig) -> PreprocessState:
    st = PreprocessState(cfg=cfg)

    X = X_train_raw.astype(np.float32, copy=False)

    if cfg.use_msc:
        st.msc_ref = X.mean(axis=0).astype(np.float32)

    # build X_pre for fitting standard scaler (if needed)
    X_pre = X
    if cfg.use_msc:
        X_pre = msc(X_pre, reference=st.msc_ref)
    if cfg.baseline == "poly":
        X_pre = baseline_poly(X_pre, degree=cfg.baseline_degree)
    elif cfg.baseline == "snip":
        X_pre = baseline_snip_pybaselines(
            X_pre,
            max_half_window=cfg.snip_max_half_window,
            smooth_half_window=cfg.snip_smooth_half_window,
        )
    elif cfg.baseline == "none":
        pass
    else:
        raise ValueError(f"Unknown baseline: {cfg.baseline}")

    if cfg.use_savgol:
        X_pre = savgol(X_pre, window=cfg.savgol_window, polyorder=cfg.savgol_polyorder, deriv=cfg.savgol_deriv)

    if cfg.scaling == "standard":
        mu, sd = standard_scale_fit(X_pre)
        st.mu, st.sd = mu, sd

    return st


def preprocess_apply(X_raw: np.ndarray, st: PreprocessState) -> np.ndarray:
    cfg = st.cfg
    X = X_raw.astype(np.float32, copy=False)

    if cfg.use_msc:
        X = msc(X, reference=st.msc_ref)

    if cfg.baseline == "poly":
        X = baseline_poly(X, degree=cfg.baseline_degree)
    elif cfg.baseline == "snip":
        X = baseline_snip_pybaselines(
            X,
            max_half_window=cfg.snip_max_half_window,
            smooth_half_window=cfg.snip_smooth_half_window,
        )
    elif cfg.baseline == "none":
        pass

    if cfg.use_savgol:
        X = savgol(X, window=cfg.savgol_window, polyorder=cfg.savgol_polyorder, deriv=cfg.savgol_deriv)

    if cfg.scaling == "standard":
        assert st.mu is not None and st.sd is not None
        X = standard_scale_apply(X, st.mu, st.sd)
    elif cfg.scaling == "per_sample_max":
        X = per_sample_max_norm(X)
    elif cfg.scaling == "none":
        X = X.astype(np.float32)
    else:
        raise ValueError(f"Unknown scaling: {cfg.scaling}")

    return X


# ----------------------------
# Augmentation (for invariance)
# ----------------------------

def augment_spectra(
    x_raw: np.ndarray,
    grid_wns: np.ndarray,
    rng: np.random.RandomState,
    scale_range: Tuple[float, float],
    baseline_amp: float,
    baseline_degree: int,
    max_shift: float,
    noise_std: float,
) -> np.ndarray:
    """
    Nuisance transforms on RAW spectra (already on grid_wns).
    - multiplicative scale
    - low-order baseline drift
    - small wavenumber shift (interp)
    - gaussian noise
    """
    X = x_raw.astype(np.float64, copy=True)
    n, d = X.shape

    # 1) scale
    scales = rng.uniform(scale_range[0], scale_range[1], size=(n, 1))
    X *= scales

    # 2) baseline drift
    if baseline_amp > 0:
        x = np.linspace(-1, 1, d, dtype=np.float64)
        for i in range(n):
            coef = rng.normal(loc=0.0, scale=1.0, size=(baseline_degree + 1,))
            coef *= baseline_amp * (np.max(np.abs(X[i])) + 1e-8)
            drift = np.polyval(coef, x)
            X[i] = X[i] + drift

    # 3) shift in wavenumber (cm^-1)
    if max_shift > 0:
        shifts = rng.uniform(-max_shift, max_shift, size=(n,))
        for i in range(n):
            xp = grid_wns.astype(np.float64) + shifts[i]
            # interpolate back onto grid
            X[i] = np.interp(grid_wns, xp=xp, fp=X[i], left=X[i, 0], right=X[i, -1])

    # 4) noise
    if noise_std > 0:
        X += rng.normal(0.0, noise_std, size=X.shape)

    return X.astype(np.float32)


# ----------------------------
# Model
# ----------------------------

class VanillaMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(512, 256, 128), dropout: float = 0.05, non_negative: bool = True):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 3)
        self.non_negative = non_negative
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        y = self.head(h)
        if self.non_negative:
            y = self.softplus(y)
        return y


class TargetScaler:
    """Standardize targets (fit on train only) and invert at eval/predict."""
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = y.mean(axis=0)
        self.std_ = np.maximum(y.std(axis=0), self.eps)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        y = np.asarray(y, dtype=np.float64)
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
        return (y_scaled * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self):
        return {
            "mean_": None if self.mean_ is None else self.mean_.astype(np.float32),
            "std_": None if self.std_ is None else self.std_.astype(np.float32),
            "eps": float(self.eps),
        }


# ----------------------------
# Metrics / Eval
# ----------------------------

@torch.no_grad()
def evaluate_rmse_r2(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    y_scaler: Optional[TargetScaler],
    predict_fn,
) -> Tuple[float, np.ndarray]:
    model.eval()
    pred_scaled = predict_fn(X).detach().cpu().numpy()
    y_scaled = y.detach().cpu().numpy()

    if y_scaler is not None:
        pred = y_scaler.inverse(pred_scaled)
        y_true = y_scaler.inverse(y_scaled)
    else:
        pred = pred_scaled
        y_true = y_scaled

    mse = float(np.mean((pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((y_true - pred) ** 2, axis=0)
    mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return rmse, r2


# ----------------------------
# Training
# ----------------------------

def make_device_onehot(device_id: torch.Tensor, num_devices: int) -> torch.Tensor:
    # device_id: (B,)
    oh = torch.nn.functional.one_hot(device_id, num_classes=num_devices).float()
    return oh


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    prep_state: PreprocessState,
    grid_wns: np.ndarray,
    args: argparse.Namespace,
    epoch: int,
    num_devices: int,
    target_weights_t: Optional[torch.Tensor],
) -> Tuple[float, float]:
    model.train()

    sup_sum = 0.0
    inv_sum = 0.0
    inv_count = 0
    n = 0

    for step, (x_raw_t, dev_t, y_t) in enumerate(loader):
        x_raw = x_raw_t.numpy().astype(np.float32, copy=False)   # raw on grid already
        dev = dev_t.to(device=device)
        y = y_t.to(device=device)

        # preprocess batch on CPU numpy, then to torch
        x_clean_np = preprocess_apply(x_raw, prep_state)
        x_clean = torch.from_numpy(x_clean_np).to(device=device, dtype=torch.float32)

        # concat device one-hot
        dev_oh = make_device_onehot(dev, num_devices=num_devices)
        x_in = torch.cat([x_clean, dev_oh], dim=1)

        pred = model(x_in)

        # supervised loss
        if target_weights_t is None:
            loss_sup = torch.mean((pred - y) ** 2)
        else:
            loss_sup = torch.mean(((pred - y) ** 2) * target_weights_t)

        loss = loss_sup

        # invariance loss (safe: warmup + detach)
        if args.lambda_inv > 0.0 and epoch >= args.inv_warmup:
            rng = np.random.RandomState(args.seed + epoch * 1000 + step)

            x1_raw = augment_spectra(
                x_raw, grid_wns, rng,
                scale_range=(args.inv_scale_lo, args.inv_scale_hi),
                baseline_amp=args.inv_baseline_amp,
                baseline_degree=args.inv_baseline_degree,
                max_shift=args.inv_max_shift,
                noise_std=args.inv_noise_std,
            )
            x2_raw = augment_spectra(
                x_raw, grid_wns, rng,
                scale_range=(args.inv_scale_lo, args.inv_scale_hi),
                baseline_amp=args.inv_baseline_amp,
                baseline_degree=args.inv_baseline_degree,
                max_shift=args.inv_max_shift,
                noise_std=args.inv_noise_std,
            )

            x1 = torch.from_numpy(preprocess_apply(x1_raw, prep_state)).to(device=device, dtype=torch.float32)
            x2 = torch.from_numpy(preprocess_apply(x2_raw, prep_state)).to(device=device, dtype=torch.float32)

            x1 = torch.cat([x1, dev_oh], dim=1)
            x2 = torch.cat([x2, dev_oh], dim=1)

            p1 = model(x1)
            p2 = model(x2)

            loss_inv = torch.mean((p1 - p2.detach()) ** 2)
            loss = loss + args.lambda_inv * loss_inv

            if args.lambda_aug_y > 0.0:
                if target_weights_t is None:
                    loss_aug = 0.5 * (torch.mean((p1 - y) ** 2) + torch.mean((p2 - y) ** 2))
                else:
                    loss_aug = 0.5 * (torch.mean(((p1 - y) ** 2) * target_weights_t) + torch.mean(((p2 - y) ** 2) * target_weights_t))
                loss = loss + args.lambda_aug_y * loss_aug

            inv_sum += float(loss_inv.item()) * len(x_raw)
            inv_count += len(x_raw)

        opt.zero_grad()
        loss.backward()
        opt.step()

        sup_sum += float(loss_sup.item()) * len(x_raw)
        n += len(x_raw)

    sup_mse = sup_sum / max(n, 1)
    inv_mse = inv_sum / max(inv_count, 1)
    return sup_mse, inv_mse


# ----------------------------
# Device-ensemble prediction for val/test
# ----------------------------

@torch.no_grad()
def predict_with_device_ensemble(
    model: nn.Module,
    X_np: np.ndarray,
    prep_state: PreprocessState,
    device: torch.device,
    num_devices: int,
    use_ensemble: bool,
) -> np.ndarray:
    """
    For transfer/test device is unknown.
    If ensemble: average predictions over all device one-hots.
    Else: use device_id=0 (arbitrary) which is usually worse.
    """
    model.eval()
    Xp = preprocess_apply(X_np, prep_state)
    Xt = torch.from_numpy(Xp).to(device=device, dtype=torch.float32)

    if not use_ensemble:
        dev_id = torch.zeros((Xt.shape[0],), dtype=torch.long, device=device)
        dev_oh = make_device_onehot(dev_id, num_devices=num_devices)
        pred = model(torch.cat([Xt, dev_oh], dim=1))
        return pred.detach().cpu().numpy().astype(np.float32)

    preds = []
    for d in range(num_devices):
        dev_id = torch.full((Xt.shape[0],), d, dtype=torch.long, device=device)
        dev_oh = make_device_onehot(dev_id, num_devices=num_devices)
        pred = model(torch.cat([Xt, dev_oh], dim=1))
        preds.append(pred.detach().cpu().numpy())

    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./outputs_v2")

    ap.add_argument("--wn_floor", type=float, default=300.0)
    ap.add_argument("--wn_ceil", type=float, default=1942.0)
    ap.add_argument("--wn_step", type=float, default=1.0)

    ap.add_argument("--plate_wn_start", type=float, default=65.0)
    ap.add_argument("--plate_wn_end", type=float, default=3350.0)

    # preprocess
    ap.add_argument("--use_msc", action="store_true")
    ap.add_argument("--baseline", choices=["none", "poly", "snip"], default="poly")
    ap.add_argument("--baseline_degree", type=int, default=3)
    ap.add_argument("--snip_max_half_window", type=int, default=20)
    ap.add_argument("--snip_smooth_half_window", type=int, default=3)

    ap.add_argument("--use_savgol", action="store_true")
    ap.add_argument("--savgol_window", type=int, default=7)
    ap.add_argument("--savgol_polyorder", type=int, default=2)
    ap.add_argument("--savgol_deriv", type=int, default=0)

    ap.add_argument("--scaling", choices=["none", "standard", "per_sample_max"], default="per_sample_max")

    # training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_stop_patience", type=int, default=20)

    ap.add_argument("--scale_y", type=int, choices=[0, 1], default=0)
    ap.add_argument("--non_negative", type=int, choices=[0, 1], default=1)

    # target weighted mse (when not scale_y)
    ap.add_argument("--target_weighted_mse", type=int, choices=[0, 1], default=1)

    # device ensemble at val/test
    ap.add_argument("--test_device_ensemble", type=int, choices=[0, 1], default=1)

    # invariance
    ap.add_argument("--lambda_inv", type=float, default=0.0)
    ap.add_argument("--lambda_aug_y", type=float, default=0.0)
    ap.add_argument("--inv_warmup", type=int, default=10)

    ap.add_argument("--inv_scale_lo", type=float, default=0.95)
    ap.add_argument("--inv_scale_hi", type=float, default=1.05)
    ap.add_argument("--inv_baseline_amp", type=float, default=0.005)
    ap.add_argument("--inv_baseline_degree", type=int, default=2)
    ap.add_argument("--inv_max_shift", type=float, default=0.2)
    ap.add_argument("--inv_noise_std", type=float, default=0.0)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Shared grid
    grid_wns = make_shared_grid(args.wn_floor, args.wn_ceil, args.wn_step)
    print("Shared grid:", float(grid_wns.min()), "..", float(grid_wns.max()), "len=", len(grid_wns))

    # Load device training data (already interpolated to grid)
    train = load_device_training(args.data_dir, grid_wns)
    print("Train:", train.X.shape, train.y.shape, "devices:", len(DEVICE_FILES))

    # Load transfer plate + test plate (plate format -> 2048 -> interpolate to grid)
    X_val_2048, y_val = load_plate_file(os.path.join(args.data_dir, "transfer_plate.csv"), is_train=True)
    X_test_2048, _ = load_plate_file(os.path.join(args.data_dir, "96_samples.csv"), is_train=False)

    X_val = interpolate_plate_to_grid(X_val_2048, grid_wns, args.plate_wn_start, args.plate_wn_end)
    X_test = interpolate_plate_to_grid(X_test_2048, grid_wns, args.plate_wn_start, args.plate_wn_end)
    y_val = y_val.astype(np.float32)

    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape)

    # Preprocess fit on TRAIN ONLY
    prep_cfg = PreprocessConfig(
        use_msc=bool(args.use_msc),
        baseline=str(args.baseline),
        baseline_degree=int(args.baseline_degree),
        snip_max_half_window=int(args.snip_max_half_window),
        snip_smooth_half_window=int(args.snip_smooth_half_window),
        use_savgol=bool(args.use_savgol),
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        savgol_deriv=int(args.savgol_deriv),
        scaling=str(args.scaling),
    )

    print("Preprocess:", prep_cfg)
    prep_state = preprocess_fit(train.X, prep_cfg)

    # Targets (optional scaling)
    y_train_np = train.y.astype(np.float32)
    y_val_np = y_val.astype(np.float32)

    y_scaler: Optional[TargetScaler] = None
    if int(args.scale_y) == 1:
        y_scaler = TargetScaler().fit(y_train_np)
        y_train_np = y_scaler.transform(y_train_np)
        y_val_np = y_scaler.transform(y_val_np)

    # If y is scaled, outputs must allow negatives
    non_negative_out = bool(args.non_negative)
    if int(args.scale_y) == 1:
        non_negative_out = False

    # Target-weighted MSE (only if not scale_y)
    target_weights_t: Optional[torch.Tensor] = None
    if int(args.scale_y) == 0 and int(args.target_weighted_mse) == 1:
        var = np.var(y_train_np, axis=0) + 1e-8
        w = (1.0 / var).astype(np.float32)
        # normalize weights so average weight ~1
        w = w / np.mean(w)
        target_weights_t = torch.tensor(w.reshape(1, 3), dtype=torch.float32, device=device)
        print("Target weights:", w)

    # Torch dataset/loader for TRAIN (keep raw on grid for augmentation)
    X_train_raw_t = torch.tensor(train.X.astype(np.float32), dtype=torch.float32)
    dev_t = torch.tensor(train.device_id.astype(np.int64), dtype=torch.long)
    y_train_t = torch.tensor(y_train_np.astype(np.float32), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_raw_t, dev_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    num_devices = len(DEVICE_FILES)
    input_dim = train.X.shape[1] + num_devices  # spectrum + one-hot device

    model = VanillaMLP(
        input_dim=input_dim,
        hidden_dims=(512, 256, 128),
        dropout=float(args.dropout),
        non_negative=non_negative_out,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # Prepare val tensors (y already scaled if scale_y)
    y_val_t = torch.tensor(y_val_np.astype(np.float32), dtype=torch.float32)

    best_rmse = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        train_mse, inv_mse = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=device,
            prep_state=prep_state,
            grid_wns=grid_wns,
            args=args,
            epoch=ep,
            num_devices=num_devices,
            target_weights_t=target_weights_t,
        )

        # val prediction using device-ensemble trick
        pred_val_scaled = predict_with_device_ensemble(
            model=model,
            X_np=X_val,
            prep_state=prep_state,
            device=device,
            num_devices=num_devices,
            use_ensemble=bool(int(args.test_device_ensemble)),
        )

        # compute val metrics in original units
        if y_scaler is not None:
            pred_val = y_scaler.inverse(pred_val_scaled)
            y_true = y_scaler.inverse(y_val_np)
        else:
            pred_val = pred_val_scaled
            y_true = y_val_np

        mse = float(np.mean((pred_val - y_true) ** 2))
        val_rmse = float(np.sqrt(mse))

        ss_res = np.sum((y_true - pred_val) ** 2, axis=0)
        mean = np.mean(y_true, axis=0, keepdims=True)
        ss_tot = np.sum((y_true - mean) ** 2, axis=0) + 1e-12
        val_r2 = 1.0 - ss_res / ss_tot

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            print(
                f"epoch={ep:03d} train_mse={train_mse:.6f} inv_mse={inv_mse:.6f} "
                f"val_rmse={val_rmse:.4f} val_r2=[{val_r2[0]:.3f}, {val_r2[1]:.3f}, {val_r2[2]:.3f}]"
            )

        if bad >= int(args.early_stop_patience):
            print(f"Early stopping at epoch {ep} (patience={args.early_stop_patience}).")
            break

    print("Best val RMSE:", best_rmse)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model payload
    payload = {
        "model_state": model.state_dict(),
        "grid_wns": grid_wns,
        "prep_cfg": prep_cfg.__dict__,
        "prep_state": {
            "mu": None if prep_state.mu is None else prep_state.mu,
            "sd": None if prep_state.sd is None else prep_state.sd,
            "msc_ref": None if prep_state.msc_ref is None else prep_state.msc_ref,
        },
        "y_scaler": None if y_scaler is None else y_scaler.state_dict(),
        "device_files": DEVICE_FILES,
        "args": vars(args),
    }

    model_path = os.path.join(args.out_dir, "model.pt")
    torch.save(payload, model_path)
    print("Saved model:", model_path)

    # Predict test (device ensemble)
    pred_test_scaled = predict_with_device_ensemble(
        model=model,
        X_np=X_test,
        prep_state=prep_state,
        device=device,
        num_devices=num_devices,
        use_ensemble=bool(int(args.test_device_ensemble)),
    )

    if y_scaler is not None:
        pred_test = y_scaler.inverse(pred_test_scaled)
    else:
        pred_test = pred_test_scaled

    # clip negatives (concentrations)
    pred_test = np.maximum(pred_test, 0.0).astype(np.float32)

    # Write submission
    sub_path = os.path.join(args.data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    out = sub.copy()
    out.loc[:, ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]] = pred_test[: len(out)]

    out_file = os.path.join(args.out_dir, "submission.csv")
    out.to_csv(out_file, index=False)
    print("Wrote:", out_file, "shape=", out.shape)

    # quick sanity stats
    print(out.describe(include="all"))


if __name__ == "__main__":
    main()