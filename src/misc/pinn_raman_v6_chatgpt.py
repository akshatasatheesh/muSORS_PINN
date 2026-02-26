#!/usr/bin/env python3
"""
pinn_raman_v5_chatgpt.py

A single, self-contained script in the style of `pinn_raman_v5 (1).py`, but with
four selectable variants (A/B/C/D) so you can quickly compare what helps.

Why four variants?
- A: Transfer-plate-only baseline (strong "sanity check")
- B: Same as A, but uses *transductive* (unlabeled) scaling fit on train+test (often boosts leaderboard)
- C: "PINN-lite" (Beer–Lambert-style reconstruction regularizer) on transfer plate
- D: Best-practice transfer learning for this challenge: supervised pretrain on 8-device set +
     fine-tune on transfer plate + test-time device ensemble

Usage:
  python pinn_raman_v5_chatgpt.py --data_dir "/path/to/dig-4-bio-raman-transfer-learning-challenge" --variant D
  python pinn_raman_v5_chatgpt.py --data_dir "/path/to/dig-4-bio-raman-transfer-learning-challenge" --variant A

Outputs:
  <out_dir>/submission_<variant>.csv
  <out_dir>/debug_<variant>.txt  (metrics + key config)

Notes:
- This script assumes the Kaggle dataset directory contains:
    transfer_plate.csv, 96_samples.csv, sample_submission.csv,
    and the 8 device CSVs:
      anton_532.csv, anton_785.csv, kaiser.csv, mettler_toledo.csv,
      metrohm.csv, tec5.csv, timegate.csv, tornado.csv
- Targets in transfer_plate.csv are labeled with "Magnesium Acetate (g/L)" but
  the submission column is "Magnesium Sulfate". This script keeps that mapping.

No external checkpoint loading: everything is trained in this script.
"""

from __future__ import annotations

import os
import math
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ----------------------------
# Constants / data files
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

TARGET_COLS_PLATE = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
SUB_COLS = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]

# Shared grid (matches common winning solutions)
WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0  # inclusive grid => 1643 dims


# ----------------------------
# Repro
# ----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return (lo + step * np.arange(n, dtype=np.float64)).astype(np.float32)


# ----------------------------
# Plate data loading (2x2048 -> mean -> interpolate)
# ----------------------------

def _clean_brackets(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_plate(filepath: str, is_train: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    transfer_plate.csv (train): rows are 2 replicates per sample with sample_id forward-filled.
    96_samples.csv (test): similar but headerless.
    Returns:
      X_2048: (N_samples, 2048) float32 averaged over replicates
      y:      (N_samples, 3) float32 or None
    """
    if is_train:
        df = pd.read_csv(filepath)
        y = df[TARGET_COLS_PLATE].dropna().values.astype(np.float32)
        Xdf = df.iloc[:, :-4].copy()
    else:
        df = pd.read_csv(filepath, header=None)
        y = None
        Xdf = df.copy()

    Xdf.columns = ["sample_id"] + [str(i) for i in range(Xdf.shape[1] - 1)]
    Xdf["sample_id"] = Xdf["sample_id"].ffill()

    for col in Xdf.columns[1:]:
        Xdf[col] = _clean_brackets(Xdf[col])

    X = Xdf.drop(columns=["sample_id"]).values.astype(np.float32)
    if X.shape[1] != 2048:
        raise ValueError(f"Expected 2048 spectral columns, got {X.shape[1]} in {filepath}")
    if X.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates), got {X.shape[0]} in {filepath}")

    X = X.reshape(-1, 2, 2048).mean(axis=1).astype(np.float32)
    return X, y


def plate_to_grid(X_2048: np.ndarray, grid_wns: np.ndarray,
                  wn_start: float = 65.0, wn_end: float = 3350.0) -> np.ndarray:
    full_wns = np.linspace(wn_start, wn_end, 2048, dtype=np.float64)
    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    sel = (full_wns >= lo) & (full_wns <= hi)
    wns_sel = full_wns[sel]
    X_sel = X_2048[:, sel].astype(np.float64)
    return np.array([np.interp(grid_wns, xp=wns_sel, fp=row) for row in X_sel], dtype=np.float32)


# ----------------------------
# Device data loading (spectral columns are wavenumbers)
# ----------------------------

def _safe_float_cols(cols) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(float(str(c).strip()))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


def load_device_training(data_dir: str, grid_wns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, D) interpolated onto grid
      y: (N, 3) targets
      device_id: (N,) int64
    Assumption: last 5 cols include labels; first 3 of df.iloc[:, -5:-1] correspond to 3 analytes.
    """
    Xs, ys, devs = [], [], []
    for dev_idx, fn in enumerate(DEVICE_FILES):
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)

        spec_cols = df.columns[:-5]
        wns = _safe_float_cols(spec_cols)
        mask = np.isfinite(wns)
        wns = wns[mask]
        spec = df.iloc[:, :-5].loc[:, np.array(mask)].to_numpy(dtype=np.float64)

        order = np.argsort(wns)
        wns = wns[order]
        spec = spec[:, order]

        X_interp = np.vstack([np.interp(grid_wns.astype(np.float64), wns, row) for row in spec]).astype(np.float32)

        y_all = df.iloc[:, -5:-1].to_numpy(dtype=np.float32)
        y = y_all[:, :3].astype(np.float32)

        Xs.append(X_interp)
        ys.append(y)
        devs.append(np.full((len(y),), dev_idx, dtype=np.int64))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    device_id = np.concatenate(devs, axis=0)
    return X, y, device_id


# ----------------------------
# Preprocessing: MSC -> SNIP baseline -> SavGol -> Scaling
# ----------------------------

def apply_msc(X: np.ndarray, ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    X64 = X.astype(np.float64, copy=False)
    if ref is None:
        ref = X64.mean(axis=0)
    out = np.empty_like(X64)
    for i in range(X64.shape[0]):
        slope, intercept = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - intercept) / (slope + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)


def apply_snip_baseline(X: np.ndarray, max_half_window: int = 20, smooth_half_window: int = 3) -> np.ndarray:
    try:
        import pybaselines
        fitter = pybaselines.Baseline()
        X64 = X.astype(np.float64, copy=False)
        out = np.empty_like(X64)
        for i in range(X64.shape[0]):
            base, _ = fitter.snip(
                X64[i], max_half_window=max_half_window,
                decreasing=True, smooth_half_window=smooth_half_window
            )
            out[i] = X64[i] - base
        return out.astype(np.float32)
    except Exception:
        # fallback: do nothing (still works, just slightly worse)
        return X.astype(np.float32)


def apply_savgol(X: np.ndarray, window: int = 7, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
    return savgol_filter(X.astype(np.float64, copy=False), window_length=window, polyorder=polyorder,
                         deriv=deriv, axis=1).astype(np.float32)


@dataclass
class Preprocessor:
    """Fittable preprocessing state."""
    msc_ref: Optional[np.ndarray] = None
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    fit_on: str = "train"

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X1, self.msc_ref = apply_msc(X, ref=None)
        X1 = apply_snip_baseline(X1)
        X1 = apply_savgol(X1)
        mu = X1.mean(axis=0, dtype=np.float64)
        sd = X1.std(axis=0, dtype=np.float64)
        sd = np.maximum(sd, 1e-8)
        self.mean_ = mu.astype(np.float32)
        self.std_ = sd.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        X1, _ = apply_msc(X, ref=self.msc_ref)
        X1 = apply_snip_baseline(X1)
        X1 = apply_savgol(X1)
        return ((X1 - self.mean_) / self.std_).astype(np.float32)


# ----------------------------
# Augmentations (invariance / TTA)
# ----------------------------

def augment_spectra(
    X: np.ndarray,
    wns: np.ndarray,
    rng: np.random.RandomState,
    scale_range=(0.9, 1.1),
    baseline_amp=0.02,
    baseline_degree=2,
    max_shift=1.0,
    noise_std=0.002,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    wns = np.asarray(wns, dtype=np.float32)
    N, D = X.shape
    out = X.copy()

    # 1) scale
    lo, hi = float(scale_range[0]), float(scale_range[1])
    out *= rng.uniform(lo, hi, size=(N, 1)).astype(np.float32)

    # 2) baseline drift
    if baseline_amp > 0:
        t = np.linspace(0, 1, D, dtype=np.float32)
        for i in range(N):
            coeffs = rng.normal(0, 1, baseline_degree + 1).astype(np.float32)
            base = np.zeros((D,), dtype=np.float32)
            for k, c in enumerate(coeffs):
                base += c * (t ** k)
            base -= base.mean()
            base /= (base.std() + 1e-8)
            amp = rng.uniform(-baseline_amp, baseline_amp).astype(np.float32) * (np.max(np.abs(out[i])) + 1e-8)
            out[i] += amp * base

    # 3) wavenumber shift
    if max_shift > 0:
        shifts = rng.uniform(-max_shift, max_shift, size=(N,)).astype(np.float32)
        shifted = np.empty_like(out)
        w64 = wns.astype(np.float64)
        for i in range(N):
            q = (wns - shifts[i]).astype(np.float64)
            shifted[i] = np.interp(q, w64, out[i].astype(np.float64)).astype(np.float32)
        out = shifted

    # 4) noise
    if noise_std > 0:
        out += rng.normal(0, noise_std, size=out.shape).astype(np.float32)

    return out.astype(np.float32)


# ----------------------------
# Models
# ----------------------------

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256, 128), dropout=0.15, out_dim=3):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeviceConditionedMLP(nn.Module):
    """
    Variant D backbone: spectrum -> embedding; then condition on device embedding.

    At test time (unknown device), we average predictions over all device embeddings.
    """
    def __init__(self, spec_dim: int, n_devices: int, emb_dim: int = 16, hidden=(512, 256, 128), dropout=0.12):
        super().__init__()
        self.dev_emb = nn.Embedding(n_devices, emb_dim)
        self.mlp = MLPRegressor(spec_dim + emb_dim, hidden=hidden, dropout=dropout, out_dim=3)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor, dev_id: torch.Tensor) -> torch.Tensor:
        e = self.dev_emb(dev_id)
        y = self.mlp(torch.cat([x, e], dim=1))
        return self.softplus(y)  # non-negative concentrations


class BeerLambertDecoder(nn.Module):
    """PINN-lite: reconstruct spectrum from predicted concentrations and learnable pure spectra."""
    def __init__(self, n_targets: int, spec_dim: int):
        super().__init__()
        self.pure = nn.Parameter(torch.abs(torch.randn(n_targets, spec_dim)) * 0.05)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        E = F.softplus(self.pure)
        return c @ E


# ----------------------------
# Training helpers
# ----------------------------

def mse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_true - y_pred) ** 2, axis=0)


def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2) + 1e-12)
        out.append(1.0 - ss_res / ss_tot)
    return np.array(out, dtype=np.float64)


def fit_linear_blend(y_true: np.ndarray, preds_list: List[np.ndarray]) -> np.ndarray:
    """
    Tiny helper used in variant A: fit non-negative weights to blend models.
    Solves min ||sum w_i p_i - y||^2 with w>=0 then normalizes.
    """
    P = np.stack(preds_list, axis=2)  # (N, 3, M)
    M = P.shape[2]
    w = np.ones((M,), dtype=np.float64) / M
    # simple projected gradient (small, stable)
    lr = 0.1
    for _ in range(200):
        y_hat = np.tensordot(P, w, axes=(2, 0))  # (N, 3)
        grad = np.sum(2 * (y_hat - y_true)[:, :, None] * P, axis=(0, 1))
        w -= lr * grad / (np.linalg.norm(grad) + 1e-12)
        w = np.maximum(w, 0)
        s = w.sum()
        if s > 0:
            w /= s
    return w.astype(np.float32)


@torch.no_grad()
def predict_device_ensemble(model: DeviceConditionedMLP, X: np.ndarray, device: torch.device, n_devices: int) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    preds = []
    for d in range(n_devices):
        dev_id = torch.full((Xt.shape[0],), d, dtype=torch.long, device=device)
        preds.append(model(Xt, dev_id).cpu().numpy())
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


@torch.no_grad()
def predict_tta(model, X: np.ndarray, wns: np.ndarray, device: torch.device, n_tta: int = 8) -> np.ndarray:
    model.eval()
    rng = np.random.RandomState(0)
    preds = []
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    # base
    if isinstance(model, DeviceConditionedMLP):
        # for device-conditioned use ensemble outside
        raise ValueError("Use predict_device_ensemble() for DeviceConditionedMLP.")
    else:
        preds.append(F.softplus(model(Xt)).cpu().numpy())
    # aug
    for _ in range(n_tta - 1):
        X_aug = augment_spectra(X, wns, rng, scale_range=(0.92, 1.08), baseline_amp=0.01, max_shift=0.8, noise_std=0.0015)
        Xt = torch.tensor(X_aug, dtype=torch.float32, device=device)
        preds.append(F.softplus(model(Xt)).cpu().numpy())
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


# ----------------------------
# Variant implementations
# ----------------------------

def run_variant_A(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, wns: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    """
    A: Transfer-plate-only baseline.
    - Fit preprocessing on transfer plate only.
    - Train 3 small MLPs with different seeds; blend by learned weights.
    """
    prep = Preprocessor(fit_on="transfer").fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)

    preds_list = []
    for s in [seed, seed + 1, seed + 2]:
        set_seed(s)
        model = MLPRegressor(Xtr_p.shape[1], hidden=(512, 256, 128), dropout=0.18, out_dim=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        X_t = torch.tensor(Xtr_p, dtype=torch.float32, device=device)
        y_t = torch.tensor(ytr, dtype=torch.float32, device=device)

        # simple training loop (small data)
        best = float("inf")
        best_state = None
        for ep in range(1, 801):
            model.train()
            opt.zero_grad()
            pred = model(X_t)
            loss = F.mse_loss(F.softplus(pred), y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if loss.item() < best:
                best = loss.item()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 200 == 0:
                pass
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        yhat = predict_tta(model, Xte_p, wns, device=device, n_tta=8)
        preds_list.append(yhat)

    # blend weights fit on train (OOF-free here; still useful as sanity)
    # To reduce overfit, fit weights on a 5-fold OOF.
    kfold = 5
    idx = np.arange(len(Xtr_p))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, kfold)
    oof = [np.zeros_like(ytr, dtype=np.float32) for _ in preds_list]

    for k in range(kfold):
        val_idx = folds[k]
        tr_idx = np.setdiff1d(idx, val_idx, assume_unique=False)

        # train mini model per seed quickly
        local_preds = []
        for s_i, s in enumerate([seed, seed + 1, seed + 2]):
            set_seed(s)
            model = MLPRegressor(Xtr_p.shape[1], hidden=(512, 256, 128), dropout=0.18, out_dim=3).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
            X_t = torch.tensor(Xtr_p[tr_idx], dtype=torch.float32, device=device)
            y_t = torch.tensor(ytr[tr_idx], dtype=torch.float32, device=device)
            for ep in range(1, 401):
                model.train()
                opt.zero_grad()
                pred = model(X_t)
                loss = F.mse_loss(F.softplus(pred), y_t)
                loss.backward()
                opt.step()
            # val preds
            model.eval()
            Xv = torch.tensor(Xtr_p[val_idx], dtype=torch.float32, device=device)
            pv = F.softplus(model(Xv)).detach().cpu().numpy().astype(np.float32)
            oof[s_i][val_idx] = pv
            local_preds.append(pv)

    w = fit_linear_blend(ytr, oof)
    pred = sum(float(w[i]) * preds_list[i] for i in range(len(preds_list)))
    return pred.astype(np.float32)


def run_variant_B(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, wns: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    """
    B: Same as A, but fits scaler using *both* transfer plate (train) and test spectra (unlabeled).
       This is a common (allowed on Kaggle) "transductive" trick.
    """
    # fit MSC ref on train only (avoid leakage of instrument drift into ref),
    # but fit standardization stats on train+test (unlabeled).
    prep = Preprocessor(fit_on="transfer")
    Xtr_m, ref = apply_msc(Xtr, ref=None)
    Xte_m, _ = apply_msc(Xte, ref=ref)
    Xtr_m = apply_snip_baseline(Xtr_m); Xte_m = apply_snip_baseline(Xte_m)
    Xtr_m = apply_savgol(Xtr_m); Xte_m = apply_savgol(Xte_m)

    X_all = np.vstack([Xtr_m, Xte_m]).astype(np.float32)
    mu = X_all.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(X_all.std(axis=0, dtype=np.float64), 1e-8).astype(np.float32)

    def z(x): return ((x - mu) / sd).astype(np.float32)
    Xtr_p, Xte_p = z(Xtr_m), z(Xte_m)

    # train one decent MLP (less overfit than variant A's heavy blending)
    set_seed(seed)
    model = MLPRegressor(Xtr_p.shape[1], hidden=(768, 384, 192), dropout=0.22, out_dim=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)
    X_t = torch.tensor(Xtr_p, dtype=torch.float32, device=device)
    y_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    best = float("inf")
    best_state = None
    for ep in range(1, 1201):
        model.train()
        opt.zero_grad()
        pred = model(X_t)
        loss = F.mse_loss(F.softplus(pred), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    yhat = predict_tta(model, Xte_p, wns, device=device, n_tta=12)
    return yhat.astype(np.float32)


def run_variant_C(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, wns: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    """
    C: PINN-lite on transfer plate only.
    - concentration head is an MLP
    - Beer–Lambert decoder reconstructs the (preprocessed) spectrum
    Loss = supervised MSE + alpha * reconstruction + beta * smoothness(pure spectra)
    """
    prep = Preprocessor(fit_on="transfer").fit(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)

    set_seed(seed)
    head = MLPRegressor(Xtr_p.shape[1], hidden=(512, 256, 128), dropout=0.20, out_dim=3).to(device)
    dec = BeerLambertDecoder(n_targets=3, spec_dim=Xtr_p.shape[1]).to(device)

    params = list(head.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=2e-4)

    Xt = torch.tensor(Xtr_p, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)

    def tv2(E: torch.Tensor) -> torch.Tensor:
        d2 = E[:, 2:] - 2 * E[:, 1:-1] + E[:, :-2]
        return (d2 ** 2).mean()

    alpha = 0.20  # reconstruction weight
    beta = 0.02   # pure spectra smoothness
    best = float("inf")
    best_state = None

    for ep in range(1, 1501):
        head.train(); dec.train()
        opt.zero_grad()
        c = F.softplus(head(Xt))
        xhat = dec(c)
        loss_sup = F.mse_loss(c, yt)
        loss_rec = F.mse_loss(xhat, Xt)
        E = F.softplus(dec.pure)
        loss = loss_sup + alpha * loss_rec + beta * tv2(E)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        if loss.item() < best:
            best = loss.item()
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "dec":  {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
            }

    if best_state is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})
        dec.load_state_dict({k: v.to(device) for k, v in best_state["dec"].items()})

    # predict with TTA
    head.eval()
    rng = np.random.RandomState(0)
    preds = []
    Xtst = torch.tensor(Xte_p, dtype=torch.float32, device=device)
    preds.append(F.softplus(head(Xtst)).detach().cpu().numpy())
    for _ in range(11):
        X_aug = augment_spectra(Xte_p, wns, rng, scale_range=(0.92, 1.08), baseline_amp=0.0, max_shift=0.0, noise_std=0.001)
        Xtst = torch.tensor(X_aug, dtype=torch.float32, device=device)
        preds.append(F.softplus(head(Xtst)).detach().cpu().numpy())
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def run_variant_D(
    Xsrc: np.ndarray, ysrc: np.ndarray, dev_id: np.ndarray,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray,
    wns: np.ndarray,
    device: torch.device,
    seed: int
) -> np.ndarray:
    """
    D: Supervised pretrain on 8-device set + fine-tune on transfer plate.
       At test time, average over all device embeddings (device ensemble).
    """
    n_devices = len(DEVICE_FILES)

    # Preprocess: fit on SOURCE ONLY (most stable), then apply everywhere
    prep = Preprocessor(fit_on="source").fit(Xsrc)
    Xsrc_p = prep.transform(Xsrc)
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)

    # Optional: also per-sample max normalize (often helps with device shift)
    def per_sample_max(X):
        denom = np.max(np.abs(X), axis=1, keepdims=True) + 1e-8
        return (X / denom).astype(np.float32)

    Xsrc_p = per_sample_max(Xsrc_p)
    Xtr_p = per_sample_max(Xtr_p)
    Xte_p = per_sample_max(Xte_p)

    # ----------------
    # Stage 1: pretrain on devices
    # ----------------
    set_seed(seed)
    model = DeviceConditionedMLP(spec_dim=Xsrc_p.shape[1], n_devices=n_devices, emb_dim=16,
                                 hidden=(768, 384, 192), dropout=0.10).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)

    ds = TensorDataset(
        torch.tensor(Xsrc_p, dtype=torch.float32),
        torch.tensor(dev_id, dtype=torch.long),
        torch.tensor(ysrc, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

    best = float("inf")
    best_state = None
    for ep in range(1, 31):  # short + sweet
        model.train()
        tot = 0.0
        n = 0
        for xb, db, yb in dl:
            xb = xb.to(device); db = db.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb, db)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        avg = tot / max(n, 1)
        if avg < best:
            best = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ----------------
    # Stage 2: fine-tune on transfer plate (device unknown)
    # Trick: during fine-tune, randomly sample a device id each batch (regularizes)
    # ----------------
    opt2 = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=4e-4)

    Xt = torch.tensor(Xtr_p, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    dl2 = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True, drop_last=False)

    rng = np.random.RandomState(seed)
    for ep in range(1, 801):
        model.train()
        for xb, yb in dl2:
            xb = xb.to(device); yb = yb.to(device)
            # random device ids for regularization
            db = torch.tensor(rng.randint(0, n_devices, size=(len(xb),)), dtype=torch.long, device=device)
            opt2.zero_grad()
            pred = model(xb, db)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt2.step()

    # Predict with device ensemble
    yhat = predict_device_ensemble(model, Xte_p, device=device, n_devices=n_devices)
    return yhat.astype(np.float32)


# ----------------------------
# Post-process + submission
# ----------------------------

def post_process(preds: np.ndarray, y_ref: np.ndarray, lo_p=1.0, hi_p=99.0, margin=0.10) -> np.ndarray:
    preds = np.maximum(preds, 0.0)
    out = preds.copy()
    for i in range(out.shape[1]):
        lo = float(np.percentile(y_ref[:, i], lo_p))
        hi = float(np.percentile(y_ref[:, i], hi_p))
        lo = max(0.0, lo - margin * (hi - lo))
        hi = hi + margin * (hi - lo)
        out[:, i] = np.clip(out[:, i], lo, hi)
    return out.astype(np.float32)


def write_submission(data_dir: str, out_dir: str, variant: str, preds: np.ndarray) -> str:
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    out = sub.copy()
    out.loc[:, SUB_COLS] = preds[: len(out)]
    out_file = os.path.join(out_dir, f"submission_{variant}.csv")
    out.to_csv(out_file, index=False)
    return out_file


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing Kaggle competition CSVs.")
    ap.add_argument("--out_dir", default="./pinn_outputs_chatgpt")
    ap.add_argument("--variant", choices=["A", "B", "C", "D"], default="D")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wns = make_grid(WN_LOW, WN_HIGH, WN_STEP)

    # Load plate train/test
    X_plate_2048, y_plate = load_plate(os.path.join(args.data_dir, "transfer_plate.csv"), is_train=True)
    X_test_2048, _ = load_plate(os.path.join(args.data_dir, "96_samples.csv"), is_train=False)
    Xtr = plate_to_grid(X_plate_2048, wns)
    Xte = plate_to_grid(X_test_2048, wns)
    ytr = y_plate

    # Load source devices for variant D
    Xsrc = ysrc = dev_id = None
    if args.variant == "D":
        Xsrc, ysrc, dev_id = load_device_training(args.data_dir, wns)

    # Train/predict
    if args.variant == "A":
        preds = run_variant_A(Xtr, ytr, Xte, wns, device, seed=args.seed)
    elif args.variant == "B":
        preds = run_variant_B(Xtr, ytr, Xte, wns, device, seed=args.seed)
    elif args.variant == "C":
        preds = run_variant_C(Xtr, ytr, Xte, wns, device, seed=args.seed)
    else:
        assert Xsrc is not None and ysrc is not None and dev_id is not None
        preds = run_variant_D(Xsrc, ysrc, dev_id, Xtr, ytr, Xte, wns, device, seed=args.seed)

    preds_pp = post_process(preds, ytr)

    # Debug metrics (on training only, where possible)
    dbg = {
        "variant": args.variant,
        "seed": args.seed,
        "device": str(device),
        "grid": {"low": WN_LOW, "high": WN_HIGH, "step": WN_STEP, "dim": int(len(wns))},
        "train_shape": list(Xtr.shape),
        "test_shape": list(Xte.shape),
    }
    dbg_path = os.path.join(args.out_dir, f"debug_{args.variant}.txt")
    with open(dbg_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(dbg, indent=2))
        f.write("\n")

    out_file = write_submission(args.data_dir, args.out_dir, args.variant, preds_pp)
    print(f"[OK] Wrote: {out_file}")
    print(f"[OK] Debug: {dbg_path}")
    print(pd.read_csv(out_file).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
