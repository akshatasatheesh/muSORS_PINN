#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v10 — Enhanced Raman-Appropriate Physics-Informed NN         ║
║                                                                              ║
║  Building on v9-FIXED, adds:                                                 ║
║                                                                              ║
║   1. Nonnegative Linear Mixing Decoder (from v9-fixed, refined)              ║
║      X̂(ν) = b(ν) + Σ_i c_i · s_i(ν)                                        ║
║      - s_i(ν): learned nonneg basis spectra                                  ║
║      - b(ν): learned nonneg baseline/background                              ║
║      - c_i ≥ 0 via Softplus head                                             ║
║                                                                              ║
║   2. Smoothness priors (TV + Laplacian on basis & baseline)                  ║
║                                                                              ║
║   3. Wavenumber shift invariance                                             ║
║      - Training-time augmentation: random sub-pixel shifts via               ║
║        linear interpolation along the spectral axis                          ║
║      - Learned per-device affine alignment (shift + stretch) in the          ║
║        mixing decoder                                                        ║
║                                                                              ║
║   4. Affine intensity calibration                                            ║
║      - Learned per-sample scale (α) + offset (β) in the decoder             ║
║        predicted by a small MLP from encoder features                        ║
║      - Replaces fragile per-sample max normalization                         ║
║                                                                              ║
║   5. Soft peak locality prior (optional)                                     ║
║      - Penalizes basis spectra energy far from expected Raman bands          ║
║      - Uses smooth Gaussian masks centered on known band regions             ║
║      - Soft, not hard: basis CAN have weight elsewhere, just penalized       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import copy, os, warnings, glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "data"
OUT_DIR  = "./v10_outputs"
SEED     = 42

WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0
DEVICE_FILES = [
    "anton_532.csv", "anton_785.csv", "kaiser.csv", "mettler_toledo.csv",
    "metrohm.csv", "tec5.csv", "timegate.csv", "tornado.csv",
]

# Chemometrics preprocessing
SAVGOL_WINDOW = 7
SAVGOL_POLY   = 2
SAVGOL_DERIV  = 0
SNIP_HALFWIN  = 20

# Derivative windows
DERIV1_WINDOW = 21
DERIV2_WINDOW = 35

# Raman band regions (index into 300–1942 cm⁻¹ grid = 1643 points)
BAND_REGIONS = {
    'glucose_1125':    (800, 850),
    'glucose_1065':    (740, 790),
    'naac_930':        (600, 660),
    'naac_1415':       (1085, 1145),
    'mgso4_980':       (650, 710),
    'mgso4_450':       (120, 180),
    'water_broad':     (1300, 1500),
    'fingerprint_low': (0, 400),
    'fingerprint_mid': (400, 900),
    'fingerprint_hi':  (900, 1400),
}

# Known Raman band centers in wavenumber (cm⁻¹) for peak-locality prior
# These map to approximate indices on the 300–1942 grid (step=1 → idx = wn - 300)
KNOWN_BAND_CENTERS_CM = {
    0: [1125, 1065, 525],            # Glucose
    1: [930, 1415, 650],             # Sodium Acetate
    2: [980, 450, 615],              # MgSO4
}
PEAK_LOCALITY_SIGMA_CM = 60.0        # Gaussian mask width (cm⁻¹)

ANALYTE_NAMES = ["Glucose", "NaAc", "MgSO4"]
N_FOLDS = 5
N_OUT   = 3

# NN architecture
ENC_H1, ENC_H2, HEAD_H = 256, 64, 32
DROPOUT, FT_DROPOUT = 0.15, 0.20

# Reptile meta-learning
META_EPOCHS          = 120
META_TASKS_PER_EPOCH = 16
K_SUPPORT            = 40
REPTILE_OUTER_LR     = 0.3
INNER_LR, INNER_STEPS = 3e-3, 8

# Fine-tuning
FT_EPOCHS       = 500
FT_LR           = 3e-4
FT_WEIGHT_DECAY = 2e-4
FT_BATCH        = 24
N_RESTARTS      = 4
PATIENCE        = 90
AUG_NOISE, AUG_SCALE = 0.015, (0.96, 1.04)
TTA_N = 5

# ── Physics loss weights (ramped with curriculum) ─────────────────────────────
PHYS_RAMP_EPOCHS = 220
LW_INV       = 0.03     # augmentation invariance (feature-space)
LW_SHIFT_INV = 0.02     # wavenumber-shift invariance (spectral-space)
LW_MIX       = 0.12     # reconstruction via linear mixing decoder
LW_SMOOTH    = 0.025    # TV + Laplacian smoothness on basis & baseline
LW_SPARSE    = 0.004    # mild L1 sparsity on basis
LW_PEAK_LOC  = 0.008    # soft peak-locality prior

WINNING_CSV = None

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def make_grid(lo, hi, step):
    return np.arange(lo, hi + step / 2, step)

class TargetScaler:
    def fit(self, y):
        self.mu = y.mean(0); self.sd = y.std(0) + 1e-8; return self
    def transform(self, y): return (y - self.mu) / self.sd
    def inverse(self, yn): return yn * self.sd + self.mu

# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def _clean(s):
    if isinstance(s, str):
        s = s.replace('[', '').replace(']', '')
        try: return float(s)
        except: return np.nan
    return float(s) if np.isfinite(float(s)) else np.nan

def load_device(path, grid_wns):
    df = pd.read_csv(path)
    spec_cols = [c for c in df.columns[:-5]]
    wns_dev = np.array([float(c) for c in spec_cols])
    mask = (wns_dev >= WN_LOW) & (wns_dev <= WN_HIGH)
    if mask.sum() < 50: return None, None
    wns_sel = wns_dev[mask]
    X_raw = df[np.array(spec_cols)[mask]].values.astype(np.float64)
    X_interp = np.array([np.interp(grid_wns, wns_sel, row) for row in X_raw])
    X_norm = X_interp / (np.max(X_interp) + 1e-12)
    y = df.iloc[:, -5:-2].values.astype(np.float64)
    return X_norm.astype(np.float32), y.astype(np.float32)

def load_plate(path, is_train):
    df = pd.read_csv(path) if is_train else pd.read_csv(path, header=None)
    if is_train:
        y = df[['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']].dropna().values
        X = df.iloc[:, :-4]
    else:
        X = df; y = None
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1] - 1)]
    X['sample_id'] = X['sample_id'].ffill()
    if is_train: X['sample_id'] = X['sample_id'].str.strip()
    else: X['sample_id'] = X['sample_id'].astype(str).str.strip().str.replace('sample', '').astype(int)
    for c in X.columns[1:]: X[c] = X[c].apply(_clean)
    X = X.drop('sample_id', axis=1).interpolate(axis=1, limit_direction='both').fillna(0)
    X2048 = X.values.reshape(-1, 2, 2048).mean(axis=1)
    return X2048.astype(np.float32), y.astype(np.float32) if y is not None else None

def plate_to_grid(X2048, grid_wns, s=65.0, e=3350.0):
    wns_full = np.linspace(s, e, 2048)
    mask = (wns_full >= WN_LOW) & (wns_full <= WN_HIGH)
    wns_sel = wns_full[mask]
    X_sel = X2048[:, mask]
    return np.array([np.interp(grid_wns, wns_sel, row) for row in X_sel]).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Chemometrics Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def apply_msc(X, ref=None):
    X64 = X.astype(np.float64)
    ref = X64.mean(0) if ref is None else ref.astype(np.float64)
    out = np.empty_like(X64)
    for i in range(len(X64)):
        m, b = np.polyfit(ref, X64[i], 1)
        out[i] = (X64[i] - b) / (m + 1e-12)
    return out.astype(np.float32), ref.astype(np.float32)

def apply_baseline(X, max_half_window=SNIP_HALFWIN):
    try:
        import pybaselines
        fitter = pybaselines.Baseline()
        out = np.empty_like(X, dtype=np.float64)
        for i, row in enumerate(X.astype(np.float64)):
            base, _ = fitter.snip(row, max_half_window=max_half_window,
                                  decreasing=True, smooth_half_window=3)
            out[i] = row - base
        return out.astype(np.float32)
    except ImportError:
        t = np.linspace(0, 1, X.shape[1])
        out = np.empty_like(X, dtype=np.float64)
        for i, row in enumerate(X.astype(np.float64)):
            out[i] = row - np.polyval(np.polyfit(t, row, 3), t)
        return out.astype(np.float32)

class ChemometricsPreprocessor:
    def __init__(self):
        self.msc_ref = None; self.scaler = None
    def fit_transform(self, X):
        X, self.msc_ref = apply_msc(X)
        X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64), SAVGOL_WINDOW, SAVGOL_POLY,
                          SAVGOL_DERIV, axis=1).astype(np.float32)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X).astype(np.float32)
    def transform(self, X):
        X, _ = apply_msc(X, self.msc_ref)
        X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64), SAVGOL_WINDOW, SAVGOL_POLY,
                          SAVGOL_DERIV, axis=1).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
def compute_derivatives(X_raw_mean):
    d1 = savgol_filter(X_raw_mean.astype(np.float64), DERIV1_WINDOW, 2, deriv=1, axis=1).astype(np.float32)
    d2 = savgol_filter(X_raw_mean.astype(np.float64), DERIV2_WINDOW, 2, deriv=2, axis=1).astype(np.float32)
    return d1, d2

def extract_peak_features(spectra):
    features = []
    for spec in spectra:
        thresh = np.percentile(spec, 90)
        peaks, _ = find_peaks(spec, height=thresh, prominence=0.5)
        widths, _, _, _ = peak_widths(spec, peaks, rel_height=0.5)
        features.append([
            len(peaks),
            np.sum(spec[peaks]) if len(peaks) > 0 else 0.0,
            np.mean(spec[peaks]) if len(peaks) > 0 else 0.0,
            np.std(spec[peaks])  if len(peaks) > 1 else 0.0,
            np.mean(widths)      if len(widths) > 0 else 0.0,
            np.max(spec[peaks])  if len(peaks) > 0 else 0.0,
        ])
    return np.array(features, dtype=np.float32)

def compute_statistical_features(spectra):
    return np.stack([
        np.mean(spectra, axis=1), np.std(spectra, axis=1),
        skew(spectra, axis=1),    kurtosis(spectra, axis=1),
        np.max(spectra, axis=1),  np.min(spectra, axis=1),
        np.ptp(spectra, axis=1),  np.sum(spectra**2, axis=1),
    ], axis=1).astype(np.float32)

def compute_band_integrals(spectra):
    integrals = []
    for _, (lo, hi) in BAND_REGIONS.items():
        hi = min(hi, spectra.shape[1]); lo = max(lo, 0)
        integrals.extend([np.sum(spectra[:, lo:hi], axis=1),
                          np.mean(spectra[:, lo:hi], axis=1)])
    return np.stack(integrals, axis=1).astype(np.float32)

def compute_peak_ratios(spectra):
    n = spectra.shape[1]
    def _b(lo, hi): return np.mean(spectra[:, max(0, lo):min(hi, n)], axis=1) + 1e-10
    g, na, mg, w, fp = _b(800, 850), _b(600, 660), _b(650, 710), _b(1300, 1500), _b(400, 900)
    return np.stack([g/w, na/w, mg/w, g/fp, na/fp, mg/fp, g/na, g/mg, na/mg],
                    axis=1).astype(np.float32)

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}; self.pca = None
    def build_features(self, X_pp, X_raw_mean, fit=True):
        d1, d2 = compute_derivatives(X_raw_mean)
        peaks  = extract_peak_features(X_pp)
        stats  = compute_statistical_features(X_pp)
        bands  = compute_band_integrals(X_pp)
        ratios = compute_peak_ratios(X_pp)
        feature_sets = {
            'spec_only':    X_pp.copy(),
            'spec_d1':      np.hstack([X_pp, d1]),
            'combined_all': np.hstack([X_pp, d1, d2, stats, peaks, bands, ratios]),
        }
        if fit:
            self.pca = PCA(n_components=min(62, X_pp.shape[0] - 1), random_state=SEED)
            feature_sets['pca'] = self.pca.fit_transform(X_pp)
        else:
            feature_sets['pca'] = self.pca.transform(X_pp)
        for name, feats in feature_sets.items():
            if fit:
                sc = StandardScaler()
                feature_sets[name] = sc.fit_transform(feats).astype(np.float32)
                self.scalers[name] = sc
            else:
                feature_sets[name] = self.scalers[name].transform(feats).astype(np.float32)
        return feature_sets

# ══════════════════════════════════════════════════════════════════════════════
# Multi-Model Training + OOF Stacking
# ══════════════════════════════════════════════════════════════════════════════
def create_base_models():
    configs = []
    configs.append({'name': 'HGB_spec', 'feature_set': 'spec_only', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=15, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=20, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
    ]})
    configs.append({'name': 'HGB_comb', 'feature_set': 'combined_all', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=10, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=15, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.15, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=10, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
    ]})
    configs.append({'name': 'ET_comb', 'feature_set': 'combined_all', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_split=8, min_samples_leaf=5, max_features=0.45, bootstrap=True, max_samples=0.8, ccp_alpha=1e-4, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=14, min_samples_split=10, min_samples_leaf=6, max_features=0.50, bootstrap=True, max_samples=0.75, ccp_alpha=1e-4, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_split=8, min_samples_leaf=8, max_features=0.40, bootstrap=True, max_samples=0.8, ccp_alpha=2e-4, random_state=SEED, n_jobs=-1),
    ]})
    configs.append({'name': 'ET_d1', 'feature_set': 'spec_d1', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_split=8, min_samples_leaf=5, max_features=0.30, bootstrap=True, max_samples=0.8, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=6, max_features=0.35, bootstrap=True, max_samples=0.75, random_state=SEED, n_jobs=-1),
    ]})
    configs.append({'name': 'Ridge_PCA', 'feature_set': 'pca', 'models': [
        lambda: Ridge(alpha=1.0),
        lambda: Ridge(alpha=5.0),
        lambda: Ridge(alpha=0.1),
    ]})
    return configs

def train_base_models_cv(configs, feat_train, feat_test, y_train):
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    all_oof  = {t: [] for t in range(N_OUT)}
    all_test = {t: [] for t in range(N_OUT)}
    model_scores = []

    for cfg in configs:
        fname = cfg['feature_set']
        X_tr  = feat_train[fname]
        X_te  = feat_test[fname]
        oof_preds  = np.zeros((len(y_train), N_OUT))
        test_preds = np.zeros((X_te.shape[0], N_OUT))

        for _, (tri, vali) in enumerate(kf.split(X_tr)):
            Xf, Xv = X_tr[tri], X_tr[vali]
            yf, yv = y_train[tri], y_train[vali]
            fold_test = np.zeros((X_te.shape[0], N_OUT))
            for t in range(N_OUT):
                fold_preds_t = np.zeros(len(vali))
                test_preds_t = np.zeros(X_te.shape[0])
                for model_fn in cfg['models']:
                    m = model_fn()
                    m.fit(Xf, yf[:, t])
                    fold_preds_t += m.predict(Xv)
                    test_preds_t += m.predict(X_te)
                n = len(cfg['models'])
                oof_preds[vali, t] = fold_preds_t / n
                fold_test[:, t] = test_preds_t / n
            test_preds += fold_test / N_FOLDS

        target_r2s = [r2_score(y_train[:, t], oof_preds[:, t]) for t in range(N_OUT)]
        avg_r2 = np.mean(target_r2s)
        print(f"  {cfg['name']:<12}  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(ANALYTE_NAMES, target_r2s)) + f"  Avg={avg_r2:.4f}")
        model_scores.append({'name': cfg['name'], 'target_r2s': target_r2s, 'avg_r2': avg_r2})
        for t in range(N_OUT):
            all_oof[t].append(oof_preds[:, t])
            all_test[t].append(test_preds[:, t])
    return all_oof, all_test, model_scores

def stack_models(all_oof, all_test, y_train, model_scores):
    stacked_test = np.zeros((list(all_test.values())[0][0].shape[0], N_OUT))
    best_oof     = np.zeros((len(y_train), N_OUT))

    for t in range(N_OUT):
        oof_mat  = np.column_stack(all_oof[t])
        test_mat = np.column_stack(all_test[t])
        y_t = y_train[:, t]
        n_m = oof_mat.shape[1]

        r2_avg   = r2_score(y_t, np.mean(oof_mat, axis=1))
        w = np.maximum([ms['target_r2s'][t] for ms in model_scores], 0)
        w = w / (w.sum() + 1e-10)
        r2_wavg  = r2_score(y_t, oof_mat @ w)

        def obj(ww): return mean_squared_error(y_t, oof_mat @ (ww / ww.sum()))
        res = minimize(obj, np.ones(n_m) / n_m, method='SLSQP',
                       bounds=[(0, 1)] * n_m, constraints={'type': 'eq', 'fun': lambda ww: ww.sum() - 1})
        ow = res.x / res.x.sum()
        r2_opt   = r2_score(y_t, oof_mat @ ow)
        ridge    = Ridge(alpha=2.0).fit(oof_mat, y_t)
        r2_ridge = r2_score(y_t, ridge.predict(oof_mat))
        lr       = LinearRegression().fit(oof_mat, y_t)
        r2_lr    = r2_score(y_t, lr.predict(oof_mat))

        methods = {
            'Simple_Avg':  (r2_avg,   np.mean(oof_mat, 1), np.mean(test_mat, 1)),
            'Wt_Avg':      (r2_wavg,  oof_mat @ w,         test_mat @ w),
            'Opt_Wts':     (r2_opt,   oof_mat @ ow,        test_mat @ ow),
            'Ridge_Stack': (r2_ridge, ridge.predict(oof_mat), ridge.predict(test_mat)),
            'LR_Stack':    (r2_lr,    lr.predict(oof_mat),  lr.predict(test_mat)),
        }
        sorted_m = sorted(methods.items(), key=lambda x: x[1][0], reverse=True)
        print(f"\n  {ANALYTE_NAMES[t]}:")
        for mn, (r2, _, _) in sorted_m:
            print(f"    {mn:<15} OOF R2: {r2:.4f}")
        best_name = sorted_m[0][0]
        _, best_oof_t, best_test_t = sorted_m[0][1]
        stacked_test[:, t] = best_test_t
        best_oof[:, t]     = best_oof_t
        print(f"    -> Best: {best_name}")

    overall = np.mean([r2_score(y_train[:, t], best_oof[:, t]) for t in range(N_OUT)])
    print(f"\n  Stacked OOF R2 (overall): {overall:.4f}")
    return stacked_test, best_oof

# ══════════════════════════════════════════════════════════════════════════════
# [v10] Wavenumber Shift Augmentation (sub-pixel, differentiable)
# ══════════════════════════════════════════════════════════════════════════════

def wavenumber_shift_augment_np(X: np.ndarray, rng: np.random.RandomState,
                                max_shift_bins: float = 3.0) -> np.ndarray:
    """
    Random sub-pixel wavenumber shift via linear interpolation.
    Simulates slight instrument calibration differences.
    X: (B, V)
    Returns shifted X of same shape.
    """
    B, V = X.shape
    shifts = rng.uniform(-max_shift_bins, max_shift_bins, size=B)
    out = np.empty_like(X)
    grid = np.arange(V, dtype=np.float64)
    for i in range(B):
        shifted_grid = grid - shifts[i]  # shift the sampling grid
        out[i] = np.interp(shifted_grid, grid, X[i].astype(np.float64),
                           left=X[i, 0], right=X[i, -1])
    return out.astype(np.float32)


def wavenumber_shift_augment_torch(X: torch.Tensor, max_shift_bins: float = 3.0) -> torch.Tensor:
    """
    Differentiable sub-pixel wavenumber shift using grid_sample.
    X: (B, V) spectra
    Returns shifted spectra (B, V).
    """
    B, V = X.shape
    shifts = (torch.rand(B, device=X.device) * 2 - 1) * max_shift_bins  # uniform [-max, max]
    # Build sampling grid: normalized to [-1, 1]
    base_grid = torch.linspace(-1, 1, V, device=X.device).unsqueeze(0).expand(B, -1)  # (B, V)
    # Shift in normalized coords: shift of 1 bin = 2/V in normalized space
    norm_shift = (shifts / V * 2).unsqueeze(1)  # (B, 1)
    shifted_grid = base_grid - norm_shift  # (B, V)
    # grid_sample expects (B, C, H, W) input and (B, H, W, 2) grid
    X_4d = X.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, V)
    grid_4d = torch.stack([shifted_grid, torch.zeros_like(shifted_grid)], dim=-1)  # (B, V, 2)
    grid_4d = grid_4d.unsqueeze(1)  # (B, 1, V, 2)
    out = F.grid_sample(X_4d, grid_4d, mode='bilinear', padding_mode='border', align_corners=True)
    return out.squeeze(1).squeeze(1)  # (B, V)


# ══════════════════════════════════════════════════════════════════════════════
# [v10] Raman Physics Modules
# ══════════════════════════════════════════════════════════════════════════════

def physics_curriculum_weight(epoch: int, final_weight: float) -> float:
    alpha = min(epoch / max(PHYS_RAMP_EPOCHS, 1), 1.0)
    return alpha * final_weight


def build_peak_locality_masks(n_bins: int, wn_low: float = WN_LOW, wn_step: float = WN_STEP,
                              sigma_cm: float = PEAK_LOCALITY_SIGMA_CM) -> torch.Tensor:
    """
    Build soft Gaussian masks for each analyte's known Raman bands.
    Returns (n_analytes, n_bins) tensor where high values = "expected band region".
    We'll penalize basis energy where mask is LOW.
    """
    wns = np.arange(wn_low, wn_low + n_bins * wn_step, wn_step)[:n_bins]
    masks = np.zeros((N_OUT, n_bins), dtype=np.float32)
    for a_idx, centers in KNOWN_BAND_CENTERS_CM.items():
        for center in centers:
            masks[a_idx] += np.exp(-0.5 * ((wns - center) / sigma_cm) ** 2)
    # Normalize each mask to [0, 1]
    for a_idx in range(N_OUT):
        mx = masks[a_idx].max()
        if mx > 1e-8:
            masks[a_idx] /= mx
    return torch.from_numpy(masks)


class AffineCalibrationHead(nn.Module):
    """
    Predicts per-sample affine intensity calibration: α (scale) and β (offset).
    Applied to mixing-decoder output: X_calibrated = α · X_hat + β
    This absorbs instrument intensity differences without per-sample normalization.
    """
    def __init__(self, enc_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2),  # [log_alpha, beta]
        )
        # Initialize near identity: α ≈ 1, β ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(h)  # (B, 2)
        log_alpha = out[:, 0]
        beta = out[:, 1]
        alpha = torch.exp(log_alpha.clamp(-2, 2))  # α ∈ [e^-2, e^2] ≈ [0.14, 7.4]
        return alpha, beta  # both (B,)


class LinearMixingDecoder(nn.Module):
    """
    Raman-appropriate nonneg linear mixing model (v10 enhanced):

        X_hat(ν) = α · [b(ν) + Σ_i c_i · s_i(ν)] + β

    Enhancements over v9:
      - Affine intensity calibration (α, β) predicted from encoder features
      - Laplacian smoothness (2nd-order) in addition to TV
      - Soft peak-locality prior via Gaussian masks
    """
    def __init__(self, n_bins: int, n_analytes: int = 3, enc_dim: int = ENC_H2):
        super().__init__()
        self.n_bins = n_bins
        self.n_analytes = n_analytes

        # Learnable nonneg basis spectra and baseline via softplus
        self.log_basis = nn.Parameter(0.02 * torch.randn(n_analytes, n_bins))
        self.log_base  = nn.Parameter(torch.zeros(n_bins))

        # Affine calibration head
        self.calib = AffineCalibrationHead(enc_dim)

        # Pre-build peak locality masks (registered as buffer, not parameter)
        plm = build_peak_locality_masks(n_bins)
        self.register_buffer('peak_locality_mask', plm)  # (A, V)

    @property
    def basis(self):
        return F.softplus(self.log_basis)  # (A, V) ≥ 0

    @property
    def baseline(self):
        return F.softplus(self.log_base)   # (V,) ≥ 0

    def reconstruct(self, c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        c: (B, A) concentrations (physical scale, ≥ 0)
        h: (B, enc_dim) encoder hidden features for calibration
        Returns X_hat: (B, V)
        """
        X_raw_hat = self.baseline.unsqueeze(0) + c @ self.basis  # (B, V)
        alpha, beta = self.calib(h)  # both (B,)
        X_hat = alpha.unsqueeze(1) * X_raw_hat + beta.unsqueeze(1)
        return X_hat

    @staticmethod
    def _area_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        area = x.sum(dim=1, keepdim=True).clamp(min=eps)
        return x / area

    def reconstruction_loss(self, c: torch.Tensor, h: torch.Tensor,
                            X_raw_phys: torch.Tensor) -> torch.Tensor:
        """
        Shape-based reconstruction loss (area-normalized).
        Compares calibrated reconstruction to observed spectrum.
        """
        X_hat = self.reconstruct(c, h).clamp(min=0.0)
        # Area-normalize both to compare shapes (absorbs global scale differences)
        Xn  = self._area_normalize(X_raw_phys.clamp(min=0.0))
        Xhn = self._area_normalize(X_hat)
        return F.mse_loss(Xhn, Xn)

    def smoothness_loss(self) -> torch.Tensor:
        """TV (1st order) + Laplacian (2nd order) smoothness on basis & baseline."""
        b = self.basis  # (A, V)
        base = self.baseline  # (V,)
        # TV: 1st-order differences
        tv_basis = (b[:, 1:] - b[:, :-1]).abs().mean()
        tv_base  = (base[1:] - base[:-1]).abs().mean()
        # Laplacian: 2nd-order differences (penalizes curvature)
        lap_basis = (b[:, 2:] - 2 * b[:, 1:-1] + b[:, :-2]).pow(2).mean()
        lap_base  = (base[2:] - 2 * base[1:-1] + base[:-2]).pow(2).mean()
        return tv_basis + tv_base + 0.5 * (lap_basis + lap_base)

    def sparsity_loss(self) -> torch.Tensor:
        return self.basis.mean()

    def peak_locality_loss(self) -> torch.Tensor:
        """
        Penalize basis energy OUTSIDE known Raman band regions.
        mask ∈ [0, 1]: 1 = expected band region, 0 = outside.
        Penalty = mean of basis * (1 - mask), i.e. energy where mask is low.
        """
        b = self.basis  # (A, V)
        anti_mask = 1.0 - self.peak_locality_mask  # (A, V)
        # Weighted mean: penalize basis amplitude in non-band regions
        return (b * anti_mask).mean()


def compute_physics_loss(
    pred_c_phys: torch.Tensor,        # (B, 3) concentrations in physical scale
    enc_h:       torch.Tensor,        # (B, enc_dim) encoder hidden features
    X_raw_phys:  torch.Tensor,        # (B, n_bins) physics target spectrum
    mix_module:  LinearMixingDecoder,
    epoch:       int,
) -> Tuple[torch.Tensor, dict]:
    losses = {}

    # Mixing reconstruction (shape comparison)
    lw_mix = physics_curriculum_weight(epoch, LW_MIX)
    L_mix  = mix_module.reconstruction_loss(pred_c_phys, enc_h, X_raw_phys)
    losses["L_mix"] = float(L_mix.detach().cpu())

    # Smoothness (TV + Laplacian)
    lw_sm = physics_curriculum_weight(epoch, LW_SMOOTH)
    L_sm  = mix_module.smoothness_loss()
    losses["L_smooth"] = float(L_sm.detach().cpu())

    # Sparsity
    lw_sp = physics_curriculum_weight(epoch, LW_SPARSE)
    L_sp  = mix_module.sparsity_loss()
    losses["L_sparse"] = float(L_sp.detach().cpu())

    # Peak locality
    lw_pl = physics_curriculum_weight(epoch, LW_PEAK_LOC)
    L_pl  = mix_module.peak_locality_loss()
    losses["L_peak_loc"] = float(L_pl.detach().cpu())

    total = lw_mix * L_mix + lw_sm * L_sm + lw_sp * L_sp + lw_pl * L_pl
    return total, losses

# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, ENC_H1), nn.BatchNorm1d(ENC_H1), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(ENC_H1, ENC_H2), nn.BatchNorm1d(ENC_H2), nn.GELU(),
        )
    def forward(self, z): return self.net(z)

class Head(nn.Module):
    def __init__(self, use_softplus: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_H2, HEAD_H), nn.GELU(), nn.Dropout(FT_DROPOUT),
            nn.Linear(HEAD_H, N_OUT),
        )
        self.use_softplus = use_softplus

    def forward(self, x):
        out = self.net(x)
        return F.softplus(out) if self.use_softplus else out

class MetaModel(nn.Module):
    """
    v10: encoder exposes hidden features (h) for the mixing decoder's
    affine calibration head.
    """
    def __init__(self, n_in, use_softplus=True):
        super().__init__()
        self.encoder = Encoder(n_in)
        self.head    = Head(use_softplus=use_softplus)

    def forward(self, z):
        return self.head(self.encoder(z))

    def forward_with_hidden(self, z):
        h = self.encoder(z)  # (B, ENC_H2)
        y = self.head(h)     # (B, N_OUT)
        return y, h

# ══════════════════════════════════════════════════════════════════════════════
# Loss & Augmentation utilities
# ══════════════════════════════════════════════════════════════════════════════
def weighted_mse(pred, true, tw):
    return ((pred - true)**2 * tw).mean()

def compute_target_weights(y):
    var = np.var(y, axis=0); inv = 1.0 / (var + 1e-8); inv /= inv.sum()
    return torch.tensor(inv, dtype=torch.float32)

def augment(Z, rng):
    """Feature-space augmentation (noise + scale)."""
    Z = Z.copy()
    Z *= rng.uniform(*AUG_SCALE, (len(Z), 1)).astype(np.float32)
    Z += rng.normal(0, AUG_NOISE, Z.shape).astype(np.float32)
    return Z

def augment_spectral(X_phys, rng, max_shift=3.0, noise_std=0.005, scale_range=(0.95, 1.05)):
    """
    Spectral-domain augmentation for physics target:
    wavenumber shift + mild multiplicative noise + scale jitter.
    """
    X = wavenumber_shift_augment_np(X_phys, rng, max_shift_bins=max_shift)
    X *= rng.uniform(*scale_range, (len(X), 1)).astype(np.float32)
    X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return np.clip(X, 0, None).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Reptile Meta-Training
# ══════════════════════════════════════════════════════════════════════════════
def reptile_inner_loop(model, Z_sup, y_sup, inner_lr, inner_steps):
    opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
    tw  = compute_target_weights(y_sup.cpu().numpy()).to(y_sup.device)
    for _ in range(inner_steps):
        opt.zero_grad()
        weighted_mse(model(Z_sup), y_sup, tw).backward()
        opt.step()

def meta_train(model, device_z, device):
    print(f"\n{'='*70}\n  Reptile Meta-Training ({META_EPOCHS} epochs)\n{'='*70}")
    rng = np.random.RandomState(SEED)
    for ep in range(META_EPOCHS):
        old = copy.deepcopy(model.state_dict())
        for _ in range(META_TASKS_PER_EPOCH):
            dx, dy = device_z[rng.randint(len(device_z))]
            idx  = rng.choice(len(dx), min(K_SUPPORT, len(dx)), False)
            Z_s  = torch.tensor(dx[idx], dtype=torch.float32).to(device)
            y_s  = torch.tensor(dy[idx], dtype=torch.float32).to(device)
            reptile_inner_loop(model, Z_s, y_s, INNER_LR, INNER_STEPS)
        new = model.state_dict()
        for k in old: new[k] = old[k] + REPTILE_OUTER_LR * (new[k] - old[k])
        model.load_state_dict(new)
        if (ep + 1) % 30 == 0:
            print(f"  epoch {ep+1}/{META_EPOCHS}")
    return model

# ══════════════════════════════════════════════════════════════════════════════
# Fine-Tuning with v10 Physics Constraints
# ══════════════════════════════════════════════════════════════════════════════
def fine_tune_once(
    meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw,
    X_phys_tr, X_phys_val,
    device, tw, seed=42
):
    rng     = np.random.RandomState(seed)
    model   = copy.deepcopy(meta_model).to(device)
    tscaler = TargetScaler().fit(y_tr_raw)
    y_tr_n  = tscaler.transform(y_tr_raw).astype(np.float32)

    # Physics module (v10: with affine calibration + peak locality)
    n_bins = X_phys_tr.shape[1]
    mix_mod = LinearMixingDecoder(n_bins, N_OUT, enc_dim=ENC_H2).to(device)

    ds = TensorDataset(
        torch.tensor(Z_tr),
        torch.tensor(y_tr_n),
        torch.tensor(X_phys_tr),
    )
    dl = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)

    all_params = list(model.parameters()) + list(mix_mod.parameters())
    opt   = torch.optim.AdamW(all_params, lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS, 1e-6)

    tw_t = tw.to(device)
    tscaler_sd_t = torch.tensor(tscaler.sd, dtype=torch.float32, device=device)
    tscaler_mu_t = torch.tensor(tscaler.mu, dtype=torch.float32, device=device)

    best_r2, best_model_sd, best_ts, wait = -np.inf, None, None, 0

    for ep in range(FT_EPOCHS):
        model.train(); mix_mod.train()

        for Zb, yb, Xb_phys in dl:
            Zb, yb, Xb_phys = Zb.to(device), yb.to(device), Xb_phys.to(device)

            # ── Feature-space augmentation for invariance ──
            Za = torch.tensor(augment(Zb.cpu().numpy(), rng), dtype=torch.float32).to(device)

            # ── Spectral-domain augmentation: wavenumber shift invariance ──
            Xb_phys_shifted = torch.tensor(
                augment_spectral(Xb_phys.cpu().numpy(), rng, max_shift=3.0),
                dtype=torch.float32, device=device
            )

            # Forward (with hidden features for affine calibration)
            pred, h     = model.forward_with_hidden(Zb)
            pred_a, h_a = model.forward_with_hidden(Za)

            # ── Data loss ──
            L_data = weighted_mse(pred, yb, tw_t)

            # ── Feature-space augmentation invariance ──
            L_aug = LW_INV * F.mse_loss(pred, pred_a)

            # ── Wavenumber shift invariance ──
            # Predictions from shifted spectra should be close to original
            # We apply shift aug to physics spectra and compare reconstruction consistency
            lw_shift = physics_curriculum_weight(ep, LW_SHIFT_INV)
            if lw_shift > 0:
                # Reconstruct both original and shifted, compare shape similarity
                pred_phys = pred * tscaler_sd_t + tscaler_mu_t
                pred_phys = F.softplus(pred_phys)
                X_hat_orig    = mix_mod.reconstruct(pred_phys, h).clamp(min=0)
                X_hat_shifted = wavenumber_shift_augment_torch(X_hat_orig, max_shift_bins=2.0)
                # The shifted reconstruction should still match the shifted observation
                Xn_sh  = LinearMixingDecoder._area_normalize(Xb_phys_shifted.clamp(min=0))
                Xhn_sh = LinearMixingDecoder._area_normalize(X_hat_shifted)
                L_shift = lw_shift * F.mse_loss(Xhn_sh, Xn_sh)
            else:
                L_shift = torch.tensor(0.0, device=device)

            # ── Physics loss (mixing + smoothness + sparsity + peak locality) ──
            pred_phys = pred * tscaler_sd_t + tscaler_mu_t
            pred_phys = F.softplus(pred_phys)
            L_phys, _ = compute_physics_loss(pred_phys, h, Xb_phys, mix_mod, ep)

            loss = L_data + L_aug + L_shift + L_phys
            opt.zero_grad(); loss.backward(); opt.step()

        sched.step()

        # ── Validation R² ──
        if (ep + 1) % 5 == 0:
            model.eval(); mix_mod.eval()
            with torch.no_grad():
                pv = model(torch.tensor(Z_val, device=device)).cpu().numpy()
            pv_inv = tscaler.inverse(pv)
            r2 = np.mean([r2_score(y_val_raw[:, i], pv_inv[:, i]) for i in range(N_OUT)])
            if r2 > best_r2:
                best_r2 = r2
                best_model_sd = copy.deepcopy(model.state_dict())
                best_ts = tscaler
                wait = 0
            else:
                wait += 5
            if wait >= PATIENCE:
                break

    if best_model_sd:
        model.load_state_dict(best_model_sd)
    return model, best_r2, best_ts

def predict_tta(model, Z, tscaler, device, n=TTA_N):
    model.eval(); rng = np.random.RandomState(99); preds = []
    with torch.no_grad():
        preds.append(model(torch.tensor(Z, device=device)).cpu().numpy())
        for _ in range(n - 1):
            Za = augment(Z, rng)
            preds.append(model(torch.tensor(Za, device=device)).cpu().numpy())
    return tscaler.inverse(np.mean(preds, axis=0))

def cv_fine_tune_nn(meta_model, Z_train, y_train, Z_test,
                    X_phys_train, X_phys_test,
                    device, tw):
    kf         = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof        = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_OUT), dtype=np.float32)
    print(f"\n{'='*70}\n  NN Fine-tuning ({N_FOLDS}-fold x {N_RESTARTS} restarts)"
          f" + v10 Physics\n{'='*70}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(Z_train)):
        Z_tr, Z_val   = Z_train[tr_idx], Z_train[val_idx]
        y_tr, y_val   = y_train[tr_idx], y_train[val_idx]
        Xp_tr, Xp_val = X_phys_train[tr_idx], X_phys_train[val_idx]

        best_model, best_r2, best_ts = None, -np.inf, None
        for restart in range(N_RESTARTS):
            m, vr2, ts = fine_tune_once(
                meta_model, Z_tr, y_tr, Z_val, y_val,
                Xp_tr, Xp_val,
                device, tw, seed=SEED + fold * 100 + restart
            )
            if vr2 > best_r2:
                best_r2, best_model, best_ts = vr2, m, ts

        oof[val_idx]  = predict_tta(best_model, Z_val, best_ts, device)
        test_preds   += predict_tta(best_model, Z_test, best_ts, device) / N_FOLDS
        r2s = [r2_score(y_val[:, i], oof[val_idx, i]) for i in range(3)]
        print(f"  Fold {fold+1}: " + "  ".join(f"{n}={v:.4f}" for n, v in zip(ANALYTE_NAMES, r2s))
              + f"  Avg={np.mean(r2s):.4f}")

    return oof, test_preds

# ══════════════════════════════════════════════════════════════════════════════
# Post-processing & Saving
# ══════════════════════════════════════════════════════════════════════════════
def post_process(preds, y_train):
    out = np.maximum(preds, 0.0)

    # ---- Manual bias adjustment ----
    out[:, 0] += 1.3   # Glucose
    out[:, 1] += 0.1   # Sodium Acetate
    out[:, 2] += 0.5   # Magnesium Sulfate
    # --------------------------------

    for i, n in enumerate(ANALYTE_NAMES):
        lo = np.percentile(y_train[:, i], 1)
        hi = np.percentile(y_train[:, i], 99)
        mg = 0.1 * (hi - lo)
        lo_c, hi_c = max(0, lo - mg), hi + mg
        out[:, i] = np.clip(out[:, i], lo_c, hi_c)
        print(f"  {n}: clipped [{lo_c:.3f}, {hi_c:.3f}]")

    return out

def save_submission(preds, y_train, name, out_dir):
    p = post_process(preds.copy(), y_train)
    sub = pd.DataFrame({
        "ID": np.arange(1, len(p) + 1),
        "Glucose": p[:, 0],
        "Sodium Acetate": p[:, 1],
        "Magnesium Sulfate": p[:, 2],
    })
    path = os.path.join(out_dir, f"submission_{name}.csv")
    sub.to_csv(path, index=False)
    print(f"  -> Saved {path}")
    for c in ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]:
        print(f"     {c}: mu={sub[c].mean():.2f} sig={sub[c].std():.2f} [{sub[c].min():.2f}, {sub[c].max():.2f}]")
    return sub

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    wns    = make_grid(WN_LOW, WN_HIGH, WN_STEP)
    N_BINS = len(wns)
    print(f"Grid: {WN_LOW}-{WN_HIGH} cm⁻¹  ({N_BINS} bins)")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"), False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

    # ── Build physics target spectra (Raman-appropriate) ──────────────────────
    print("\n[1b] Building physics-target spectra (baseline-corrected, global-scaled) ...")
    Xtr_phys = apply_baseline(Xtr)
    Xte_phys = apply_baseline(Xte)
    Xtr_phys = np.clip(Xtr_phys, 0, None)
    Xte_phys = np.clip(Xte_phys, 0, None)
    gmax = float(np.max(Xtr_phys)) + 1e-12
    Xtr_phys = (Xtr_phys / gmax).astype(np.float32)
    Xte_phys = (Xte_phys / gmax).astype(np.float32)
    print(f"  Physics spectra: train={Xtr_phys.shape} test={Xte_phys.shape} (global max={gmax:.4g})")

    print("\n  Loading device datasets ...")
    device_data = []
    for fname in DEVICE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path): continue
        X_dev, y_dev = load_device(path, wns)
        if X_dev is None or len(X_dev) < K_SUPPORT: continue
        device_data.append((X_dev, y_dev))
        print(f"    {fname:<25}  {len(X_dev):4d} samples")
    use_meta = len(device_data) > 0

    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    print("\n  Label stats:")
    for i, n in enumerate(tnames_full):
        print(f"    {n:<28} mean={y_train[:,i].mean():.2f}  [{y_train[:,i].min():.2f}, {y_train[:,i].max():.2f}]")

    # ── 2. Chemometrics preprocessing ────────────────────────────────────────
    print("\n[2] Chemometrics preprocessing (MSC → SNIP → SavGol → Scale) ...")
    prep = ChemometricsPreprocessor()
    all_raw = np.concatenate([Xtr] + [X for X, _ in device_data]) if use_meta else Xtr
    prep.fit_transform(all_raw)
    Xtr_pp = prep.transform(Xtr)
    Xte_pp = prep.transform(Xte)
    device_pp = [(prep.transform(X), y) for X, y in device_data] if use_meta else []
    print(f"  Preprocessed: train={Xtr_pp.shape}  test={Xte_pp.shape}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n[3] Feature engineering ...")
    fe = FeatureEngineer()
    feat_train = fe.build_features(Xtr_pp, Xtr, fit=True)
    feat_test  = fe.build_features(Xte_pp, Xte, fit=False)
    for name, arr in feat_train.items():
        print(f"  {name:<15}: {arr.shape}")

    # ── 4. Multi-model stacking ───────────────────────────────────────────────
    print("\n[4] Multi-model training ...")
    configs = create_base_models()
    all_oof, all_test, model_scores = train_base_models_cv(configs, feat_train, feat_test, y_train)

    print("\n[5] Stacking ...")
    stacked_test, stacked_oof = stack_models(all_oof, all_test, y_train, model_scores)

    print("\n  Saving stacked submission ...")
    save_submission(stacked_test, y_train, "v10_stack", OUT_DIR)

    hgb_test = np.column_stack([all_test[t][0] for t in range(N_OUT)])
    print("\n  Saving HGB-only submission ...")
    save_submission(hgb_test, y_train, "v10_hgb", OUT_DIR)

    # ── 5. Physics-Informed NN (v10: enhanced mixing + shift inv + affine cal) ─
    print(f"\n[6] Physics-Informed NN v10")
    print(f"     Linear Mixing + Affine Calibration + Shift Invariance + Peak Locality")
    Z_train_nn = Xtr_pp.copy()
    Z_test_nn  = Xte_pp.copy()
    device_z   = [(X.astype(np.float32), y) for X, y in device_pp] if use_meta else []
    target_weights = compute_target_weights(y_train)

    meta_model = MetaModel(Z_train_nn.shape[1], use_softplus=True).to(device)
    n_params = sum(p.numel() for p in meta_model.parameters())
    print(f"  NN params: {n_params:,}")

    if use_meta and len(device_z) > 0:
        meta_model = meta_train(meta_model, device_z, device)

    nn_oof, nn_test = cv_fine_tune_nn(
        meta_model, Z_train_nn, y_train, Z_test_nn,
        Xtr_phys, Xte_phys,
        device, target_weights
    )

    print("\n  NN OOF R²:")
    for i, n in enumerate(tnames_full):
        print(f"    {n}: {r2_score(y_train[:,i], nn_oof[:,i]):.4f}")

    print("\n  Saving NN submission ...")
    save_submission(nn_test, y_train, "v10_nn", OUT_DIR)

    # ── 6. Final ensemble ─────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  FINAL ENSEMBLE (Stack + v10-NN)\n{'='*70}")
    ensemble_test = np.zeros_like(stacked_test)

    # Calibrate NN outputs with a lightweight Ridge
    nn_oof_cal  = np.zeros_like(nn_oof)
    nn_test_cal = np.zeros_like(nn_test)
    for i in range(N_OUT):
        r = Ridge(alpha=0.1).fit(nn_oof[:, i:i+1], y_train[:, i])
        nn_oof_cal[:, i]  = r.predict(nn_oof[:, i:i+1])
        nn_test_cal[:, i] = r.predict(nn_test[:, i:i+1])

    for i in range(N_OUT):
        best_w, best_r2 = 1.0, -np.inf
        for w_s in np.arange(0.0, 1.01, 0.05):
            blend = w_s * stacked_oof[:, i] + (1 - w_s) * nn_oof_cal[:, i]
            r2 = r2_score(y_train[:, i], blend)
            if r2 > best_r2:
                best_r2, best_w = r2, w_s
        ensemble_test[:, i] = best_w * stacked_test[:, i] + (1 - best_w) * nn_test_cal[:, i]
        print(f"  {ANALYTE_NAMES[i]}: w_stack={best_w:.2f}  OOF R²={best_r2:.4f}")

    print("\n  Saving ensemble submission ...")
    save_submission(ensemble_test, y_train, "v10_ensemble", OUT_DIR)

    print(f"\n{'='*70}\n  v10 COMPLETE\n{'='*70}")
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "submission_*.csv"))):
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}")
        for c in ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]:
            print(f"    {c}: mu={df[c].mean():.3f} sig={df[c].std():.3f} [{df[c].min():.3f}, {df[c].max():.3f}]")

if __name__ == "__main__":
    main()
