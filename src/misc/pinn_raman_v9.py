#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v9  —  Physics-Informed Neural Network                       ║
║                                                                              ║
║  v8 results: HGB standalone pub=0.64 priv=0.46                               ║
║              Stacked ensemble: pub~0.68 (target)                             ║
║                                                                              ║
║  v9 strategy — Physics-Informed constraints added to NN loss:                ║
║                                                                              ║
║    [NEW] PHYSICS CONSTRAINTS (embedded as soft loss terms):                  ║
║                                                                              ║
║    1. MODIFIED BEER–LAMBERT (L_BL):                                          ║
║       I(ν) = I₀(ν)·exp(−Σᵢ εᵢ(ν)·cᵢ·Leff)                                 ║
║       NN learns per-analyte extinction spectra εᵢ(ν) as parameters.         ║
║       Loss: ||log(X_raw) − log(I0) + Σᵢ εᵢ·cᵢ||²                          ║
║       Analytes: Glucose (~1125 cm⁻¹), NaAc (~930 cm⁻¹), MgSO₄ (~980 cm⁻¹) ║
║                                                                              ║
║    2. χ⁽³⁾ NONLINEAR SUSCEPTIBILITY / CARS PROPAGATION (L_chi3):            ║
║       I_CARS(ν) ∝ |χR(ν) + χNR|²                                            ║
║          = (Re(χR) + χNR)² + Im(χR)²                                        ║
║       χR(ν) = Σᵢ Aᵢ·cᵢ·Γᵢ/((ν−Ωᵢ)²+Γᵢ²) (Lorentzian, amplitude ∝ cᵢ)   ║
║       Loss: ||I_CARS_norm − X_raw_norm||²                                    ║
║                                                                              ║
║    3. KRAMERS–KRONIG CONSISTENCY (L_KK):                                     ║
║       Re(χR) and Im(χR) must satisfy KK dispersion relations.                ║
║       Re(χR(ω)) = (2/π)·P∫ ω'·Im(χR(ω'))/(ω'²−ω²) dω'                    ║
║       Enforced via FFT-based discrete Hilbert transform.                      ║
║       Loss: ||Re(χR) − Hilbert(Im(χR))||²                                   ║
║                                                                              ║
║    4. POSITIVITY / SMOOTHNESS (L_phys):                                      ║
║       - Concentrations ≥ 0 via Softplus output activation                    ║
║       - Total-variation smoothness on learned εᵢ(ν) spectra                  ║
║       - Smoothness on reconstructed CARS spectrum                             ║
║                                                                              ║
║  Physics loss curriculum: weights ramp from 0 → max over first 200 epochs   ║
║  to allow data loss to establish a good initialization first.                ║
║                                                                              ║
║  [KEPT FROM v8]                                                              ║
║    - MSC → SNIP → SavGol chemometrics preprocessing                         ║
║    - Full feature engineering (derivatives, peaks, bands, ratios)            ║
║    - Multi-model stacking (HGB × 3 configs, ExtraTrees × 2, Ridge+PCA)      ║
║    - Reptile meta-training on auxiliary device datasets                       ║
║    - 5-fold CV fine-tuning with TTA and augmentation                         ║
║    - Per-target optimal Stack + NN ensemble blending                         ║
║                                                                              ║
║  Outputs: submission_v9_hgb.csv, submission_v9_stack.csv,                    ║
║           submission_v9_nn.csv, submission_v9_ensemble.csv                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import copy, os, warnings, glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
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
OUT_DIR  = "./v9_outputs"
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

# Raman band regions (index into 300–1942 cm-1 grid = 1643 points)
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

# Known vibrational peak positions (cm-1) and analyte mapping
# Glucose: 1125 cm-1 (C-O-C stretch), NaAc: 930 cm-1, MgSO4: 980 cm-1
ANALYTE_PEAKS_WN = [1125.0, 930.0, 980.0]   # [Glucose, NaAc, MgSO4]
ANALYTE_NAMES    = ["Glucose", "NaAc", "MgSO4"]

N_FOLDS = 5
N_OUT   = 3

# NN architecture
ENC_H1, ENC_H2, HEAD_H = 256, 64, 32
DROPOUT, FT_DROPOUT = 0.15, 0.20

# Reptile
META_EPOCHS          = 120
META_TASKS_PER_EPOCH = 16
K_SUPPORT            = 40
REPTILE_OUTER_LR     = 0.3
INNER_LR, INNER_STEPS = 3e-3, 8

# Fine-tuning
FT_EPOCHS       = 500     # extended for physics curriculum
FT_LR           = 3e-4
FT_WEIGHT_DECAY = 2e-4
FT_BATCH        = 24
N_RESTARTS      = 4
PATIENCE        = 80      # extended
AUG_NOISE, AUG_SCALE = 0.015, (0.96, 1.04)
TTA_N = 4

# ── Physics loss weights ──────────────────────────────────────────────────────
# Curriculum: ramp from 0 to final weight over PHYS_RAMP_EPOCHS epochs
PHYS_RAMP_EPOCHS = 200    # epochs before physics losses reach full weight

LW_INV   = 0.03           # augmentation invariance loss (v8)
LW_BL    = 0.10           # Beer-Lambert reconstruction
LW_CHI3  = 0.08           # χ(3) CARS intensity reconstruction
LW_KK    = 0.05           # Kramers-Kronig consistency
LW_SMOOTH = 0.02          # TV smoothness on extinction spectra
LW_POS   = 0.01           # soft positivity penalty (backup, softplus handles hard)

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
        peaks, props = find_peaks(spec, height=thresh, prominence=0.5)
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
    for name, (lo, hi) in BAND_REGIONS.items():
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
# Multi-Model Training + OOF Stacking  (unchanged from v8)
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

        for fold, (tri, vali) in enumerate(kf.split(X_tr)):
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
# [NEW v9] Physics Modules
# ══════════════════════════════════════════════════════════════════════════════

class BeerLambertDecoder(nn.Module):
    """
    Modified Beer-Lambert Law:  I(ν) = I₀(ν)·exp(−Σᵢ εᵢ(ν)·cᵢ·Leff)

    Learns per-analyte extinction spectra εᵢ(ν) (positive via softplus).
    Reconstructs log spectrum from predicted concentrations.

    Loss:  ||log(X_raw + eps) − [log_I0 − Σᵢ εᵢ·cᵢ]||²
    """
    def __init__(self, n_bins: int, n_analytes: int = 3):
        super().__init__()
        self.n_bins     = n_bins
        self.n_analytes = n_analytes

        # Learnable log-extinction spectra (softplus → strictly positive)
        self.log_eps = nn.Parameter(0.1 * torch.randn(n_analytes, n_bins))
        # Learnable log-baseline I₀(ν)
        self.log_I0  = nn.Parameter(torch.zeros(n_bins))
        # Learnable effective path length scalar per analyte
        self.log_Leff = nn.Parameter(torch.zeros(n_analytes))

    @property
    def eps(self):
        """Extinction spectra εᵢ(ν) — positive"""
        return F.softplus(self.log_eps)          # (n_analytes, n_bins)

    @property
    def Leff(self):
        """Effective optical path length — positive"""
        return F.softplus(self.log_Leff)         # (n_analytes,)

    def reconstruct_log_spectrum(self, c: torch.Tensor) -> torch.Tensor:
        """
        c : (B, n_analytes)  — predicted concentrations (physical units)
        returns: (B, n_bins) — reconstructed log-spectrum
        """
        eps  = self.eps                           # (n_analytes, n_bins)
        Leff = self.Leff                          # (n_analytes,)
        # BL attenuation: Σᵢ εᵢ(ν)·cᵢ·Leff_i
        # c: (B,A), eps: (A,V) → (B,V)
        attenuation = (c * Leff.unsqueeze(0)) @ eps    # (B, n_bins)
        return self.log_I0.unsqueeze(0) - attenuation  # (B, n_bins)

    def beer_lambert_loss(self, c: torch.Tensor, X_raw: torch.Tensor) -> torch.Tensor:
        """
        X_raw : (B, n_bins) raw (unnormalized, positive) spectrum
        c     : (B, 3) predicted concentrations
        """
        X_safe      = X_raw.clamp(min=1e-6)
        log_X       = torch.log(X_safe)               # (B, n_bins)
        log_X_recon = self.reconstruct_log_spectrum(c) # (B, n_bins)
        return F.mse_loss(log_X_recon, log_X)

    def smoothness_loss(self) -> torch.Tensor:
        """Total variation regularization on extinction spectra εᵢ(ν)"""
        diff = self.eps[:, 1:] - self.eps[:, :-1]     # (n_analytes, n_bins-1)
        return diff.abs().mean()


class Chi3PhysicsModule(nn.Module):
    """
    Third-order nonlinear susceptibility χ⁽³⁾ CARS model.

    CARS intensity:
        I_CARS(ν) ∝ |χ_R(ν) + χ_NR|²
                  = (Re(χ_R) + χ_NR)² + Im(χ_R)²

    Resonant susceptibility (sum of Lorentzians at analyte frequencies):
        χ_R(ν) = Σᵢ  Aᵢ·cᵢ · Γᵢ / ((ν−Ωᵢ)² + Γᵢ²)
        Im(χ_R) = Σᵢ Aᵢ·cᵢ · [same Lorentzian shape]
        Re(χ_R) = Σᵢ Aᵢ·cᵢ · (ν−Ωᵢ)·Γᵢ / ((ν−Ωᵢ)²+Γᵢ²)   [dispersive part]

    Kramers-Kronig constraint:
        Re(χ_R(ω)) = H[Im(χ_R)](ω)   (Hilbert transform)
        We compute H via FFT and enforce agreement.
    """
    def __init__(self, wns: np.ndarray, n_analytes: int = 3):
        super().__init__()
        self.register_buffer('wns', torch.tensor(wns, dtype=torch.float32))
        self.n_bins    = len(wns)
        self.n_analytes = n_analytes

        # Peak positions (cm⁻¹): Glucose, NaAc, MgSO4
        peak_positions = torch.tensor(ANALYTE_PEAKS_WN, dtype=torch.float32)
        self.register_buffer('peak_positions', peak_positions)

        # Learnable linewidths Γᵢ (positive via softplus, init ~15 cm⁻¹)
        self.log_gamma = nn.Parameter(torch.full((n_analytes,), np.log(15.0)))

        # Learnable peak amplitude scalings Aᵢ (positive)
        self.log_amp = nn.Parameter(torch.zeros(n_analytes))

        # Non-resonant background χ_NR (real constant)
        self.chi_nr = nn.Parameter(torch.tensor(0.1))

    @property
    def gamma(self):
        return F.softplus(self.log_gamma).clamp(min=1.0)   # (n_analytes,)

    @property
    def amp(self):
        return F.softplus(self.log_amp)                    # (n_analytes,)

    def compute_chi_R(self, c: torch.Tensor):
        """
        c     : (B, n_analytes) concentrations
        returns: im_chi (B, V), re_chi (B, V)
        """
        wns  = self.wns.unsqueeze(0)          # (1, V)
        Omega = self.peak_positions            # (A,)
        Gamma = self.gamma                    # (A,)
        A     = self.amp                      # (A,)

        # dν = ν - Ω_i: (1, V) - (A, 1) → (A, V)
        dnu = wns - Omega.unsqueeze(1)        # (A, V)
        denom = dnu**2 + Gamma.unsqueeze(1)**2  # (A, V)

        # Lorentzian shapes
        im_shape = Gamma.unsqueeze(1) / denom   # Im part (A, V)
        re_shape = dnu * Gamma.unsqueeze(1) / denom  # Re part / dispersive (A, V)

        # Scale by concentration and learnable amplitude: (B,A) * (A,V) → (B,V)
        # (B,A)*(A,) → (B,A) then matmul with (A,V)
        weights = c * A.unsqueeze(0)          # (B, A)
        im_chi  = weights @ im_shape          # (B, V)
        re_chi  = weights @ re_shape          # (B, V)
        return im_chi, re_chi

    def hilbert_real_from_imag(self, im_chi: torch.Tensor) -> torch.Tensor:
        """
        Compute Re(χ) from Im(χ) via discrete Kramers-Kronig / Hilbert transform.
        KK:  Re(χ(ω)) = H[Im(χ)](ω)

        Uses FFT: for causal response, H[x](t) = IFFT(-i·sign(f)·FFT(x)(f))
        For real signals, the 'real part from imaginary' KK is implemented as:
            F_im → multiply by sign(n) → IFFT → gives -Re component
        We negate to recover Re.
        """
        n = self.n_bins
        # rfft of the imag part along spectral axis
        F_im = torch.fft.rfft(im_chi, n=n, dim=1)   # (B, n//2+1) complex

        # sign vector in frequency domain (H transform: multiply by -i·sign(f))
        freqs = torch.arange(n // 2 + 1, device=im_chi.device, dtype=torch.float32)
        sign  = torch.sign(freqs)
        sign[0] = 0.0   # DC component → 0

        # Multiply by -i·sign: (-i)·sign·F_im
        F_re = torch.complex(-sign * F_im.imag, sign * F_im.real)  # (-i)*(a+ib) = b-ia

        re_kk = torch.fft.irfft(F_re, n=n, dim=1)  # (B, V)
        return re_kk

    def cars_intensity(self, c: torch.Tensor):
        """Compute CARS intensity from concentrations."""
        im_chi, re_chi = self.compute_chi_R(c)
        I_cars = (re_chi + self.chi_nr)**2 + im_chi**2   # (B, V)
        return I_cars, im_chi, re_chi

    def kk_consistency_loss(self, c: torch.Tensor) -> torch.Tensor:
        """
        KK constraint: Re(χ_R) computed from Lorentzian formula must agree
        with Re(χ_R) computed via Hilbert transform of Im(χ_R).
        """
        im_chi, re_chi_model = self.compute_chi_R(c)
        re_chi_kk = self.hilbert_real_from_imag(im_chi)
        # Normalize before comparing (scale can differ)
        scale = re_chi_model.abs().mean(dim=1, keepdim=True) + 1e-8
        return F.mse_loss(re_chi_model / scale, re_chi_kk / scale)

    def cars_reconstruction_loss(self, c: torch.Tensor, X_raw: torch.Tensor) -> torch.Tensor:
        """
        χ(3) constraint: reconstructed I_CARS should match observed spectrum.
        Both normalized to [0,1] before comparison.
        """
        I_cars, _, _ = self.cars_intensity(c)
        # Min-max normalize per sample
        I_min = I_cars.min(dim=1, keepdim=True)[0]
        I_max = I_cars.max(dim=1, keepdim=True)[0]
        I_norm = (I_cars - I_min) / (I_max - I_min + 1e-8)

        X_min = X_raw.min(dim=1, keepdim=True)[0]
        X_max = X_raw.max(dim=1, keepdim=True)[0]
        X_norm = (X_raw - X_min) / (X_max - X_min + 1e-8)

        return F.mse_loss(I_norm, X_norm)

    def smoothness_loss(self, c: torch.Tensor) -> torch.Tensor:
        """TV regularization on reconstructed CARS spectrum."""
        I_cars, _, _ = self.cars_intensity(c)
        diff = I_cars[:, 1:] - I_cars[:, :-1]
        return diff.abs().mean()


def physics_curriculum_weight(epoch: int, final_weight: float) -> float:
    """Ramp physics loss weight from 0 to final_weight over PHYS_RAMP_EPOCHS."""
    alpha = min(epoch / max(PHYS_RAMP_EPOCHS, 1), 1.0)
    return alpha * final_weight

# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Architecture  (v9: positivity via Softplus output)
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
    """
    [v9] Output via Softplus (enforces positivity constraint #4).
    The Softplus approximation to ReLU keeps gradients alive everywhere
    and naturally restricts concentrations to ≥ 0.
    The TargetScaler will un-standardize after prediction.
    """
    def __init__(self, use_softplus: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ENC_H2, HEAD_H), nn.GELU(), nn.Dropout(FT_DROPOUT),
            nn.Linear(HEAD_H, N_OUT),
        )
        self.use_softplus = use_softplus

    def forward(self, x):
        out = self.net(x)
        if self.use_softplus:
            return F.softplus(out)    # ≥ 0 always
        return out

class MetaModel(nn.Module):
    def __init__(self, n_in, use_softplus=True):
        super().__init__()
        self.encoder = Encoder(n_in)
        self.head    = Head(use_softplus=use_softplus)
    def forward(self, z):
        return self.head(self.encoder(z))

# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════
def weighted_mse(pred, true, tw):
    return ((pred - true)**2 * tw).mean()

def compute_target_weights(y):
    var = np.var(y, axis=0); inv = 1.0 / (var + 1e-8); inv /= inv.sum()
    return torch.tensor(inv, dtype=torch.float32)

def augment(Z, rng):
    Z = Z.copy()
    Z *= rng.uniform(*AUG_SCALE, (len(Z), 1)).astype(np.float32)
    Z += rng.normal(0, AUG_NOISE, Z.shape).astype(np.float32)
    return Z

def compute_physics_loss(
    pred_c:    torch.Tensor,    # (B, 3) predicted concentrations (scaled)
    pred_c_phys: torch.Tensor, # (B, 3) concentrations in physical scale
    X_raw:     torch.Tensor,    # (B, n_bins) raw spectrum
    bl_module: BeerLambertDecoder,
    chi3_module: Chi3PhysicsModule,
    epoch:     int,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute all physics loss terms with curriculum weighting.
    Returns total physics loss and a dict of individual terms for logging.
    """
    losses = {}

    # 1. Beer-Lambert
    lw_bl = physics_curriculum_weight(epoch, LW_BL)
    L_BL  = bl_module.beer_lambert_loss(pred_c_phys, X_raw)
    losses['L_BL'] = L_BL.item()

    # 2. χ(3) CARS reconstruction
    lw_chi3 = physics_curriculum_weight(epoch, LW_CHI3)
    L_chi3  = chi3_module.cars_reconstruction_loss(pred_c_phys, X_raw)
    losses['L_chi3'] = L_chi3.item()

    # 3. Kramers-Kronig consistency
    lw_kk = physics_curriculum_weight(epoch, LW_KK)
    L_KK  = chi3_module.kk_consistency_loss(pred_c_phys)
    losses['L_KK'] = L_KK.item()

    # 4. Smoothness (BL extinction spectra + CARS spectrum)
    lw_sm = physics_curriculum_weight(epoch, LW_SMOOTH)
    L_BL_smooth   = bl_module.smoothness_loss()
    L_chi3_smooth = chi3_module.smoothness_loss(pred_c_phys)
    L_smooth = L_BL_smooth + L_chi3_smooth
    losses['L_smooth'] = L_smooth.item()

    total = (lw_bl   * L_BL   +
             lw_chi3 * L_chi3 +
             lw_kk   * L_KK   +
             lw_sm   * L_smooth)
    return total, losses

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
# Fine-Tuning with Physics Constraints
# ══════════════════════════════════════════════════════════════════════════════
def fine_tune_once(
    meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw,
    X_raw_tr, X_raw_val,           # raw spectra for physics losses
    device, tw, wns_arr, seed=42
):
    """
    Fine-tune with:
      L_total = L_data + L_aug_invariance + L_physics(BL + chi3 + KK + smooth)
    """
    rng     = np.random.RandomState(seed)
    model   = copy.deepcopy(meta_model).to(device)
    tscaler = TargetScaler().fit(y_tr_raw)

    # Normalise targets
    y_tr_n = tscaler.transform(y_tr_raw).astype(np.float32)

    # Physics modules (on device)
    n_bins = Z_tr.shape[1]
    bl_mod   = BeerLambertDecoder(n_bins, N_OUT).to(device)
    chi3_mod = Chi3PhysicsModule(wns_arr, N_OUT).to(device)

    ds = TensorDataset(
        torch.tensor(Z_tr),
        torch.tensor(y_tr_n),
        torch.tensor(X_raw_tr),
    )
    dl = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)

    # Optimise main model + physics parameters together
    all_params = list(model.parameters()) + list(bl_mod.parameters()) + list(chi3_mod.parameters())
    opt   = torch.optim.AdamW(all_params, lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS, 1e-6)

    tw_t = tw.to(device)
    best_r2, best_model_sd, best_ts, wait = -np.inf, None, None, 0

    X_raw_val_t = torch.tensor(X_raw_val).to(device)

    for ep in range(FT_EPOCHS):
        model.train(); bl_mod.train(); chi3_mod.train()

        for Zb, yb, Xb_raw in dl:
            Zb, yb, Xb_raw = Zb.to(device), yb.to(device), Xb_raw.to(device)

            # Augmented view
            Za = torch.tensor(augment(Zb.cpu().numpy(), rng), dtype=torch.float32).to(device)

            pred   = model(Zb)
            pred_a = model(Za)

            # Data loss (in normalised space)
            L_data = weighted_mse(pred, yb, tw_t)

            # Augmentation invariance
            L_aug = LW_INV * F.mse_loss(pred, pred_a)

            # Convert to physical space for physics losses
            # tscaler.inverse works on numpy; use tensor version
            pred_phys = pred.detach() * torch.tensor(tscaler.sd, dtype=torch.float32).to(device) \
                        + torch.tensor(tscaler.mu, dtype=torch.float32).to(device)
            pred_phys = F.softplus(pred_phys)   # extra positivity guard

            # Physics losses
            L_phys, _ = compute_physics_loss(
                pred, pred_phys, Xb_raw, bl_mod, chi3_mod, ep
            )

            loss = L_data + L_aug + L_phys
            opt.zero_grad(); loss.backward(); opt.step()

        sched.step()

        # Validation R²
        if (ep + 1) % 5 == 0:
            model.eval(); bl_mod.eval(); chi3_mod.eval()
            with torch.no_grad():
                pv = model(torch.tensor(Z_val).to(device)).cpu().numpy()
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
        preds.append(model(torch.tensor(Z).to(device)).cpu().numpy())
        for _ in range(n - 1):
            Za = augment(Z, rng)
            preds.append(model(torch.tensor(Za).to(device)).cpu().numpy())
    return tscaler.inverse(np.mean(preds, axis=0))

def cv_fine_tune_nn(meta_model, Z_train, y_train, Z_test,
                    X_raw_train, X_raw_test,
                    device, tw, wns_arr):
    kf        = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof       = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_OUT), dtype=np.float32)
    print(f"\n{'='*70}\n  NN Fine-tuning ({N_FOLDS}-fold x {N_RESTARTS} restarts) + Physics Constraints\n{'='*70}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(Z_train)):
        Z_tr, Z_val   = Z_train[tr_idx], Z_train[val_idx]
        y_tr, y_val   = y_train[tr_idx], y_train[val_idx]
        Xr_tr, Xr_val = X_raw_train[tr_idx], X_raw_train[val_idx]

        best_model, best_r2, best_ts = None, -np.inf, None
        for restart in range(N_RESTARTS):
            m, vr2, ts = fine_tune_once(
                meta_model, Z_tr, y_tr, Z_val, y_val,
                Xr_tr, Xr_val,
                device, tw, wns_arr, seed=SEED + fold * 100 + restart
            )
            if vr2 > best_r2:
                best_r2, best_model, best_ts = vr2, m, ts

        oof[val_idx]  = predict_tta(best_model, Z_val, best_ts, device)
        test_preds   += predict_tta(best_model, Z_test, best_ts, device) / N_FOLDS
        r2s = [r2_score(y_val[:, i], oof[val_idx, i]) for i in range(3)]
        print(f"  Fold {fold+1}: " + "  ".join(f"{n}={v:.4f}" for n, v in zip(ANALYTE_NAMES, r2s)) + f"  Avg={np.mean(r2s):.4f}")

    return oof, test_preds

# ══════════════════════════════════════════════════════════════════════════════
# Post-processing & Saving
# ══════════════════════════════════════════════════════════════════════════════
def post_process(preds, y_train):
    out = np.maximum(preds, 0.0)
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
    print(f"\nPhysics peaks: " + ", ".join(f"{n}@{p:.0f}cm⁻¹" for n, p in zip(ANALYTE_NAMES, ANALYTE_PEAKS_WN)))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"), False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

    # Keep raw spectra (positive, pre-preprocessing) for physics losses
    # Clip to positive range and normalise per-row to [0,1]
    Xtr_raw_phys = np.clip(Xtr, 0, None)
    Xtr_raw_phys = Xtr_raw_phys / (Xtr_raw_phys.max(axis=1, keepdims=True) + 1e-12)
    Xte_raw_phys = np.clip(Xte, 0, None)
    Xte_raw_phys = Xte_raw_phys / (Xte_raw_phys.max(axis=1, keepdims=True) + 1e-12)

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
    save_submission(stacked_test, y_train, "v9_stack", OUT_DIR)

    hgb_test = np.column_stack([all_test[t][0] for t in range(N_OUT)])
    print("\n  Saving HGB-only submission ...")
    save_submission(hgb_test, y_train, "v9_hgb", OUT_DIR)

    # ── 5. Physics-Informed NN ────────────────────────────────────────────────
    print("\n[6] Physics-Informed NN (Beer-Lambert + χ⁽³⁾ + KK + Positivity) ...")
    Z_train_nn = Xtr_pp.copy()
    Z_test_nn  = Xte_pp.copy()
    device_z   = [(X.astype(np.float32), y) for X, y in device_pp] if use_meta else []
    target_weights = compute_target_weights(y_train)

    meta_model = MetaModel(Z_train_nn.shape[1], use_softplus=True).to(device)
    n_params = sum(p.numel() for p in meta_model.parameters())
    n_bl     = N_OUT * N_BINS + N_BINS + N_OUT   # BeerLambert params
    n_chi3   = N_OUT + N_OUT + 1                  # Chi3 params (approx)
    print(f"  NN params: {n_params:,}")
    print(f"  BeerLambert params per fold: {n_bl:,}")
    print(f"  Chi3 params per fold: ~{n_chi3:,}")

    if use_meta and len(device_z) > 0:
        meta_model = meta_train(meta_model, device_z, device)

    nn_oof, nn_test = cv_fine_tune_nn(
        meta_model, Z_train_nn, y_train, Z_test_nn,
        Xtr_raw_phys, Xte_raw_phys,
        device, target_weights, wns
    )

    print("\n  NN OOF R²:")
    for i, n in enumerate(tnames_full):
        print(f"    {n}: {r2_score(y_train[:,i], nn_oof[:,i]):.4f}")

    print("\n  Saving NN submission ...")
    save_submission(nn_test, y_train, "v9_nn", OUT_DIR)

    # ── 6. Final ensemble ─────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  FINAL ENSEMBLE (Stack + Physics-NN)\n{'='*70}")
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
    save_submission(ensemble_test, y_train, "v9_ensemble", OUT_DIR)

    # ── 7. Optional blend with winning CSV ───────────────────────────────────
    winning_path = None
    for candidate in [WINNING_CSV,
                      os.path.join(DATA_DIR, "submission_pp_hgb_7_2_0.csv"),
                      "./submission_pp_hgb_7_2_0.csv"]:
        if candidate and os.path.exists(candidate):
            winning_path = candidate; break

    if winning_path:
        print(f"\n{'='*70}\n  BLENDING WITH WINNING CSV\n{'='*70}")
        win = pd.read_csv(winning_path)
        win_preds = win[["Glucose", "Sodium Acetate", "Magnesium Sulfate"]].values
        for alpha in [0.10, 0.15, 0.20, 0.25, 0.30]:
            blended = alpha * ensemble_test + (1 - alpha) * win_preds
            save_submission(blended, y_train, f"v9_win_{int(alpha*100)}", OUT_DIR)
    else:
        print("\n  No winning CSV found — skipping winning blend.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  v9 COMPLETE\n{'='*70}")
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "submission_*.csv"))):
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}")
        for c in ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]:
            print(f"    {c}: mu={df[c].mean():.3f} sig={df[c].std():.3f} [{df[c].min():.3f}, {df[c].max():.3f}]")


if __name__ == "__main__":
    main()
