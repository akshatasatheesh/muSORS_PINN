#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v14 — v13 Physics Features + v10 NN + Domain Calibration     ║
║                                                                              ║
║  Merges:                                                                     ║
║    FROM v13: Physics as HGB features (NMF coefficients, residuals, ratios)   ║
║             Domain-shift calibration (affine from winning CSV)               ║
║             PLS regression, expanded model diversity                         ║
║             NO manual bias offset                                            ║
║                                                                              ║
║    FROM v10: Full NN with physics loss (LinearMixingDecoder, affine calib,   ║
║             wavenumber shift invariance, peak locality, smoothness priors)    ║
║             Reptile meta-learning, CV fine-tuning, TTA                       ║
║                                                                              ║
║  NN is AUTO-GATED: included in ensemble only if OOF R² > threshold          ║
║  Calibrators applied to ALL outputs (stack, HGB, NN, ensemble)              ║
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
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.cross_decomposition import PLSRegression
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
OUT_DIR  = "./v14_outputs"
SEED     = 42

WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0
DEVICE_FILES = [
    "anton_532.csv", "anton_785.csv", "kaiser.csv", "mettler_toledo.csv",
    "metrohm.csv", "tec5.csv", "timegate.csv", "tornado.csv",
]

SAVGOL_WINDOW = 7
SAVGOL_POLY   = 2
SAVGOL_DERIV  = 0
SNIP_HALFWIN  = 20
DERIV1_WINDOW = 21
DERIV2_WINDOW = 35

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

KNOWN_BAND_CENTERS_CM = {
    0: [1125, 1065, 525],
    1: [930, 1415, 650],
    2: [980, 450, 615],
}
PEAK_LOCALITY_SIGMA_CM = 60.0

ANALYTE_NAMES = ["Glucose", "NaAc", "MgSO4"]
ANALYTE_COLS  = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]
N_FOLDS = 5
N_OUT   = 3

# ── NN architecture (from v10) ────────────────────────────────────────────────
ENC_H1, ENC_H2, HEAD_H = 256, 64, 32
DROPOUT, FT_DROPOUT = 0.15, 0.20

# Reptile
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

# Physics loss weights (from v10)
PHYS_RAMP_EPOCHS = 220
LW_INV       = 0.03
LW_SHIFT_INV = 0.02
LW_MIX       = 0.12
LW_SMOOTH    = 0.025
LW_SPARSE    = 0.004
LW_PEAK_LOC  = 0.008

# Auto-gate: exclude NN from ensemble if avg OOF R² below this
NN_AUTO_GATE_R2 = 0.05

# Winning CSV
WINNING_CSV_CANDIDATES = [
    "submission_v8_win_15.csv",
    "submission_pp_hgb_7_2_0.csv",
    os.path.join("data", "submission_pp_hgb_7_2_0.csv"),
    os.path.join("data", "submission_v8_win_15.csv"),
]

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

def find_winning_csv():
    for path in WINNING_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

def load_winning_csv(path):
    df = pd.read_csv(path)
    return df[ANALYTE_COLS].values.astype(np.float32)

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
# [v13] Physics-Informed Feature Engineering (for HGB)
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsFeatureExtractor:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.nmf = None
        self.basis = None
        self.baseline = None

    def fit(self, X_phys):
        X = np.clip(X_phys, 0, None).astype(np.float64)
        self.nmf = NMF(n_components=self.n_components, init='nndsvda',
                       max_iter=500, random_state=SEED, l1_ratio=0.1, alpha_W=0.01)
        W = self.nmf.fit_transform(X)
        self.basis = self.nmf.components_
        self.baseline = np.clip(X.mean(0) - W.mean(0) @ self.basis, 0, None)
        print(f"    NMF recon error: {self.nmf.reconstruction_err_:.4f}")
        return self

    def transform(self, X_phys):
        X = np.clip(X_phys, 0, None).astype(np.float64)
        N = X.shape[0]
        W = self.nmf.transform(X)
        X_hat = W @ self.basis + self.baseline
        residual = X - X_hat

        res_total = np.sqrt(np.mean(residual ** 2, axis=1))
        res_max   = np.max(np.abs(residual), axis=1)
        res_pos   = np.mean(np.clip(residual, 0, None), axis=1)
        res_neg   = np.mean(np.clip(residual, None, 0), axis=1)

        band_res = []
        for _, (lo, hi) in BAND_REGIONS.items():
            hi = min(hi, X.shape[1]); lo = max(lo, 0)
            band_res.append(np.mean(residual[:, lo:hi], axis=1) if hi > lo else np.zeros(N))
        band_res = np.stack(band_res, axis=1)

        W_safe = W + 1e-10
        ratios = np.stack([W_safe[:,0]/W_safe[:,1], W_safe[:,0]/W_safe[:,2], W_safe[:,1]/W_safe[:,2]], axis=1)

        cos_sims = []
        for k in range(self.n_components):
            b = self.basis[k]
            cos_sims.append((X @ b) / (np.linalg.norm(X, axis=1) * np.linalg.norm(b) + 1e-10))
        cos_sims = np.stack(cos_sims, axis=1)

        return np.hstack([W, res_total[:,None], res_max[:,None], res_pos[:,None], res_neg[:,None],
                          band_res, ratios, cos_sims]).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Standard Feature Engineering
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
        features.append([len(peaks),
            np.sum(spec[peaks]) if len(peaks) > 0 else 0.0,
            np.mean(spec[peaks]) if len(peaks) > 0 else 0.0,
            np.std(spec[peaks])  if len(peaks) > 1 else 0.0,
            np.mean(widths)      if len(widths) > 0 else 0.0,
            np.max(spec[peaks])  if len(peaks) > 0 else 0.0])
    return np.array(features, dtype=np.float32)

def compute_statistical_features(spectra):
    return np.stack([np.mean(spectra,1), np.std(spectra,1), skew(spectra,axis=1),
        kurtosis(spectra,axis=1), np.max(spectra,1), np.min(spectra,1),
        np.ptp(spectra,1), np.sum(spectra**2,axis=1)], axis=1).astype(np.float32)

def compute_band_integrals(spectra):
    integrals = []
    for _, (lo, hi) in BAND_REGIONS.items():
        hi = min(hi, spectra.shape[1]); lo = max(lo, 0)
        integrals.extend([np.sum(spectra[:,lo:hi],1), np.mean(spectra[:,lo:hi],1)])
    return np.stack(integrals, axis=1).astype(np.float32)

def compute_peak_ratios(spectra):
    n = spectra.shape[1]
    def _b(lo, hi): return np.mean(spectra[:, max(0,lo):min(hi,n)], axis=1) + 1e-10
    g, na, mg, w, fp = _b(800,850), _b(600,660), _b(650,710), _b(1300,1500), _b(400,900)
    return np.stack([g/w,na/w,mg/w,g/fp,na/fp,mg/fp,g/na,g/mg,na/mg], axis=1).astype(np.float32)

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}; self.pca = None; self.phys_scaler = None

    def build_features(self, X_pp, X_raw_mean, phys_feats=None, fit=True):
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
        if phys_feats is not None:
            if fit:
                self.phys_scaler = StandardScaler()
                pf = self.phys_scaler.fit_transform(phys_feats).astype(np.float32)
            else:
                pf = self.phys_scaler.transform(phys_feats).astype(np.float32)
            feature_sets['phys_only'] = pf
            feature_sets['spec_phys'] = np.hstack([X_pp, pf])
            feature_sets['combined_phys'] = np.hstack([X_pp, d1, d2, stats, peaks, bands, ratios, pf])

        if fit:
            self.pca = PCA(n_components=min(62, X_pp.shape[0]-1), random_state=SEED)
            feature_sets['pca'] = self.pca.fit_transform(X_pp)
        else:
            feature_sets['pca'] = self.pca.transform(X_pp)
        for name, feats in feature_sets.items():
            if name == 'phys_only': continue
            if fit:
                sc = StandardScaler(); feature_sets[name] = sc.fit_transform(feats).astype(np.float32)
                self.scalers[name] = sc
            else:
                feature_sets[name] = self.scalers[name].transform(feats).astype(np.float32)
        return feature_sets

# ══════════════════════════════════════════════════════════════════════════════
# Multi-Model Training + Stacking (v13 expanded + physics feature sets)
# ══════════════════════════════════════════════════════════════════════════════
def create_base_models(has_phys=True):
    configs = []
    configs.append({'name': 'HGB_spec', 'feature_set': 'spec_only', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.1,max_leaf_nodes=15,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
    ]})
    configs.append({'name': 'HGB_comb', 'feature_set': 'combined_all', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=10,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=10,learning_rate=0.1,max_leaf_nodes=15,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
    ]})
    if has_phys:
        configs.append({'name': 'HGB_phys', 'feature_set': 'spec_phys', 'models': [
            lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
            lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        ]})
        configs.append({'name': 'HGB_cphys', 'feature_set': 'combined_phys', 'models': [
            lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=12,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
            lambda: HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=18,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        ]})
    configs.append({'name': 'ET_comb', 'feature_set': 'combined_all', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_split=8,min_samples_leaf=5,max_features=0.45,bootstrap=True,max_samples=0.8,ccp_alpha=1e-4,random_state=SEED,n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=500,max_depth=14,min_samples_split=10,min_samples_leaf=6,max_features=0.50,bootstrap=True,max_samples=0.75,ccp_alpha=1e-4,random_state=SEED,n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400,max_depth=10,min_samples_split=8,min_samples_leaf=8,max_features=0.40,bootstrap=True,max_samples=0.8,ccp_alpha=2e-4,random_state=SEED,n_jobs=-1),
    ]})
    configs.append({'name': 'ET_d1', 'feature_set': 'spec_d1', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_split=8,min_samples_leaf=5,max_features=0.30,bootstrap=True,max_samples=0.8,random_state=SEED,n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400,max_depth=10,min_samples_split=10,min_samples_leaf=6,max_features=0.35,bootstrap=True,max_samples=0.75,random_state=SEED,n_jobs=-1),
    ]})
    configs.append({'name': 'Ridge_PCA', 'feature_set': 'pca', 'models': [
        lambda: Ridge(alpha=1.0), lambda: Ridge(alpha=5.0), lambda: Ridge(alpha=0.1),
    ]})
    configs.append({'name': 'PLS_spec', 'feature_set': 'spec_only', 'models': [
        lambda: PLSRegression(n_components=8), lambda: PLSRegression(n_components=12), lambda: PLSRegression(n_components=5),
    ]})
    if has_phys:
        configs.append({'name': 'Ridge_phys', 'feature_set': 'phys_only', 'models': [
            lambda: Ridge(alpha=1.0), lambda: Ridge(alpha=10.0), lambda: Ridge(alpha=0.1),
        ]})
    return configs

def train_base_models_cv(configs, feat_train, feat_test, y_train):
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    all_oof  = {t: [] for t in range(N_OUT)}
    all_test = {t: [] for t in range(N_OUT)}
    model_scores = []
    for cfg in configs:
        fname = cfg['feature_set']
        if fname not in feat_train:
            print(f"  {cfg['name']:<12}  SKIPPED"); continue
        X_tr, X_te = feat_train[fname], feat_test[fname]
        oof_preds  = np.zeros((len(y_train), N_OUT))
        test_preds = np.zeros((X_te.shape[0], N_OUT))
        for _, (tri, vali) in enumerate(kf.split(X_tr)):
            Xf, Xv, yf = X_tr[tri], X_tr[vali], y_train[tri]
            fold_test = np.zeros((X_te.shape[0], N_OUT))
            for t in range(N_OUT):
                fp, tp = np.zeros(len(vali)), np.zeros(X_te.shape[0])
                for mfn in cfg['models']:
                    m = mfn(); m.fit(Xf, yf[:,t])
                    fp += m.predict(Xv).ravel(); tp += m.predict(X_te).ravel()
                n = len(cfg['models'])
                oof_preds[vali,t] = fp/n; fold_test[:,t] = tp/n
            test_preds += fold_test / N_FOLDS
        target_r2s = [r2_score(y_train[:,t], oof_preds[:,t]) for t in range(N_OUT)]
        avg_r2 = np.mean(target_r2s)
        print(f"  {cfg['name']:<12}  "+"  ".join(f"{n}={v:.4f}" for n,v in zip(ANALYTE_NAMES, target_r2s))+f"  Avg={avg_r2:.4f}")
        model_scores.append({'name': cfg['name'], 'target_r2s': target_r2s, 'avg_r2': avg_r2})
        for t in range(N_OUT):
            all_oof[t].append(oof_preds[:,t]); all_test[t].append(test_preds[:,t])
    return all_oof, all_test, model_scores

def stack_models(all_oof, all_test, y_train, model_scores):
    stacked_test = np.zeros((list(all_test.values())[0][0].shape[0], N_OUT))
    best_oof = np.zeros((len(y_train), N_OUT))
    for t in range(N_OUT):
        oof_mat, test_mat = np.column_stack(all_oof[t]), np.column_stack(all_test[t])
        y_t = y_train[:,t]; n_m = oof_mat.shape[1]
        r2_avg = r2_score(y_t, np.mean(oof_mat,1))
        w = np.maximum([ms['target_r2s'][t] for ms in model_scores], 0); w = w/(w.sum()+1e-10)
        r2_wavg = r2_score(y_t, oof_mat@w)
        def obj(ww): return mean_squared_error(y_t, oof_mat@(ww/ww.sum()))
        res = minimize(obj, np.ones(n_m)/n_m, method='SLSQP', bounds=[(0,1)]*n_m,
                       constraints={'type':'eq','fun':lambda ww:ww.sum()-1})
        ow = res.x/res.x.sum(); r2_opt = r2_score(y_t, oof_mat@ow)
        ridge = Ridge(alpha=2.0).fit(oof_mat,y_t); r2_ridge = r2_score(y_t, ridge.predict(oof_mat))
        lr = LinearRegression().fit(oof_mat,y_t); r2_lr = r2_score(y_t, lr.predict(oof_mat))
        methods = {'Simple_Avg':(r2_avg,np.mean(oof_mat,1),np.mean(test_mat,1)),
            'Wt_Avg':(r2_wavg,oof_mat@w,test_mat@w), 'Opt_Wts':(r2_opt,oof_mat@ow,test_mat@ow),
            'Ridge_Stack':(r2_ridge,ridge.predict(oof_mat),ridge.predict(test_mat)),
            'LR_Stack':(r2_lr,lr.predict(oof_mat),lr.predict(test_mat))}
        sorted_m = sorted(methods.items(), key=lambda x: x[1][0], reverse=True)
        print(f"\n  {ANALYTE_NAMES[t]}:")
        for mn,(r2,_,_) in sorted_m: print(f"    {mn:<15} OOF R2: {r2:.4f}")
        _,best_oof_t,best_test_t = sorted_m[0][1]
        stacked_test[:,t]=best_test_t; best_oof[:,t]=best_oof_t
        print(f"    -> Best: {sorted_m[0][0]}")
    overall = np.mean([r2_score(y_train[:,t], best_oof[:,t]) for t in range(N_OUT)])
    print(f"\n  Stacked OOF R2 (overall): {overall:.4f}")
    return stacked_test, best_oof

# ══════════════════════════════════════════════════════════════════════════════
# [v10] NN Physics Modules (exact copy)
# ══════════════════════════════════════════════════════════════════════════════
def wavenumber_shift_augment_np(X, rng, max_shift_bins=3.0):
    B, V = X.shape; shifts = rng.uniform(-max_shift_bins, max_shift_bins, size=B)
    out = np.empty_like(X); grid = np.arange(V, dtype=np.float64)
    for i in range(B):
        out[i] = np.interp(grid - shifts[i], grid, X[i].astype(np.float64), left=X[i,0], right=X[i,-1])
    return out.astype(np.float32)

def wavenumber_shift_augment_torch(X, max_shift_bins=3.0):
    B, V = X.shape
    shifts = (torch.rand(B, device=X.device)*2-1)*max_shift_bins
    base_grid = torch.linspace(-1,1,V,device=X.device).unsqueeze(0).expand(B,-1)
    norm_shift = (shifts/V*2).unsqueeze(1)
    shifted_grid = base_grid - norm_shift
    X_4d = X.unsqueeze(1).unsqueeze(2)
    grid_4d = torch.stack([shifted_grid, torch.zeros_like(shifted_grid)], dim=-1).unsqueeze(1)
    out = F.grid_sample(X_4d, grid_4d, mode='bilinear', padding_mode='border', align_corners=True)
    return out.squeeze(1).squeeze(1)

def physics_curriculum_weight(epoch, final_weight):
    return min(epoch / max(PHYS_RAMP_EPOCHS, 1), 1.0) * final_weight

def build_peak_locality_masks_torch(n_bins, wn_low=WN_LOW, wn_step=WN_STEP, sigma_cm=PEAK_LOCALITY_SIGMA_CM):
    wns = np.arange(wn_low, wn_low + n_bins*wn_step, wn_step)[:n_bins]
    masks = np.zeros((N_OUT, n_bins), dtype=np.float32)
    for a_idx, centers in KNOWN_BAND_CENTERS_CM.items():
        for center in centers:
            masks[a_idx] += np.exp(-0.5*((wns-center)/sigma_cm)**2)
    for a_idx in range(N_OUT):
        mx = masks[a_idx].max()
        if mx > 1e-8: masks[a_idx] /= mx
    return torch.from_numpy(masks)

class AffineCalibrationHead(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(enc_dim,16), nn.GELU(), nn.Linear(16,2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, h):
        out = self.net(h)
        return torch.exp(out[:,0].clamp(-2,2)), out[:,1]

class LinearMixingDecoder(nn.Module):
    def __init__(self, n_bins, n_analytes=3, enc_dim=ENC_H2):
        super().__init__()
        self.n_bins, self.n_analytes = n_bins, n_analytes
        self.log_basis = nn.Parameter(0.02*torch.randn(n_analytes, n_bins))
        self.log_base  = nn.Parameter(torch.zeros(n_bins))
        self.calib = AffineCalibrationHead(enc_dim)
        self.register_buffer('peak_locality_mask', build_peak_locality_masks_torch(n_bins))
    @property
    def basis(self): return F.softplus(self.log_basis)
    @property
    def baseline(self): return F.softplus(self.log_base)
    def reconstruct(self, c, h):
        X_raw_hat = self.baseline.unsqueeze(0) + c @ self.basis
        alpha, beta = self.calib(h)
        return alpha.unsqueeze(1)*X_raw_hat + beta.unsqueeze(1)
    @staticmethod
    def _area_normalize(x, eps=1e-8):
        return x / x.sum(dim=1, keepdim=True).clamp(min=eps)
    def reconstruction_loss(self, c, h, X_raw_phys):
        X_hat = self.reconstruct(c, h).clamp(min=0.0)
        return F.mse_loss(self._area_normalize(X_hat), self._area_normalize(X_raw_phys.clamp(min=0.0)))
    def smoothness_loss(self):
        b, base = self.basis, self.baseline
        tv = (b[:,1:]-b[:,:-1]).abs().mean() + (base[1:]-base[:-1]).abs().mean()
        lap = (b[:,2:]-2*b[:,1:-1]+b[:,:-2]).pow(2).mean() + (base[2:]-2*base[1:-1]+base[:-2]).pow(2).mean()
        return tv + 0.5*lap
    def sparsity_loss(self): return self.basis.mean()
    def peak_locality_loss(self): return (self.basis*(1.0-self.peak_locality_mask)).mean()

def compute_physics_loss(pred_c_phys, enc_h, X_raw_phys, mix_module, epoch):
    losses = {}
    lw_mix = physics_curriculum_weight(epoch, LW_MIX)
    L_mix = mix_module.reconstruction_loss(pred_c_phys, enc_h, X_raw_phys); losses["L_mix"] = float(L_mix.detach())
    lw_sm = physics_curriculum_weight(epoch, LW_SMOOTH)
    L_sm = mix_module.smoothness_loss(); losses["L_smooth"] = float(L_sm.detach())
    lw_sp = physics_curriculum_weight(epoch, LW_SPARSE)
    L_sp = mix_module.sparsity_loss(); losses["L_sparse"] = float(L_sp.detach())
    lw_pl = physics_curriculum_weight(epoch, LW_PEAK_LOC)
    L_pl = mix_module.peak_locality_loss(); losses["L_peak_loc"] = float(L_pl.detach())
    total = lw_mix*L_mix + lw_sm*L_sm + lw_sp*L_sp + lw_pl*L_pl
    return total, losses

# ══════════════════════════════════════════════════════════════════════════════
# [v10] NN Architecture + Training
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,ENC_H1),nn.BatchNorm1d(ENC_H1),nn.GELU(),nn.Dropout(DROPOUT),
                                 nn.Linear(ENC_H1,ENC_H2),nn.BatchNorm1d(ENC_H2),nn.GELU())
    def forward(self, z): return self.net(z)

class Head(nn.Module):
    def __init__(self, use_softplus=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ENC_H2,HEAD_H),nn.GELU(),nn.Dropout(FT_DROPOUT),nn.Linear(HEAD_H,N_OUT))
        self.use_softplus = use_softplus
    def forward(self, x):
        out = self.net(x); return F.softplus(out) if self.use_softplus else out

class MetaModel(nn.Module):
    def __init__(self, n_in, use_softplus=True):
        super().__init__()
        self.encoder = Encoder(n_in); self.head = Head(use_softplus)
    def forward(self, z): return self.head(self.encoder(z))
    def forward_with_hidden(self, z):
        h = self.encoder(z); return self.head(h), h

def weighted_mse(pred, true, tw): return ((pred-true)**2*tw).mean()

def compute_target_weights(y):
    var = np.var(y,axis=0); inv = 1.0/(var+1e-8); inv /= inv.sum()
    return torch.tensor(inv, dtype=torch.float32)

def augment(Z, rng):
    Z = Z.copy(); Z *= rng.uniform(*AUG_SCALE,(len(Z),1)).astype(np.float32)
    Z += rng.normal(0,AUG_NOISE,Z.shape).astype(np.float32); return Z

def augment_spectral(X_phys, rng, max_shift=3.0, noise_std=0.005, scale_range=(0.95,1.05)):
    X = wavenumber_shift_augment_np(X_phys, rng, max_shift)
    X *= rng.uniform(*scale_range,(len(X),1)).astype(np.float32)
    X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return np.clip(X,0,None).astype(np.float32)

def reptile_inner_loop(model, Z_sup, y_sup, inner_lr, inner_steps):
    opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
    tw = compute_target_weights(y_sup.cpu().numpy()).to(y_sup.device)
    for _ in range(inner_steps):
        opt.zero_grad(); weighted_mse(model(Z_sup), y_sup, tw).backward(); opt.step()

def meta_train(model, device_z, device):
    print(f"\n{'='*70}\n  Reptile Meta-Training ({META_EPOCHS} epochs)\n{'='*70}")
    rng = np.random.RandomState(SEED)
    for ep in range(META_EPOCHS):
        old = copy.deepcopy(model.state_dict())
        for _ in range(META_TASKS_PER_EPOCH):
            dx, dy = device_z[rng.randint(len(device_z))]
            idx = rng.choice(len(dx), min(K_SUPPORT,len(dx)), False)
            Z_s = torch.tensor(dx[idx],dtype=torch.float32).to(device)
            y_s = torch.tensor(dy[idx],dtype=torch.float32).to(device)
            reptile_inner_loop(model, Z_s, y_s, INNER_LR, INNER_STEPS)
        new = model.state_dict()
        for k in old: new[k] = old[k] + REPTILE_OUTER_LR*(new[k]-old[k])
        model.load_state_dict(new)
        if (ep+1)%30==0: print(f"  epoch {ep+1}/{META_EPOCHS}")
    return model

def fine_tune_once(meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw,
                   X_phys_tr, X_phys_val, device, tw, seed=42):
    rng = np.random.RandomState(seed)
    model = copy.deepcopy(meta_model).to(device)
    tscaler = TargetScaler().fit(y_tr_raw)
    y_tr_n = tscaler.transform(y_tr_raw).astype(np.float32)
    n_bins = X_phys_tr.shape[1]
    mix_mod = LinearMixingDecoder(n_bins, N_OUT, ENC_H2).to(device)
    ds = TensorDataset(torch.tensor(Z_tr), torch.tensor(y_tr_n), torch.tensor(X_phys_tr))
    dl = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)
    all_params = list(model.parameters()) + list(mix_mod.parameters())
    opt = torch.optim.AdamW(all_params, lr=FT_LR, weight_decay=FT_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS, 1e-6)
    tw_t = tw.to(device)
    tscaler_sd_t = torch.tensor(tscaler.sd, dtype=torch.float32, device=device)
    tscaler_mu_t = torch.tensor(tscaler.mu, dtype=torch.float32, device=device)
    best_r2, best_sd, best_ts, wait = -np.inf, None, None, 0
    for ep in range(FT_EPOCHS):
        model.train(); mix_mod.train()
        for Zb,yb,Xb_phys in dl:
            Zb,yb,Xb_phys = Zb.to(device),yb.to(device),Xb_phys.to(device)
            Za = torch.tensor(augment(Zb.cpu().numpy(),rng),dtype=torch.float32).to(device)
            Xb_shifted = torch.tensor(augment_spectral(Xb_phys.cpu().numpy(),rng,3.0),dtype=torch.float32,device=device)
            pred,h = model.forward_with_hidden(Zb); pred_a,_ = model.forward_with_hidden(Za)
            L_data = weighted_mse(pred, yb, tw_t)
            L_aug = LW_INV * F.mse_loss(pred, pred_a)
            lw_shift = physics_curriculum_weight(ep, LW_SHIFT_INV)
            if lw_shift > 0:
                pp = F.softplus(pred*tscaler_sd_t+tscaler_mu_t)
                Xh = mix_mod.reconstruct(pp,h).clamp(min=0)
                Xhs = wavenumber_shift_augment_torch(Xh, 2.0)
                L_shift = lw_shift*F.mse_loss(LinearMixingDecoder._area_normalize(Xhs),
                                               LinearMixingDecoder._area_normalize(Xb_shifted.clamp(min=0)))
            else: L_shift = torch.tensor(0.0, device=device)
            pred_phys = F.softplus(pred*tscaler_sd_t+tscaler_mu_t)
            L_phys, _ = compute_physics_loss(pred_phys, h, Xb_phys, mix_mod, ep)
            loss = L_data + L_aug + L_shift + L_phys
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1)%5==0:
            model.eval(); mix_mod.eval()
            with torch.no_grad():
                pv = model(torch.tensor(Z_val,device=device)).cpu().numpy()
            r2 = np.mean([r2_score(y_val_raw[:,i], tscaler.inverse(pv)[:,i]) for i in range(N_OUT)])
            if r2>best_r2: best_r2,best_sd,best_ts,wait = r2,copy.deepcopy(model.state_dict()),tscaler,0
            else: wait += 5
            if wait >= PATIENCE: break
    if best_sd: model.load_state_dict(best_sd)
    return model, best_r2, best_ts

def predict_tta(model, Z, tscaler, device, n=TTA_N):
    model.eval(); rng = np.random.RandomState(99); preds = []
    with torch.no_grad():
        preds.append(model(torch.tensor(Z,device=device)).cpu().numpy())
        for _ in range(n-1):
            preds.append(model(torch.tensor(augment(Z,rng),device=device)).cpu().numpy())
    return tscaler.inverse(np.mean(preds, axis=0))

def cv_fine_tune_nn(meta_model, Z_train, y_train, Z_test, X_phys_train, X_phys_test, device, tw):
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y_train, dtype=np.float32)
    test_preds = np.zeros((Z_test.shape[0], N_OUT), dtype=np.float32)
    print(f"\n{'='*70}\n  NN Fine-tuning ({N_FOLDS}-fold x {N_RESTARTS} restarts) + v10 Physics\n{'='*70}")
    for fold,(tr_idx,val_idx) in enumerate(kf.split(Z_train)):
        Z_tr,Z_val = Z_train[tr_idx],Z_train[val_idx]
        y_tr,y_val = y_train[tr_idx],y_train[val_idx]
        Xp_tr,Xp_val = X_phys_train[tr_idx],X_phys_train[val_idx]
        best_model,best_r2,best_ts = None,-np.inf,None
        for restart in range(N_RESTARTS):
            m,vr2,ts = fine_tune_once(meta_model,Z_tr,y_tr,Z_val,y_val,Xp_tr,Xp_val,device,tw,seed=SEED+fold*100+restart)
            if vr2>best_r2: best_r2,best_model,best_ts = vr2,m,ts
        oof[val_idx] = predict_tta(best_model,Z_val,best_ts,device)
        test_preds += predict_tta(best_model,Z_test,best_ts,device)/N_FOLDS
        r2s = [r2_score(y_val[:,i], oof[val_idx,i]) for i in range(3)]
        print(f"  Fold {fold+1}: "+"  ".join(f"{n}={v:.4f}" for n,v in zip(ANALYTE_NAMES,r2s))+f"  Avg={np.mean(r2s):.4f}")
    return oof, test_preds

# ══════════════════════════════════════════════════════════════════════════════
# [v13] Domain-Shift Calibration (NO manual offset)
# ══════════════════════════════════════════════════════════════════════════════
class DomainShiftCalibrator:
    def __init__(self):
        self.scales = np.ones(N_OUT, dtype=np.float32)
        self.offsets = np.zeros(N_OUT, dtype=np.float32)
        self.method = "none"
    def fit_from_winning_csv(self, our_preds, win_preds):
        self.method = "winning_csv_affine"
        for i in range(N_OUT):
            A = np.column_stack([our_preds[:,i], np.ones(len(our_preds))])
            result, _, _, _ = np.linalg.lstsq(A, win_preds[:,i], rcond=None)
            self.scales[i], self.offsets[i] = result[0], result[1]
    def transform(self, preds):
        out = preds.copy()
        for i in range(N_OUT): out[:,i] = self.scales[i]*preds[:,i] + self.offsets[i]
        return out
    def report(self):
        print(f"    Method: {self.method}")
        for i,n in enumerate(ANALYTE_NAMES):
            print(f"    {n}: scale={self.scales[i]:.4f}  offset={self.offsets[i]:.4f}")

def grid_search_bias(our_test, win_test):
    cal = DomainShiftCalibrator(); cal.method = "grid_search_vs_winning"
    for i in range(N_OUT):
        best_mse, best_s, best_o = np.inf, 1.0, 0.0
        for s in np.arange(0.8, 1.25, 0.02):
            for o in np.arange(-2.0, 3.0, 0.05):
                mse = np.mean((s*our_test[:,i]+o - win_test[:,i])**2)
                if mse < best_mse: best_mse, best_s, best_o = mse, s, o
        cal.scales[i], cal.offsets[i] = best_s, best_o
    return cal

# ══════════════════════════════════════════════════════════════════════════════
# Post-processing & Saving (NO manual offset)
# ══════════════════════════════════════════════════════════════════════════════
def post_process(preds, y_train):
    out = np.maximum(preds, 0.0)
    for i, n in enumerate(ANALYTE_NAMES):
        lo = np.percentile(y_train[:,i], 1); hi = np.percentile(y_train[:,i], 99)
        mg = 0.15*(hi-lo); lo_c, hi_c = max(0, lo-mg), hi+mg
        out[:,i] = np.clip(out[:,i], lo_c, hi_c)
        print(f"  {n}: clipped [{lo_c:.3f}, {hi_c:.3f}]")
    return out

def save_submission(preds, y_train, name, out_dir):
    p = post_process(preds.copy(), y_train)
    sub = pd.DataFrame({"ID":np.arange(1,len(p)+1), "Glucose":p[:,0],
                         "Sodium Acetate":p[:,1], "Magnesium Sulfate":p[:,2]})
    path = os.path.join(out_dir, f"submission_{name}.csv"); sub.to_csv(path, index=False)
    print(f"  -> Saved {path}")
    for c in ANALYTE_COLS:
        print(f"     {c}: mu={sub[c].mean():.2f} sig={sub[c].std():.2f} [{sub[c].min():.2f}, {sub[c].max():.2f}]")
    return sub

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED); os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"v14 — v13 Physics Features + v10 NN + Domain Calibration\nDevice: {device}")
    wns = make_grid(WN_LOW, WN_HIGH, WN_STEP); N_BINS = len(wns)
    print(f"Grid: {WN_LOW}-{WN_HIGH} cm⁻¹  ({N_BINS} bins)")

    # ── 0. Winning CSV
    win_path = find_winning_csv(); win_test_preds = None
    if win_path:
        print(f"\n[0] Winning CSV: {win_path}")
        win_test_preds = load_winning_csv(win_path)
    else: print("\n[0] No winning CSV found.")

    # ── 1. Load data
    print("\n[1] Loading data ...")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _ = load_plate(os.path.join(DATA_DIR, "96_samples.csv"), False)
    Xtr = plate_to_grid(Xtr_raw, wns); Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

    # ── 1b. Physics spectra + NMF features
    print("\n[1b] Physics features (NMF) ...")
    Xtr_phys = np.clip(apply_baseline(Xtr), 0, None)
    Xte_phys = np.clip(apply_baseline(Xte), 0, None)
    gmax = float(np.max(Xtr_phys)) + 1e-12
    Xtr_phys = (Xtr_phys/gmax).astype(np.float32); Xte_phys = (Xte_phys/gmax).astype(np.float32)
    phys_fe = PhysicsFeatureExtractor(N_OUT); phys_fe.fit(Xtr_phys)
    phys_train = phys_fe.transform(Xtr_phys); phys_test = phys_fe.transform(Xte_phys)
    print(f"  Physics features: {phys_train.shape}")

    # ── Load devices
    device_data = []
    for fname in DEVICE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path): continue
        X_dev, y_dev = load_device(path, wns)
        if X_dev is None or len(X_dev) < K_SUPPORT: continue
        device_data.append((X_dev, y_dev)); print(f"    {fname:<25} {len(X_dev):4d} samples")
    use_meta = len(device_data) > 0

    # ── 2. Preprocessing
    print("\n[2] Chemometrics preprocessing ...")
    prep = ChemometricsPreprocessor()
    all_raw = np.concatenate([Xtr]+[X for X,_ in device_data]) if use_meta else Xtr
    prep.fit_transform(all_raw)
    Xtr_pp = prep.transform(Xtr); Xte_pp = prep.transform(Xte)
    device_pp = [(prep.transform(X),y) for X,y in device_data] if use_meta else []

    # ── 3. Features (with physics)
    print("\n[3] Feature engineering ...")
    fe = FeatureEngineer()
    feat_train = fe.build_features(Xtr_pp, Xtr, phys_feats=phys_train, fit=True)
    feat_test  = fe.build_features(Xte_pp, Xte, phys_feats=phys_test, fit=False)
    for name, arr in feat_train.items(): print(f"  {name:<15}: {arr.shape}")

    # ── 4. Base models
    print("\n[4] Multi-model training ...")
    configs = create_base_models(has_phys=True)
    all_oof, all_test, model_scores = train_base_models_cv(configs, feat_train, feat_test, y_train)

    # ── 5. Stacking
    print("\n[5] Stacking ...")
    stacked_test, stacked_oof = stack_models(all_oof, all_test, y_train, model_scores)
    hgb_test = np.column_stack([all_test[t][0] for t in range(N_OUT)])

    # ── 6. NN (v10 physics)
    print(f"\n[6] Physics-Informed NN (v10)")
    Z_train_nn, Z_test_nn = Xtr_pp.copy(), Xte_pp.copy()
    device_z = [(X.astype(np.float32),y) for X,y in device_pp] if use_meta else []
    tw = compute_target_weights(y_train)
    meta_model = MetaModel(Z_train_nn.shape[1], use_softplus=True).to(device)
    print(f"  NN params: {sum(p.numel() for p in meta_model.parameters()):,}")
    if use_meta and len(device_z) > 0:
        meta_model = meta_train(meta_model, device_z, device)
    nn_oof, nn_test = cv_fine_tune_nn(meta_model, Z_train_nn, y_train, Z_test_nn, Xtr_phys, Xte_phys, device, tw)
    nn_r2s = [r2_score(y_train[:,i], nn_oof[:,i]) for i in range(N_OUT)]
    nn_avg = np.mean(nn_r2s)
    print(f"\n  NN OOF R²: " + "  ".join(f"{n}={v:.4f}" for n,v in zip(ANALYTE_NAMES, nn_r2s)) + f"  Avg={nn_avg:.4f}")

    # ── 7. Auto-gated ensemble
    print(f"\n{'='*70}\n  ENSEMBLE (NN auto-gate @ R²>{NN_AUTO_GATE_R2})\n{'='*70}")
    if nn_avg > NN_AUTO_GATE_R2:
        print(f"  NN INCLUDED (OOF R²={nn_avg:.4f})")
        nn_oof_cal, nn_test_cal = np.zeros_like(nn_oof), np.zeros_like(nn_test)
        for i in range(N_OUT):
            r = Ridge(alpha=0.1).fit(nn_oof[:,i:i+1], y_train[:,i])
            nn_oof_cal[:,i] = r.predict(nn_oof[:,i:i+1]); nn_test_cal[:,i] = r.predict(nn_test[:,i:i+1])
        ensemble_test = np.zeros_like(stacked_test)
        for i in range(N_OUT):
            best_w, best_r2 = 1.0, -np.inf
            for w_s in np.arange(0.0, 1.01, 0.05):
                r2 = r2_score(y_train[:,i], w_s*stacked_oof[:,i]+(1-w_s)*nn_oof_cal[:,i])
                if r2>best_r2: best_r2,best_w = r2,w_s
            ensemble_test[:,i] = best_w*stacked_test[:,i] + (1-best_w)*nn_test_cal[:,i]
            print(f"  {ANALYTE_NAMES[i]}: w_stack={best_w:.2f}  OOF R²={best_r2:.4f}")
    else:
        print(f"  NN EXCLUDED (OOF R²={nn_avg:.4f} < {NN_AUTO_GATE_R2}) → stack only")
        ensemble_test = stacked_test.copy()

    # ── 8. Save raw outputs
    print(f"\n{'='*70}\n  SUBMISSIONS\n{'='*70}")
    save_submission(stacked_test, y_train, "v14_stack_raw", OUT_DIR)
    save_submission(hgb_test, y_train, "v14_hgb_raw", OUT_DIR)
    save_submission(nn_test, y_train, "v14_nn_raw", OUT_DIR)
    save_submission(ensemble_test, y_train, "v14_ensemble_raw", OUT_DIR)

    # ── 9. Domain calibration + blends
    if win_test_preds is not None and win_test_preds.shape[0] == stacked_test.shape[0]:
        print(f"\n  --- Domain-shift calibration from winning CSV ---")
        for src_name, src in [('stack',stacked_test),('hgb',hgb_test),('ensemble',ensemble_test),('nn',nn_test)]:
            cal = DomainShiftCalibrator(); cal.fit_from_winning_csv(src, win_test_preds)
            print(f"\n  Calibrator ({src_name} → winning CSV):"); cal.report()
            save_submission(cal.transform(src), y_train, f"v14_{src_name}_affine", OUT_DIR)

        cal_grid = grid_search_bias(stacked_test, win_test_preds)
        print(f"\n  Grid-search calibrator (stack):"); cal_grid.report()
        save_submission(cal_grid.transform(stacked_test), y_train, "v14_stack_grid", OUT_DIR)

        print(f"\n  --- Blends with winning CSV ---")
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for sn, sp in [('ensemble',ensemble_test),('stack',stacked_test)]:
                blended = alpha*sp + (1-alpha)*win_test_preds
                save_submission(blended, y_train, f"v14_{sn}_win_{int(alpha*100):02d}", OUT_DIR)

    # ── Summary
    print(f"\n{'='*70}\n  v14 COMPLETE\n{'='*70}")
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "submission_*.csv"))):
        df = pd.read_csv(f); print(f"  {os.path.basename(f)}")
        for c in ANALYTE_COLS:
            print(f"    {c}: mu={df[c].mean():.3f} sig={df[c].std():.3f} [{df[c].min():.3f}, {df[c].max():.3f}]")

if __name__ == "__main__":
    main()
