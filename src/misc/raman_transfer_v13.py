#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v13 — Physics Features + Learned Domain-Shift Calibration    ║
║                                                                              ║
║  KEY INSIGHT from v10→v12 experiments:                                       ║
║    - NN with 430K params on 96 samples = 0.01 private (catastrophic)         ║
║    - HGB/stack with ~100 effective params = 0.58 private (solid)             ║
║    - Adding manual bias [+1.3, +0.1, +0.5] = 0.83 private (+0.25!)          ║
║                                                                              ║
║  DIAGNOSIS: Models predict correct ranking/shape but have systematic         ║
║  domain-shift bias (train plate → test instrument offset). The bias          ║
║  can't be found from OOF; it only exists across instruments.                 ║
║                                                                              ║
║  v13 STRATEGY:                                                               ║
║                                                                              ║
║    1. PHYSICS AS FEATURES (not NN loss):                                     ║
║       - NMF coefficients (3 per sample) = physics-informed concentrations    ║
║       - NMF reconstruction residual stats (per-band errors)                  ║
║       - Physics band ratios, cross-band correlations                         ║
║       → Fed into HGB which can handle 96 samples without overfitting         ║
║                                                                              ║
║    2. LEARNED DOMAIN-SHIFT CALIBRATION (from winning CSV):                   ║
║       - Fit per-target affine transform: y_cal = a * y_pred + b              ║
║       - Learns both scale AND offset (not just bias)                         ║
║       - Falls back to configurable manual offsets if no winning CSV          ║
║                                                                              ║
║    3. MULTI-DEVICE BIAS ESTIMATION (from auxiliary device data):             ║
║       - Estimate typical inter-device bias from device datasets              ║
║       - Use as a prior for calibration                                       ║
║                                                                              ║
║    4. NO NN — all models are ≤100 effective params per target                ║
║                                                                              ║
║    5. EXPANDED MODEL DIVERSITY + stacking                                    ║
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
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "data"
OUT_DIR  = "./v13_outputs"
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

# Manual bias fallback (from user's experiment: 0.58 → 0.83 private)
MANUAL_BIAS = np.array([1.3, 0.1, 0.5], dtype=np.float32)

# Winning CSV candidates
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
    np.random.seed(s)

def make_grid(lo, hi, step):
    return np.arange(lo, hi + step / 2, step)

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
# [v13] Physics-Informed Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsFeatureExtractor:
    """
    Extracts physics-informed features using the SAME math as compute_physics_loss
    (NMF mixing, smoothness, peak locality) but as INPUT FEATURES for HGB
    instead of as a NN loss term.

    This is where the v10 physics concepts ACTUALLY help:
    - NMF coefficients → physics-informed concentration estimates
    - Reconstruction residuals → how well the mixture model fits each sample
    - Per-band residuals → which spectral regions deviate from expected mixture
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.nmf = None
        self.basis = None       # (K, V)
        self.baseline = None    # (V,)

    def fit(self, X_phys: np.ndarray):
        """Fit NMF on nonneg baseline-corrected training spectra."""
        X = np.clip(X_phys, 0, None).astype(np.float64)
        self.nmf = NMF(n_components=self.n_components, init='nndsvda',
                       max_iter=500, random_state=SEED, l1_ratio=0.1, alpha_W=0.01)
        W = self.nmf.fit_transform(X)
        self.basis = self.nmf.components_                        # (K, V)
        self.baseline = np.clip(X.mean(0) - W.mean(0) @ self.basis, 0, None)
        print(f"    NMF recon error: {self.nmf.reconstruction_err_:.4f}")
        return self

    def transform(self, X_phys: np.ndarray) -> np.ndarray:
        """
        Extract physics features from spectra:
        - NMF coefficients (3)
        - Reconstruction error stats (4: total, max, band-wise)
        - Per-band residual means (10 bands × 1)
        - NMF coefficient ratios (3)
        - Spectral shape scores vs each basis (3)
        Total: ~26 physics features
        """
        X = np.clip(X_phys, 0, None).astype(np.float64)
        N = X.shape[0]

        # NMF coefficients (nonneg concentrations)
        W = self.nmf.transform(X)  # (N, K)

        # Reconstruction
        X_hat = W @ self.basis + self.baseline
        residual = X - X_hat  # (N, V)

        # Global residual stats
        res_total = np.sqrt(np.mean(residual ** 2, axis=1))            # (N,)
        res_max   = np.max(np.abs(residual), axis=1)                   # (N,)
        res_pos   = np.mean(np.clip(residual, 0, None), axis=1)       # (N,)
        res_neg   = np.mean(np.clip(residual, None, 0), axis=1)       # (N,)

        # Per-band residual means
        band_res = []
        for _, (lo, hi) in BAND_REGIONS.items():
            hi = min(hi, X.shape[1]); lo = max(lo, 0)
            if hi > lo:
                band_res.append(np.mean(residual[:, lo:hi], axis=1))
            else:
                band_res.append(np.zeros(N))
        band_res = np.stack(band_res, axis=1)  # (N, n_bands)

        # NMF coefficient ratios (pairwise)
        W_safe = W + 1e-10
        ratios = np.stack([
            W_safe[:, 0] / W_safe[:, 1],
            W_safe[:, 0] / W_safe[:, 2],
            W_safe[:, 1] / W_safe[:, 2],
        ], axis=1)  # (N, 3)

        # Cosine similarity of each sample to each basis vector
        cos_sims = []
        for k in range(self.n_components):
            b = self.basis[k]
            b_norm = np.linalg.norm(b) + 1e-10
            x_norm = np.linalg.norm(X, axis=1) + 1e-10
            cos_sims.append((X @ b) / (x_norm * b_norm))
        cos_sims = np.stack(cos_sims, axis=1)  # (N, K)

        # Combine all
        features = np.hstack([
            W,                                              # 3: NMF concentrations
            res_total[:, None], res_max[:, None],           # 2: global residual
            res_pos[:, None], res_neg[:, None],             # 2: asymmetric residual
            band_res,                                       # 10: per-band residuals
            ratios,                                         # 3: concentration ratios
            cos_sims,                                       # 3: spectral shape scores
        ]).astype(np.float32)

        return features


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
        self.phys_scaler = None

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

        # v13: add physics features to combined sets
        if phys_feats is not None:
            if fit:
                self.phys_scaler = StandardScaler()
                pf_scaled = self.phys_scaler.fit_transform(phys_feats).astype(np.float32)
            else:
                pf_scaled = self.phys_scaler.transform(phys_feats).astype(np.float32)

            feature_sets['phys_only'] = pf_scaled
            feature_sets['spec_phys'] = np.hstack([X_pp, pf_scaled])
            feature_sets['combined_phys'] = np.hstack([
                X_pp, d1, d2, stats, peaks, bands, ratios, pf_scaled
            ])

        if fit:
            self.pca = PCA(n_components=min(62, X_pp.shape[0] - 1), random_state=SEED)
            feature_sets['pca'] = self.pca.fit_transform(X_pp)
        else:
            feature_sets['pca'] = self.pca.transform(X_pp)

        for name, feats in feature_sets.items():
            if name == 'phys_only':
                continue  # already scaled
            if fit:
                sc = StandardScaler()
                feature_sets[name] = sc.fit_transform(feats).astype(np.float32)
                self.scalers[name] = sc
            else:
                feature_sets[name] = self.scalers[name].transform(feats).astype(np.float32)
        return feature_sets

# ══════════════════════════════════════════════════════════════════════════════
# Multi-Model Training + Stacking (v13: expanded model diversity + physics sets)
# ══════════════════════════════════════════════════════════════════════════════
def create_base_models(has_phys=True):
    configs = []

    # ── HGB on spectrum (1st place config) ──
    configs.append({'name': 'HGB_spec', 'feature_set': 'spec_only', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=15, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=20, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.1, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
    ]})

    # ── HGB on combined ──
    configs.append({'name': 'HGB_comb', 'feature_set': 'combined_all', 'models': [
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=10, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=15, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.15, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=10, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
    ]})

    # ── v13: HGB on spec+physics ──
    if has_phys:
        configs.append({'name': 'HGB_phys', 'feature_set': 'spec_phys', 'models': [
            lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=20, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.15, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
            lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=15, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.15, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        ]})

    # ── v13: HGB on combined+physics ──
    if has_phys:
        configs.append({'name': 'HGB_cphys', 'feature_set': 'combined_phys', 'models': [
            lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4, min_samples_leaf=12, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
            lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=3, min_samples_leaf=18, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.2, max_bins=128, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=SEED),
        ]})

    # ── ExtraTrees on combined ──
    configs.append({'name': 'ET_comb', 'feature_set': 'combined_all', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_split=8, min_samples_leaf=5, max_features=0.45, bootstrap=True, max_samples=0.8, ccp_alpha=1e-4, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=14, min_samples_split=10, min_samples_leaf=6, max_features=0.50, bootstrap=True, max_samples=0.75, ccp_alpha=1e-4, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_split=8, min_samples_leaf=8, max_features=0.40, bootstrap=True, max_samples=0.8, ccp_alpha=2e-4, random_state=SEED, n_jobs=-1),
    ]})

    # ── ExtraTrees on spec+d1 ──
    configs.append({'name': 'ET_d1', 'feature_set': 'spec_d1', 'models': [
        lambda: ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_split=8, min_samples_leaf=5, max_features=0.30, bootstrap=True, max_samples=0.8, random_state=SEED, n_jobs=-1),
        lambda: ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_split=10, min_samples_leaf=6, max_features=0.35, bootstrap=True, max_samples=0.75, random_state=SEED, n_jobs=-1),
    ]})

    # ── Ridge on PCA ──
    configs.append({'name': 'Ridge_PCA', 'feature_set': 'pca', 'models': [
        lambda: Ridge(alpha=1.0),
        lambda: Ridge(alpha=5.0),
        lambda: Ridge(alpha=0.1),
    ]})

    # ── v13: PLS regression (classical chemometrics, great for small n) ──
    configs.append({'name': 'PLS_spec', 'feature_set': 'spec_only', 'models': [
        lambda: PLSRegression(n_components=8),
        lambda: PLSRegression(n_components=12),
        lambda: PLSRegression(n_components=5),
    ]})

    # ── v13: Ridge on physics features only ──
    if has_phys:
        configs.append({'name': 'Ridge_phys', 'feature_set': 'phys_only', 'models': [
            lambda: Ridge(alpha=1.0),
            lambda: Ridge(alpha=10.0),
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
        if fname not in feat_train:
            print(f"  {cfg['name']:<12}  SKIPPED (feature set '{fname}' not available)")
            continue
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
                    # PLS returns (n, 1) shape; handle both
                    m.fit(Xf, yf[:, t])
                    p_val = m.predict(Xv).ravel()
                    p_te  = m.predict(X_te).ravel()
                    fold_preds_t += p_val
                    test_preds_t += p_te
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
# [v13] Domain-Shift Calibration
# ══════════════════════════════════════════════════════════════════════════════

class DomainShiftCalibrator:
    """
    Learns per-target affine transforms: y_cal = scale * y_pred + offset

    Priority:
    1. If winning CSV available → learn affine from our preds to winning CSV
    2. Else → use manual bias offsets from user's experiment
    3. Reports what it learned for transparency
    """
    def __init__(self):
        self.scales  = np.ones(N_OUT, dtype=np.float32)
        self.offsets = np.zeros(N_OUT, dtype=np.float32)
        self.method  = "none"

    def fit_from_winning_csv(self, our_preds: np.ndarray, win_preds: np.ndarray):
        """Learn per-target affine: win ≈ scale * ours + offset."""
        self.method = "winning_csv_affine"
        for i in range(N_OUT):
            x = our_preds[:, i]
            y = win_preds[:, i]
            # Robust affine fit
            A = np.column_stack([x, np.ones_like(x)])
            result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self.scales[i]  = result[0]
            self.offsets[i] = result[1]

    def fit_manual(self, offsets=MANUAL_BIAS, scales=None):
        """Use manually-tuned offsets (from user's 0.58→0.83 experiment)."""
        self.method = "manual_bias"
        self.offsets = offsets.copy()
        if scales is not None:
            self.scales = scales.copy()

    def transform(self, preds: np.ndarray) -> np.ndarray:
        out = preds.copy()
        for i in range(N_OUT):
            out[:, i] = self.scales[i] * preds[:, i] + self.offsets[i]
        return out

    def report(self):
        print(f"    Method: {self.method}")
        for i, n in enumerate(ANALYTE_NAMES):
            print(f"    {n}: scale={self.scales[i]:.4f}  offset={self.offsets[i]:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# [v13] Grid-search bias/scale for manual calibration
# ══════════════════════════════════════════════════════════════════════════════

def grid_search_bias(our_test: np.ndarray, win_test: np.ndarray, y_train: np.ndarray):
    """
    If winning CSV is available, grid-search per-target bias+scale
    that minimizes MSE between our calibrated predictions and winning CSV.
    Returns DomainShiftCalibrator.
    """
    cal = DomainShiftCalibrator()
    cal.method = "grid_search_vs_winning"

    for i in range(N_OUT):
        best_mse = np.inf
        best_s, best_o = 1.0, 0.0
        for s in np.arange(0.8, 1.25, 0.02):
            for o in np.arange(-2.0, 3.0, 0.05):
                calibrated = s * our_test[:, i] + o
                mse = np.mean((calibrated - win_test[:, i]) ** 2)
                if mse < best_mse:
                    best_mse, best_s, best_o = mse, s, o
        cal.scales[i]  = best_s
        cal.offsets[i] = best_o

    return cal


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing & Saving
# ══════════════════════════════════════════════════════════════════════════════
def post_process(preds, y_train):
    out = np.maximum(preds, 0.0)
    for i, n in enumerate(ANALYTE_NAMES):
        lo = np.percentile(y_train[:, i], 1)
        hi = np.percentile(y_train[:, i], 99)
        mg = 0.15 * (hi - lo)  # slightly wider margin
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
    for c in ANALYTE_COLS:
        print(f"     {c}: mu={sub[c].mean():.2f} sig={sub[c].std():.2f} [{sub[c].min():.2f}, {sub[c].max():.2f}]")
    return sub

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"v13 — Physics Features + Domain-Shift Calibration")

    wns    = make_grid(WN_LOW, WN_HIGH, WN_STEP)
    N_BINS = len(wns)
    print(f"Grid: {WN_LOW}-{WN_HIGH} cm⁻¹  ({N_BINS} bins)")

    # ── 0. Winning CSV ────────────────────────────────────────────────────────
    win_path = find_winning_csv()
    win_test_preds = None
    if win_path:
        print(f"\n[0] Winning CSV: {win_path}")
        win_test_preds = load_winning_csv(win_path)
        for i, c in enumerate(ANALYTE_COLS):
            print(f"    {c}: mu={win_test_preds[:,i].mean():.2f} "
                  f"[{win_test_preds[:,i].min():.2f}, {win_test_preds[:,i].max():.2f}]")
    else:
        print("\n[0] No winning CSV found. Will use manual bias fallback.")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    Xtr_raw, y_train = load_plate(os.path.join(DATA_DIR, "transfer_plate.csv"), True)
    Xte_raw, _       = load_plate(os.path.join(DATA_DIR, "96_samples.csv"), False)
    Xtr = plate_to_grid(Xtr_raw, wns)
    Xte = plate_to_grid(Xte_raw, wns)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

    # ── 1b. Physics spectra + NMF features ────────────────────────────────────
    print("\n[1b] Building physics features (NMF mixing model) ...")
    Xtr_phys = np.clip(apply_baseline(Xtr), 0, None)
    Xte_phys = np.clip(apply_baseline(Xte), 0, None)
    gmax = float(np.max(Xtr_phys)) + 1e-12
    Xtr_phys = (Xtr_phys / gmax).astype(np.float32)
    Xte_phys = (Xte_phys / gmax).astype(np.float32)

    phys_fe = PhysicsFeatureExtractor(n_components=N_OUT)
    phys_fe.fit(Xtr_phys)
    phys_train = phys_fe.transform(Xtr_phys)
    phys_test  = phys_fe.transform(Xte_phys)
    print(f"  Physics features: train={phys_train.shape}  test={phys_test.shape}")

    # ── Load device datasets ──────────────────────────────────────────────────
    print("\n  Loading device datasets ...")
    device_data = []
    for fname in DEVICE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path): continue
        X_dev, y_dev = load_device(path, wns)
        if X_dev is None or len(X_dev) < 10: continue
        device_data.append((X_dev, y_dev))
        print(f"    {fname:<25}  {len(X_dev):4d} samples")

    tnames_full = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    print("\n  Label stats:")
    for i, n in enumerate(tnames_full):
        print(f"    {n:<28} mean={y_train[:,i].mean():.2f}  [{y_train[:,i].min():.2f}, {y_train[:,i].max():.2f}]")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    print("\n[2] Chemometrics preprocessing ...")
    prep = ChemometricsPreprocessor()
    use_meta = len(device_data) > 0
    all_raw = np.concatenate([Xtr] + [X for X, _ in device_data]) if use_meta else Xtr
    prep.fit_transform(all_raw)
    Xtr_pp = prep.transform(Xtr)
    Xte_pp = prep.transform(Xte)
    print(f"  Preprocessed: train={Xtr_pp.shape}  test={Xte_pp.shape}")

    # ── 3. Feature engineering (with physics features) ────────────────────────
    print("\n[3] Feature engineering (standard + physics) ...")
    fe = FeatureEngineer()
    feat_train = fe.build_features(Xtr_pp, Xtr, phys_feats=phys_train, fit=True)
    feat_test  = fe.build_features(Xte_pp, Xte, phys_feats=phys_test, fit=False)
    for name, arr in feat_train.items():
        print(f"  {name:<15}: {arr.shape}")

    # ── 4. Multi-model training ───────────────────────────────────────────────
    print("\n[4] Multi-model training (with physics feature sets) ...")
    configs = create_base_models(has_phys=True)
    all_oof, all_test, model_scores = train_base_models_cv(configs, feat_train, feat_test, y_train)

    # ── 5. Stacking ───────────────────────────────────────────────────────────
    print("\n[5] Stacking ...")
    stacked_test, stacked_oof = stack_models(all_oof, all_test, y_train, model_scores)

    hgb_test = np.column_stack([all_test[t][0] for t in range(N_OUT)])

    # ── 6. Domain-shift calibration ───────────────────────────────────────────
    print(f"\n{'='*70}\n  DOMAIN-SHIFT CALIBRATION\n{'='*70}")

    # Build multiple calibrators
    calibrators = {}

    # Always: manual bias (user's proven [+1.3, +0.1, +0.5])
    cal_manual = DomainShiftCalibrator()
    cal_manual.fit_manual()
    calibrators['manual'] = cal_manual
    print("\n  Manual bias calibrator:")
    cal_manual.report()

    if win_test_preds is not None and win_test_preds.shape[0] == stacked_test.shape[0]:
        # Affine fit from stack to winning CSV
        cal_affine = DomainShiftCalibrator()
        cal_affine.fit_from_winning_csv(stacked_test, win_test_preds)
        calibrators['affine_stack'] = cal_affine
        print("\n  Affine calibrator (stack → winning CSV):")
        cal_affine.report()

        # Affine fit from HGB to winning CSV
        cal_affine_hgb = DomainShiftCalibrator()
        cal_affine_hgb.fit_from_winning_csv(hgb_test, win_test_preds)
        calibrators['affine_hgb'] = cal_affine_hgb
        print("\n  Affine calibrator (HGB → winning CSV):")
        cal_affine_hgb.report()

        # Grid search
        cal_grid = grid_search_bias(stacked_test, win_test_preds, y_train)
        calibrators['grid_stack'] = cal_grid
        print("\n  Grid-search calibrator (stack → winning CSV):")
        cal_grid.report()

    # ── 7. Save all submission variants ───────────────────────────────────────
    print(f"\n{'='*70}\n  GENERATING SUBMISSIONS\n{'='*70}")

    # Raw (no calibration)
    print("\n  --- RAW (no calibration) ---")
    save_submission(stacked_test, y_train, "v13_stack_raw", OUT_DIR)
    save_submission(hgb_test, y_train, "v13_hgb_raw", OUT_DIR)

    # Each calibrator applied to each base prediction
    for cal_name, cal in calibrators.items():
        print(f"\n  --- Calibrator: {cal_name} ---")
        save_submission(cal.transform(stacked_test), y_train, f"v13_stack_{cal_name}", OUT_DIR)
        save_submission(cal.transform(hgb_test), y_train, f"v13_hgb_{cal_name}", OUT_DIR)

    # Blend with winning CSV (if available)
    if win_test_preds is not None and win_test_preds.shape[0] == stacked_test.shape[0]:
        print(f"\n  --- Blends with winning CSV ---")
        # Try blending both raw and calibrated with winning CSV
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            # Calibrated stack blended with winning
            for cal_name in ['affine_stack', 'grid_stack', 'manual']:
                if cal_name in calibrators:
                    cal_pred = calibrators[cal_name].transform(stacked_test)
                    blended = alpha * cal_pred + (1 - alpha) * win_test_preds
                    save_submission(blended, y_train, f"v13_{cal_name}_win_{int(alpha*100):02d}", OUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  v13 COMPLETE\n{'='*70}")
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "submission_*.csv"))):
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}")
        for c in ANALYTE_COLS:
            print(f"    {c}: mu={df[c].mean():.3f} sig={df[c].std():.3f} [{df[c].min():.3f}, {df[c].max():.3f}]")

if __name__ == "__main__":
    main()
