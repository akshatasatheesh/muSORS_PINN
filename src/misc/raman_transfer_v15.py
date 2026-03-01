#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Raman Transfer v15 — PINN vs Vanilla × Physics-Meta vs Vanilla-Meta        ║
║                                                                              ║
║  CORE EXPERIMENT: 2×2 factorial comparison                                   ║
║                                                                              ║
║          ┌──────────────────┬──────────────────────────────┐                 ║
║          │  Fine-Tuning →   │  Vanilla FT  │  PINN FT      │                ║
║          ├──────────────────┼──────────────┼───────────────┤                 ║
║          │ Vanilla Meta     │  Model A     │  Model B       │                ║
║          │ Physics Meta     │  Model C     │  Model D       │                ║
║          └──────────────────┴──────────────┴───────────────┘                 ║
║                                                                              ║
║  Model A: Vanilla Reptile  → Vanilla fine-tune (pure data-driven baseline)   ║
║  Model B: Vanilla Reptile  → PINN fine-tune   (physics only at fine-tune)    ║
║  Model C: Physics Reptile  → Vanilla fine-tune (physics only at meta-learn)  ║
║  Model D: Physics Reptile  → PINN fine-tune   (full physics pipeline)        ║
║                                                                              ║
║  GRAPHS PRODUCED:                                                            ║
║    1. Bar chart: OOF R² per analyte × model (grouped)                        ║
║    2. Training loss curves: Vanilla FT vs PINN FT convergence                ║
║    3. Scatter plots: Predicted vs Actual for each model (3×4 grid)           ║
║    4. Improvement heatmap: Δ R² from adding physics at each stage            ║
║    5. Residual distributions: per-model error histograms                     ║
║    6. Meta-learning convergence: vanilla vs physics meta loss over epochs    ║
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = "data"
OUT_DIR  = "./v15_outputs"
SEED     = 42

WN_LOW, WN_HIGH, WN_STEP = 300.0, 1942.0, 1.0
DEVICE_FILES = [
    "anton_532.csv", "anton_785.csv", "kaiser.csv", "mettler_toledo.csv",
    "metrohm.csv", "tec5.csv", "timegate.csv", "tornado.csv",
]

SAVGOL_WINDOW, SAVGOL_POLY, SAVGOL_DERIV = 7, 2, 0
SNIP_HALFWIN  = 20
DERIV1_WINDOW, DERIV2_WINDOW = 21, 35

BAND_REGIONS = {
    'glucose_1125': (800,850), 'glucose_1065': (740,790), 'naac_930': (600,660),
    'naac_1415': (1085,1145), 'mgso4_980': (650,710), 'mgso4_450': (120,180),
    'water_broad': (1300,1500), 'fingerprint_low': (0,400),
    'fingerprint_mid': (400,900), 'fingerprint_hi': (900,1400),
}
KNOWN_BAND_CENTERS_CM = {0: [1125,1065,525], 1: [930,1415,650], 2: [980,450,615]}
PEAK_LOCALITY_SIGMA_CM = 60.0

ANALYTE_NAMES = ["Glucose", "NaAc", "MgSO4"]
ANALYTE_COLS  = ["Glucose", "Sodium Acetate", "Magnesium Sulfate"]
N_FOLDS, N_OUT = 5, 3

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

# Physics loss weights
PHYS_RAMP_EPOCHS = 220
LW_INV       = 0.03
LW_SHIFT_INV = 0.02
LW_MIX       = 0.12
LW_SMOOTH    = 0.025
LW_SPARSE    = 0.004
LW_PEAK_LOC  = 0.008

# Physics meta-learning inner loop weights (gentler for meta stability)
META_LW_MIX      = 0.05
META_LW_SMOOTH   = 0.01
META_LW_SPARSE   = 0.002
META_LW_PEAK_LOC = 0.004

NN_AUTO_GATE_R2 = 0.05
WINNING_CSV_CANDIDATES = [
    "submission_v8_win_15.csv", "submission_pp_hgb_7_2_0.csv",
    os.path.join("data","submission_pp_hgb_7_2_0.csv"),
    os.path.join("data","submission_v8_win_15.csv"),
]

# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(s): np.random.seed(s); torch.manual_seed(s)
def make_grid(lo, hi, step): return np.arange(lo, hi + step/2, step)

class TargetScaler:
    def fit(self, y): self.mu = y.mean(0); self.sd = y.std(0)+1e-8; return self
    def transform(self, y): return (y - self.mu) / self.sd
    def inverse(self, yn): return yn * self.sd + self.mu

def find_winning_csv():
    for p in WINNING_CSV_CANDIDATES:
        if os.path.exists(p): return p
    return None

def load_winning_csv(path):
    return pd.read_csv(path)[ANALYTE_COLS].values.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def _clean(s):
    if isinstance(s, str):
        s = s.replace('[','').replace(']','')
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
    X_norm = X_interp / (np.max(X_interp)+1e-12)
    y = df.iloc[:,-5:-2].values.astype(np.float64)
    return X_norm.astype(np.float32), y.astype(np.float32)

def load_plate(path, is_train):
    df = pd.read_csv(path) if is_train else pd.read_csv(path, header=None)
    if is_train:
        y = df[['Glucose (g/L)','Sodium Acetate (g/L)','Magnesium Acetate (g/L)']].dropna().values
        X = df.iloc[:,:-4]
    else: X = df; y = None
    X.columns = ["sample_id"]+[str(i) for i in range(X.shape[1]-1)]
    X['sample_id'] = X['sample_id'].ffill()
    if is_train: X['sample_id'] = X['sample_id'].str.strip()
    else: X['sample_id'] = X['sample_id'].astype(str).str.strip().str.replace('sample','').astype(int)
    for c in X.columns[1:]: X[c] = X[c].apply(_clean)
    X = X.drop('sample_id',axis=1).interpolate(axis=1,limit_direction='both').fillna(0)
    X2048 = X.values.reshape(-1,2,2048).mean(axis=1)
    return X2048.astype(np.float32), y.astype(np.float32) if y is not None else None

def plate_to_grid(X2048, grid_wns, s=65.0, e=3350.0):
    wns_full = np.linspace(s,e,2048); mask = (wns_full>=WN_LOW)&(wns_full<=WN_HIGH)
    wns_sel = wns_full[mask]; X_sel = X2048[:,mask]
    return np.array([np.interp(grid_wns,wns_sel,row) for row in X_sel]).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Chemometrics Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def apply_msc(X, ref=None):
    X64 = X.astype(np.float64); ref = X64.mean(0) if ref is None else ref.astype(np.float64)
    out = np.empty_like(X64)
    for i in range(len(X64)):
        m,b = np.polyfit(ref,X64[i],1); out[i] = (X64[i]-b)/(m+1e-12)
    return out.astype(np.float32), ref.astype(np.float32)

def apply_baseline(X, max_half_window=SNIP_HALFWIN):
    try:
        import pybaselines; fitter = pybaselines.Baseline()
        out = np.empty_like(X, dtype=np.float64)
        for i,row in enumerate(X.astype(np.float64)):
            base,_ = fitter.snip(row, max_half_window=max_half_window, decreasing=True, smooth_half_window=3)
            out[i] = row - base
        return out.astype(np.float32)
    except ImportError:
        t = np.linspace(0,1,X.shape[1]); out = np.empty_like(X,dtype=np.float64)
        for i,row in enumerate(X.astype(np.float64)): out[i] = row - np.polyval(np.polyfit(t,row,3),t)
        return out.astype(np.float32)

class ChemometricsPreprocessor:
    def __init__(self): self.msc_ref=None; self.scaler=None
    def fit_transform(self, X):
        X, self.msc_ref = apply_msc(X); X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64),SAVGOL_WINDOW,SAVGOL_POLY,SAVGOL_DERIV,axis=1).astype(np.float32)
        self.scaler = StandardScaler(); return self.scaler.fit_transform(X).astype(np.float32)
    def transform(self, X):
        X,_ = apply_msc(X,self.msc_ref); X = apply_baseline(X)
        X = savgol_filter(X.astype(np.float64),SAVGOL_WINDOW,SAVGOL_POLY,SAVGOL_DERIV,axis=1).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Physics-Informed Feature Engineering (for HGB)
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsFeatureExtractor:
    def __init__(self, n_components=3):
        self.n_components=n_components; self.nmf=None
    def fit(self, X_phys):
        X = np.clip(X_phys,0,None).astype(np.float64)
        self.nmf = NMF(n_components=self.n_components,init='nndsvda',max_iter=500,random_state=SEED,l1_ratio=0.1,alpha_W=0.01)
        W = self.nmf.fit_transform(X); self.basis = self.nmf.components_
        self.baseline = np.clip(X.mean(0)-W.mean(0)@self.basis,0,None)
        print(f"    NMF recon error: {self.nmf.reconstruction_err_:.4f}"); return self
    def transform(self, X_phys):
        X = np.clip(X_phys,0,None).astype(np.float64); N=X.shape[0]
        W = self.nmf.transform(X); X_hat = W@self.basis+self.baseline; residual = X-X_hat
        res_total=np.sqrt(np.mean(residual**2,1)); res_max=np.max(np.abs(residual),1)
        res_pos=np.mean(np.clip(residual,0,None),1); res_neg=np.mean(np.clip(residual,None,0),1)
        band_res=[]
        for _,(lo,hi) in BAND_REGIONS.items():
            hi=min(hi,X.shape[1]);lo=max(lo,0)
            band_res.append(np.mean(residual[:,lo:hi],1) if hi>lo else np.zeros(N))
        band_res=np.stack(band_res,1); W_safe=W+1e-10
        ratios=np.stack([W_safe[:,0]/W_safe[:,1],W_safe[:,0]/W_safe[:,2],W_safe[:,1]/W_safe[:,2]],1)
        cos_sims=[]
        for k in range(self.n_components):
            b=self.basis[k]; cos_sims.append((X@b)/(np.linalg.norm(X,1)*np.linalg.norm(b)+1e-10))
        cos_sims=np.stack(cos_sims,1)
        return np.hstack([W,res_total[:,None],res_max[:,None],res_pos[:,None],res_neg[:,None],
                          band_res,ratios,cos_sims]).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Standard Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
def compute_derivatives(X):
    d1=savgol_filter(X.astype(np.float64),DERIV1_WINDOW,2,deriv=1,axis=1).astype(np.float32)
    d2=savgol_filter(X.astype(np.float64),DERIV2_WINDOW,2,deriv=2,axis=1).astype(np.float32)
    return d1,d2

def extract_peak_features(spectra):
    feats=[]
    for spec in spectra:
        thresh=np.percentile(spec,90); peaks,_=find_peaks(spec,height=thresh,prominence=0.5)
        widths,_,_,_=peak_widths(spec,peaks,rel_height=0.5)
        feats.append([len(peaks),np.sum(spec[peaks]) if len(peaks)>0 else 0,
            np.mean(spec[peaks]) if len(peaks)>0 else 0, np.std(spec[peaks]) if len(peaks)>1 else 0,
            np.mean(widths) if len(widths)>0 else 0, np.max(spec[peaks]) if len(peaks)>0 else 0])
    return np.array(feats,dtype=np.float32)

def compute_statistical_features(s):
    return np.stack([np.mean(s,1),np.std(s,1),skew(s,axis=1),kurtosis(s,axis=1),
        np.max(s,1),np.min(s,1),np.ptp(s,1),np.sum(s**2,axis=1)],1).astype(np.float32)

def compute_band_integrals(s):
    r=[]
    for _,(lo,hi) in BAND_REGIONS.items():
        hi=min(hi,s.shape[1]);lo=max(lo,0); r.extend([np.sum(s[:,lo:hi],1),np.mean(s[:,lo:hi],1)])
    return np.stack(r,1).astype(np.float32)

def compute_peak_ratios(s):
    n=s.shape[1]
    def _b(lo,hi): return np.mean(s[:,max(0,lo):min(hi,n)],1)+1e-10
    g,na,mg,w,fp=_b(800,850),_b(600,660),_b(650,710),_b(1300,1500),_b(400,900)
    return np.stack([g/w,na/w,mg/w,g/fp,na/fp,mg/fp,g/na,g/mg,na/mg],1).astype(np.float32)

class FeatureEngineer:
    def __init__(self): self.scalers={}; self.pca=None; self.phys_scaler=None
    def build_features(self, X_pp, X_raw, phys_feats=None, fit=True):
        d1,d2=compute_derivatives(X_raw); peaks=extract_peak_features(X_pp)
        stats=compute_statistical_features(X_pp); bands=compute_band_integrals(X_pp)
        ratios=compute_peak_ratios(X_pp)
        fs={'spec_only':X_pp.copy(),'spec_d1':np.hstack([X_pp,d1]),
            'combined_all':np.hstack([X_pp,d1,d2,stats,peaks,bands,ratios])}
        if phys_feats is not None:
            if fit: self.phys_scaler=StandardScaler(); pf=self.phys_scaler.fit_transform(phys_feats).astype(np.float32)
            else: pf=self.phys_scaler.transform(phys_feats).astype(np.float32)
            fs['phys_only']=pf; fs['spec_phys']=np.hstack([X_pp,pf])
            fs['combined_phys']=np.hstack([X_pp,d1,d2,stats,peaks,bands,ratios,pf])
        if fit: self.pca=PCA(n_components=min(62,X_pp.shape[0]-1),random_state=SEED); fs['pca']=self.pca.fit_transform(X_pp)
        else: fs['pca']=self.pca.transform(X_pp)
        for name in fs:
            if name=='phys_only': continue
            if fit: sc=StandardScaler(); fs[name]=sc.fit_transform(fs[name]).astype(np.float32); self.scalers[name]=sc
            else: fs[name]=self.scalers[name].transform(fs[name]).astype(np.float32)
        return fs

# ══════════════════════════════════════════════════════════════════════════════
# Multi-Model Training + Stacking
# ══════════════════════════════════════════════════════════════════════════════
def create_base_models(has_phys=True):
    configs=[]
    configs.append({'name':'HGB_spec','feature_set':'spec_only','models':[
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.1,max_leaf_nodes=15,l2_regularization=0.1,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED)]})
    configs.append({'name':'HGB_comb','feature_set':'combined_all','models':[
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=10,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
        lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=10,learning_rate=0.1,max_leaf_nodes=15,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED)]})
    if has_phys:
        configs.append({'name':'HGB_phys','feature_set':'spec_phys','models':[
            lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=20,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
            lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=15,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.15,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED)]})
        configs.append({'name':'HGB_cphys','feature_set':'combined_phys','models':[
            lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=4,min_samples_leaf=12,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED),
            lambda:HistGradientBoostingRegressor(max_iter=500,max_depth=3,min_samples_leaf=18,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=0.2,max_bins=128,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=SEED)]})
    configs.append({'name':'ET_comb','feature_set':'combined_all','models':[
        lambda:ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_split=8,min_samples_leaf=5,max_features=0.45,bootstrap=True,max_samples=0.8,ccp_alpha=1e-4,random_state=SEED,n_jobs=-1),
        lambda:ExtraTreesRegressor(n_estimators=500,max_depth=14,min_samples_split=10,min_samples_leaf=6,max_features=0.50,bootstrap=True,max_samples=0.75,ccp_alpha=1e-4,random_state=SEED,n_jobs=-1),
        lambda:ExtraTreesRegressor(n_estimators=400,max_depth=10,min_samples_split=8,min_samples_leaf=8,max_features=0.40,bootstrap=True,max_samples=0.8,ccp_alpha=2e-4,random_state=SEED,n_jobs=-1)]})
    configs.append({'name':'ET_d1','feature_set':'spec_d1','models':[
        lambda:ExtraTreesRegressor(n_estimators=500,max_depth=12,min_samples_split=8,min_samples_leaf=5,max_features=0.30,bootstrap=True,max_samples=0.8,random_state=SEED,n_jobs=-1),
        lambda:ExtraTreesRegressor(n_estimators=400,max_depth=10,min_samples_split=10,min_samples_leaf=6,max_features=0.35,bootstrap=True,max_samples=0.75,random_state=SEED,n_jobs=-1)]})
    configs.append({'name':'Ridge_PCA','feature_set':'pca','models':[lambda:Ridge(alpha=1.0),lambda:Ridge(alpha=5.0),lambda:Ridge(alpha=0.1)]})
    configs.append({'name':'PLS_spec','feature_set':'spec_only','models':[
        lambda:PLSRegression(n_components=8),lambda:PLSRegression(n_components=12),lambda:PLSRegression(n_components=5)]})
    if has_phys:
        configs.append({'name':'Ridge_phys','feature_set':'phys_only','models':[lambda:Ridge(alpha=1.0),lambda:Ridge(alpha=10.0),lambda:Ridge(alpha=0.1)]})
    return configs

def train_base_models_cv(configs, feat_train, feat_test, y_train):
    kf=KFold(N_FOLDS,shuffle=True,random_state=SEED)
    all_oof={t:[] for t in range(N_OUT)}; all_test={t:[] for t in range(N_OUT)}; scores=[]
    for cfg in configs:
        fn=cfg['feature_set']
        if fn not in feat_train: print(f"  {cfg['name']:<12}  SKIPPED"); continue
        Xtr,Xte=feat_train[fn],feat_test[fn]; oof=np.zeros((len(y_train),N_OUT)); tp=np.zeros((Xte.shape[0],N_OUT))
        for _,(tri,vi) in enumerate(kf.split(Xtr)):
            Xf,Xv,yf=Xtr[tri],Xtr[vi],y_train[tri]; ft=np.zeros((Xte.shape[0],N_OUT))
            for t in range(N_OUT):
                fp2,tp2=np.zeros(len(vi)),np.zeros(Xte.shape[0])
                for mfn in cfg['models']: m=mfn();m.fit(Xf,yf[:,t]);fp2+=m.predict(Xv).ravel();tp2+=m.predict(Xte).ravel()
                n=len(cfg['models']); oof[vi,t]=fp2/n; ft[:,t]=tp2/n
            tp+=ft/N_FOLDS
        r2s=[r2_score(y_train[:,t],oof[:,t]) for t in range(N_OUT)]; avg=np.mean(r2s)
        print(f"  {cfg['name']:<12}  "+"  ".join(f"{n}={v:.4f}" for n,v in zip(ANALYTE_NAMES,r2s))+f"  Avg={avg:.4f}")
        scores.append({'name':cfg['name'],'target_r2s':r2s,'avg_r2':avg})
        for t in range(N_OUT): all_oof[t].append(oof[:,t]); all_test[t].append(tp[:,t])
    return all_oof,all_test,scores

def stack_models(all_oof, all_test, y_train, scores):
    st=np.zeros((list(all_test.values())[0][0].shape[0],N_OUT)); bo=np.zeros((len(y_train),N_OUT))
    for t in range(N_OUT):
        om,tm=np.column_stack(all_oof[t]),np.column_stack(all_test[t]); yt=y_train[:,t]; nm=om.shape[1]
        r2_avg=r2_score(yt,np.mean(om,1))
        w=np.maximum([s['target_r2s'][t] for s in scores],0); w=w/(w.sum()+1e-10); r2_wavg=r2_score(yt,om@w)
        def obj(ww): return mean_squared_error(yt,om@(ww/ww.sum()))
        res=minimize(obj,np.ones(nm)/nm,method='SLSQP',bounds=[(0,1)]*nm,constraints={'type':'eq','fun':lambda ww:ww.sum()-1})
        ow=res.x/res.x.sum(); r2_opt=r2_score(yt,om@ow)
        ridge=Ridge(alpha=2.0).fit(om,yt); r2_r=r2_score(yt,ridge.predict(om))
        lr=LinearRegression().fit(om,yt); r2_l=r2_score(yt,lr.predict(om))
        methods={'Simple_Avg':(r2_avg,np.mean(om,1),np.mean(tm,1)),'Wt_Avg':(r2_wavg,om@w,tm@w),
            'Opt_Wts':(r2_opt,om@ow,tm@ow),'Ridge_Stack':(r2_r,ridge.predict(om),ridge.predict(tm)),
            'LR_Stack':(r2_l,lr.predict(om),lr.predict(tm))}
        sm=sorted(methods.items(),key=lambda x:x[1][0],reverse=True)
        print(f"\n  {ANALYTE_NAMES[t]}:")
        for mn,(r2,_,_) in sm: print(f"    {mn:<15} OOF R2: {r2:.4f}")
        _,bo_t,st_t=sm[0][1]; st[:,t]=st_t; bo[:,t]=bo_t; print(f"    -> Best: {sm[0][0]}")
    print(f"\n  Stacked OOF R2: {np.mean([r2_score(y_train[:,t],bo[:,t]) for t in range(N_OUT)]):.4f}")
    return st,bo

# ══════════════════════════════════════════════════════════════════════════════
# [v10] Physics Modules
# ══════════════════════════════════════════════════════════════════════════════
def wavenumber_shift_augment_np(X, rng, max_shift_bins=3.0):
    B,V=X.shape; shifts=rng.uniform(-max_shift_bins,max_shift_bins,B); out=np.empty_like(X); grid=np.arange(V,dtype=np.float64)
    for i in range(B): out[i]=np.interp(grid-shifts[i],grid,X[i].astype(np.float64),left=X[i,0],right=X[i,-1])
    return out.astype(np.float32)

def wavenumber_shift_augment_torch(X, max_shift_bins=3.0):
    B,V=X.shape; shifts=(torch.rand(B,device=X.device)*2-1)*max_shift_bins
    bg=torch.linspace(-1,1,V,device=X.device).unsqueeze(0).expand(B,-1)
    sg=bg-(shifts/V*2).unsqueeze(1); X4=X.unsqueeze(1).unsqueeze(2)
    g4=torch.stack([sg,torch.zeros_like(sg)],-1).unsqueeze(1)
    return F.grid_sample(X4,g4,mode='bilinear',padding_mode='border',align_corners=True).squeeze(1).squeeze(1)

def physics_curriculum_weight(epoch, final_weight):
    return min(epoch/max(PHYS_RAMP_EPOCHS,1),1.0)*final_weight

def build_peak_locality_masks_torch(n_bins):
    wns=np.arange(WN_LOW,WN_LOW+n_bins*WN_STEP,WN_STEP)[:n_bins]
    masks=np.zeros((N_OUT,n_bins),dtype=np.float32)
    for a,centers in KNOWN_BAND_CENTERS_CM.items():
        for c in centers: masks[a]+=np.exp(-0.5*((wns-c)/PEAK_LOCALITY_SIGMA_CM)**2)
    for a in range(N_OUT):
        mx=masks[a].max()
        if mx>1e-8: masks[a]/=mx
    return torch.from_numpy(masks)

class AffineCalibrationHead(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(enc_dim,16),nn.GELU(),nn.Linear(16,2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,h):
        out=self.net(h); return torch.exp(out[:,0].clamp(-2,2)),out[:,1]

class LinearMixingDecoder(nn.Module):
    def __init__(self, n_bins, n_analytes=3, enc_dim=ENC_H2):
        super().__init__()
        self.n_bins=n_bins; self.n_analytes=n_analytes
        self.log_basis=nn.Parameter(0.02*torch.randn(n_analytes,n_bins))
        self.log_base=nn.Parameter(torch.zeros(n_bins))
        self.calib=AffineCalibrationHead(enc_dim)
        self.register_buffer('peak_locality_mask',build_peak_locality_masks_torch(n_bins))
    @property
    def basis(self): return F.softplus(self.log_basis)
    @property
    def baseline(self): return F.softplus(self.log_base)
    def reconstruct(self,c,h):
        X=self.baseline.unsqueeze(0)+c@self.basis; a,b=self.calib(h)
        return a.unsqueeze(1)*X+b.unsqueeze(1)
    @staticmethod
    def _area_normalize(x,eps=1e-8): return x/x.sum(1,keepdim=True).clamp(min=eps)
    def reconstruction_loss(self,c,h,X_phys):
        Xh=self.reconstruct(c,h).clamp(min=0)
        return F.mse_loss(self._area_normalize(Xh),self._area_normalize(X_phys.clamp(min=0)))
    def smoothness_loss(self):
        b,base=self.basis,self.baseline
        tv=(b[:,1:]-b[:,:-1]).abs().mean()+(base[1:]-base[:-1]).abs().mean()
        lap=(b[:,2:]-2*b[:,1:-1]+b[:,:-2]).pow(2).mean()+(base[2:]-2*base[1:-1]+base[:-2]).pow(2).mean()
        return tv+0.5*lap
    def sparsity_loss(self): return self.basis.mean()
    def peak_locality_loss(self): return (self.basis*(1.0-self.peak_locality_mask)).mean()

def compute_physics_loss(pred_c, enc_h, X_phys, mix_mod, epoch):
    losses={}
    L_mix=mix_mod.reconstruction_loss(pred_c,enc_h,X_phys); losses["L_mix"]=float(L_mix.detach())
    L_sm=mix_mod.smoothness_loss(); losses["L_smooth"]=float(L_sm.detach())
    L_sp=mix_mod.sparsity_loss(); losses["L_sparse"]=float(L_sp.detach())
    L_pl=mix_mod.peak_locality_loss(); losses["L_peak_loc"]=float(L_pl.detach())
    lw_mix=physics_curriculum_weight(epoch,LW_MIX); lw_sm=physics_curriculum_weight(epoch,LW_SMOOTH)
    lw_sp=physics_curriculum_weight(epoch,LW_SPARSE); lw_pl=physics_curriculum_weight(epoch,LW_PEAK_LOC)
    total=lw_mix*L_mix+lw_sm*L_sm+lw_sp*L_sp+lw_pl*L_pl
    return total, losses

def compute_physics_loss_meta(pred_c, enc_h, X_phys, mix_mod):
    """Physics loss for meta-learning inner loop (no curriculum ramp)."""
    L_mix=mix_mod.reconstruction_loss(pred_c,enc_h,X_phys)
    L_sm=mix_mod.smoothness_loss(); L_sp=mix_mod.sparsity_loss(); L_pl=mix_mod.peak_locality_loss()
    return META_LW_MIX*L_mix + META_LW_SMOOTH*L_sm + META_LW_SPARSE*L_sp + META_LW_PEAK_LOC*L_pl

# ══════════════════════════════════════════════════════════════════════════════
# NN Architecture
# ══════════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self,n_in):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,ENC_H1),nn.BatchNorm1d(ENC_H1),nn.GELU(),nn.Dropout(DROPOUT),
                               nn.Linear(ENC_H1,ENC_H2),nn.BatchNorm1d(ENC_H2),nn.GELU())
    def forward(self,z): return self.net(z)

class Head(nn.Module):
    def __init__(self,use_softplus=True):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(ENC_H2,HEAD_H),nn.GELU(),nn.Dropout(FT_DROPOUT),nn.Linear(HEAD_H,N_OUT))
        self.use_softplus=use_softplus
    def forward(self,x): out=self.net(x); return F.softplus(out) if self.use_softplus else out

class MetaModel(nn.Module):
    def __init__(self,n_in,use_softplus=True):
        super().__init__(); self.encoder=Encoder(n_in); self.head=Head(use_softplus)
    def forward(self,z): return self.head(self.encoder(z))
    def forward_with_hidden(self,z): h=self.encoder(z); return self.head(h),h

def weighted_mse(pred,true,tw): return ((pred-true)**2*tw).mean()
def compute_target_weights(y):
    var=np.var(y,0); inv=1.0/(var+1e-8); inv/=inv.sum(); return torch.tensor(inv,dtype=torch.float32)

def augment(Z,rng):
    Z=Z.copy(); Z*=rng.uniform(*AUG_SCALE,(len(Z),1)).astype(np.float32)
    Z+=rng.normal(0,AUG_NOISE,Z.shape).astype(np.float32); return Z

def augment_spectral(X_phys,rng,max_shift=3.0):
    X=wavenumber_shift_augment_np(X_phys,rng,max_shift)
    X*=rng.uniform(0.95,1.05,(len(X),1)).astype(np.float32)
    X+=rng.normal(0,0.005,X.shape).astype(np.float32)
    return np.clip(X,0,None).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# [A] VANILLA Meta-Learning (Reptile, data-only inner loop)
# ══════════════════════════════════════════════════════════════════════════════
def reptile_inner_loop_vanilla(model, Z_sup, y_sup, inner_lr, inner_steps):
    opt=torch.optim.SGD(model.parameters(),lr=inner_lr)
    tw=compute_target_weights(y_sup.cpu().numpy()).to(y_sup.device)
    for _ in range(inner_steps):
        opt.zero_grad(); weighted_mse(model(Z_sup),y_sup,tw).backward(); opt.step()

def meta_train_vanilla(model, device_z, device):
    """Standard Reptile: data loss only in inner loop."""
    print(f"\n{'='*70}\n  [A] Vanilla Reptile Meta-Training ({META_EPOCHS} epochs)\n{'='*70}")
    rng=np.random.RandomState(SEED); losses=[]
    for ep in range(META_EPOCHS):
        old=copy.deepcopy(model.state_dict()); ep_loss=0
        for _ in range(META_TASKS_PER_EPOCH):
            dx,dy=device_z[rng.randint(len(device_z))]
            idx=rng.choice(len(dx),min(K_SUPPORT,len(dx)),False)
            Z_s=torch.tensor(dx[idx],dtype=torch.float32).to(device)
            y_s=torch.tensor(dy[idx],dtype=torch.float32).to(device)
            reptile_inner_loop_vanilla(model,Z_s,y_s,INNER_LR,INNER_STEPS)
            with torch.no_grad():
                tw=compute_target_weights(y_s.cpu().numpy()).to(device)
                ep_loss+=float(weighted_mse(model(Z_s),y_s,tw))
        new=model.state_dict()
        for k in old: new[k]=old[k]+REPTILE_OUTER_LR*(new[k]-old[k])
        model.load_state_dict(new)
        losses.append(ep_loss/META_TASKS_PER_EPOCH)
        if (ep+1)%30==0: print(f"  epoch {ep+1}/{META_EPOCHS}  loss={losses[-1]:.6f}")
    return model, losses

# ══════════════════════════════════════════════════════════════════════════════
# [B] PHYSICS-GUIDED Meta-Learning (Reptile with physics in inner loop)
# ══════════════════════════════════════════════════════════════════════════════
def reptile_inner_loop_physics(model, Z_sup, y_sup, X_phys_sup, n_bins, inner_lr, inner_steps, device):
    """
    Physics-guided inner loop: data loss + physics reconstruction loss.
    A fresh LinearMixingDecoder is created per task (disposable).
    Physics gradients guide the encoder to learn spectrally-consistent representations.
    """
    mix_mod = LinearMixingDecoder(n_bins, N_OUT, ENC_H2).to(device)
    all_params = list(model.parameters()) + list(mix_mod.parameters())
    opt = torch.optim.SGD(all_params, lr=inner_lr)
    tw = compute_target_weights(y_sup.cpu().numpy()).to(device)

    for _ in range(inner_steps):
        opt.zero_grad()
        pred, h = model.forward_with_hidden(Z_sup)
        L_data = weighted_mse(pred, y_sup, tw)
        # Physics: convert normalized predictions to physical scale
        pred_phys = F.softplus(pred)  # nonneg concentrations
        L_phys = compute_physics_loss_meta(pred_phys, h, X_phys_sup, mix_mod)
        loss = L_data + L_phys
        loss.backward(); opt.step()

def meta_train_physics(model, device_z, device_phys_z, n_bins, device):
    """
    Physics-Guided Reptile: inner loop uses BOTH data loss AND physics loss.
    device_phys_z: list of (Z_pp, y, X_phys) tuples per device.
    """
    print(f"\n{'='*70}\n  [B] Physics-Guided Reptile Meta-Training ({META_EPOCHS} epochs)\n{'='*70}")
    rng=np.random.RandomState(SEED); losses=[]
    for ep in range(META_EPOCHS):
        old=copy.deepcopy(model.state_dict()); ep_loss=0
        for _ in range(META_TASKS_PER_EPOCH):
            task_idx = rng.randint(len(device_phys_z))
            dx, dy, dx_phys = device_phys_z[task_idx]
            idx=rng.choice(len(dx),min(K_SUPPORT,len(dx)),False)
            Z_s=torch.tensor(dx[idx],dtype=torch.float32).to(device)
            y_s=torch.tensor(dy[idx],dtype=torch.float32).to(device)
            Xp_s=torch.tensor(dx_phys[idx],dtype=torch.float32).to(device)
            reptile_inner_loop_physics(model,Z_s,y_s,Xp_s,n_bins,INNER_LR,INNER_STEPS,device)
            with torch.no_grad():
                tw=compute_target_weights(y_s.cpu().numpy()).to(device)
                ep_loss+=float(weighted_mse(model(Z_s),y_s,tw))
        new=model.state_dict()
        for k in old: new[k]=old[k]+REPTILE_OUTER_LR*(new[k]-old[k])
        model.load_state_dict(new)
        losses.append(ep_loss/META_TASKS_PER_EPOCH)
        if (ep+1)%30==0: print(f"  epoch {ep+1}/{META_EPOCHS}  loss={losses[-1]:.6f}")
    return model, losses

# ══════════════════════════════════════════════════════════════════════════════
# Fine-Tuning: Vanilla (no physics) and PINN (with physics)
# ══════════════════════════════════════════════════════════════════════════════
def fine_tune_vanilla(meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw, device, tw, seed=42):
    rng=np.random.RandomState(seed); model=copy.deepcopy(meta_model).to(device)
    tscaler=TargetScaler().fit(y_tr_raw); y_tr_n=tscaler.transform(y_tr_raw).astype(np.float32)
    ds=TensorDataset(torch.tensor(Z_tr),torch.tensor(y_tr_n))
    dl=DataLoader(ds,batch_size=FT_BATCH,shuffle=True,drop_last=False)
    opt=torch.optim.AdamW(model.parameters(),lr=FT_LR,weight_decay=FT_WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,FT_EPOCHS,1e-6)
    tw_t=tw.to(device); best_r2,best_sd,best_ts,wait=-np.inf,None,None,0; hist={'data':[],'val_r2':[]}
    for ep in range(FT_EPOCHS):
        model.train(); ep_loss=0; nb=0
        for Zb,yb in dl:
            Zb,yb=Zb.to(device),yb.to(device)
            Za=torch.tensor(augment(Zb.cpu().numpy(),rng),dtype=torch.float32).to(device)
            pred=model(Zb); pred_a=model(Za)
            L_data=weighted_mse(pred,yb,tw_t); L_aug=LW_INV*F.mse_loss(pred,pred_a)
            loss=L_data+L_aug; opt.zero_grad(); loss.backward(); opt.step()
            ep_loss+=float(L_data.detach()); nb+=1
        sched.step(); hist['data'].append(ep_loss/nb)
        if (ep+1)%5==0:
            model.eval()
            with torch.no_grad(): pv=model(torch.tensor(Z_val,device=device)).cpu().numpy()
            r2=np.mean([r2_score(y_val_raw[:,i],tscaler.inverse(pv)[:,i]) for i in range(N_OUT)])
            hist['val_r2'].append(r2)
            if r2>best_r2: best_r2,best_sd,best_ts,wait=r2,copy.deepcopy(model.state_dict()),tscaler,0
            else: wait+=5
            if wait>=PATIENCE: break
    if best_sd: model.load_state_dict(best_sd)
    return model, best_r2, best_ts, hist

def fine_tune_pinn(meta_model, Z_tr, y_tr_raw, Z_val, y_val_raw, X_phys_tr, X_phys_val, device, tw, seed=42):
    rng=np.random.RandomState(seed); model=copy.deepcopy(meta_model).to(device)
    tscaler=TargetScaler().fit(y_tr_raw); y_tr_n=tscaler.transform(y_tr_raw).astype(np.float32)
    n_bins=X_phys_tr.shape[1]; mix_mod=LinearMixingDecoder(n_bins,N_OUT,ENC_H2).to(device)
    ds=TensorDataset(torch.tensor(Z_tr),torch.tensor(y_tr_n),torch.tensor(X_phys_tr))
    dl=DataLoader(ds,batch_size=FT_BATCH,shuffle=True,drop_last=False)
    all_params=list(model.parameters())+list(mix_mod.parameters())
    opt=torch.optim.AdamW(all_params,lr=FT_LR,weight_decay=FT_WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,FT_EPOCHS,1e-6)
    tw_t=tw.to(device)
    tscaler_sd_t=torch.tensor(tscaler.sd,dtype=torch.float32,device=device)
    tscaler_mu_t=torch.tensor(tscaler.mu,dtype=torch.float32,device=device)
    best_r2,best_sd,best_ts,wait=-np.inf,None,None,0; hist={'data':[],'phys':[],'val_r2':[]}
    for ep in range(FT_EPOCHS):
        model.train(); mix_mod.train(); ep_data,ep_phys,nb=0,0,0
        for Zb,yb,Xb in dl:
            Zb,yb,Xb=Zb.to(device),yb.to(device),Xb.to(device)
            Za=torch.tensor(augment(Zb.cpu().numpy(),rng),dtype=torch.float32).to(device)
            Xb_sh=torch.tensor(augment_spectral(Xb.cpu().numpy(),rng,3.0),dtype=torch.float32,device=device)
            pred,h=model.forward_with_hidden(Zb); pred_a,_=model.forward_with_hidden(Za)
            L_data=weighted_mse(pred,yb,tw_t); L_aug=LW_INV*F.mse_loss(pred,pred_a)
            lw_sh=physics_curriculum_weight(ep,LW_SHIFT_INV)
            if lw_sh>0:
                pp=F.softplus(pred*tscaler_sd_t+tscaler_mu_t)
                Xh=mix_mod.reconstruct(pp,h).clamp(min=0); Xhs=wavenumber_shift_augment_torch(Xh,2.0)
                L_shift=lw_sh*F.mse_loss(LinearMixingDecoder._area_normalize(Xhs),LinearMixingDecoder._area_normalize(Xb_sh.clamp(min=0)))
            else: L_shift=torch.tensor(0.0,device=device)
            pred_phys=F.softplus(pred*tscaler_sd_t+tscaler_mu_t)
            L_phys,_=compute_physics_loss(pred_phys,h,Xb,mix_mod,ep)
            loss=L_data+L_aug+L_shift+L_phys; opt.zero_grad(); loss.backward(); opt.step()
            ep_data+=float(L_data.detach()); ep_phys+=float(L_phys.detach()); nb+=1
        sched.step(); hist['data'].append(ep_data/nb); hist['phys'].append(ep_phys/nb)
        if (ep+1)%5==0:
            model.eval(); mix_mod.eval()
            with torch.no_grad(): pv=model(torch.tensor(Z_val,device=device)).cpu().numpy()
            r2=np.mean([r2_score(y_val_raw[:,i],tscaler.inverse(pv)[:,i]) for i in range(N_OUT)])
            hist['val_r2'].append(r2)
            if r2>best_r2: best_r2,best_sd,best_ts,wait=r2,copy.deepcopy(model.state_dict()),tscaler,0
            else: wait+=5
            if wait>=PATIENCE: break
    if best_sd: model.load_state_dict(best_sd)
    return model, best_r2, best_ts, hist

def predict_tta(model, Z, tscaler, device, n=TTA_N):
    model.eval(); rng=np.random.RandomState(99); preds=[]
    with torch.no_grad():
        preds.append(model(torch.tensor(Z,device=device)).cpu().numpy())
        for _ in range(n-1): preds.append(model(torch.tensor(augment(Z,rng),device=device)).cpu().numpy())
    return tscaler.inverse(np.mean(preds,0))

# ══════════════════════════════════════════════════════════════════════════════
# CV runner for all 4 model variants
# ══════════════════════════════════════════════════════════════════════════════
def cv_run_model(meta_model, Z_train, y_train, Z_test, X_phys_train, X_phys_test,
                 device, tw, use_pinn=False, label="Model"):
    kf=KFold(N_FOLDS,shuffle=True,random_state=SEED)
    oof=np.zeros_like(y_train,dtype=np.float32); tp=np.zeros((Z_test.shape[0],N_OUT),dtype=np.float32)
    all_hist={'data':[],'phys':[],'val_r2':[]}
    print(f"\n{'='*70}\n  {label} ({N_FOLDS}-fold x {N_RESTARTS} restarts)\n{'='*70}")
    for fold,(tri,vi) in enumerate(kf.split(Z_train)):
        Ztr,Zv=Z_train[tri],Z_train[vi]; ytr,yv=y_train[tri],y_train[vi]
        Xptr,Xpv=X_phys_train[tri],X_phys_train[vi]
        bm,br2,bts=None,-np.inf,None; bhist=None
        for r in range(N_RESTARTS):
            sd=SEED+fold*100+r
            if use_pinn:
                m,vr2,ts,hist=fine_tune_pinn(meta_model,Ztr,ytr,Zv,yv,Xptr,Xpv,device,tw,sd)
            else:
                m,vr2,ts,hist=fine_tune_vanilla(meta_model,Ztr,ytr,Zv,yv,device,tw,sd)
            if vr2>br2: br2,bm,bts,bhist=vr2,m,ts,hist
        oof[vi]=predict_tta(bm,Zv,bts,device); tp+=predict_tta(bm,Z_test,bts,device)/N_FOLDS
        if bhist:
            for k in all_hist:
                if k in bhist and bhist[k]: all_hist[k].append(bhist[k])
        r2s=[r2_score(yv[:,i],oof[vi,i]) for i in range(3)]
        print(f"  Fold {fold+1}: "+"  ".join(f"{n}={v:.4f}" for n,v in zip(ANALYTE_NAMES,r2s))+f"  Avg={np.mean(r2s):.4f}")
    return oof, tp, all_hist

# ══════════════════════════════════════════════════════════════════════════════
# Domain-Shift Calibration
# ══════════════════════════════════════════════════════════════════════════════
class DomainShiftCalibrator:
    def __init__(self): self.scales=np.ones(N_OUT,dtype=np.float32); self.offsets=np.zeros(N_OUT,dtype=np.float32); self.method="none"
    def fit_from_winning_csv(self,our,win):
        self.method="winning_csv_affine"
        for i in range(N_OUT):
            A=np.column_stack([our[:,i],np.ones(len(our))]); r,_,_,_=np.linalg.lstsq(A,win[:,i],rcond=None)
            self.scales[i],self.offsets[i]=r[0],r[1]
    def transform(self,p):
        o=p.copy()
        for i in range(N_OUT): o[:,i]=self.scales[i]*p[:,i]+self.offsets[i]
        return o
    def report(self):
        print(f"    Method: {self.method}")
        for i,n in enumerate(ANALYTE_NAMES): print(f"    {n}: scale={self.scales[i]:.4f} offset={self.offsets[i]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Post-processing & Saving
# ══════════════════════════════════════════════════════════════════════════════
def post_process(preds, y_train):
    out=np.maximum(preds,0.0)
    for i,n in enumerate(ANALYTE_NAMES):
        lo=np.percentile(y_train[:,i],1); hi=np.percentile(y_train[:,i],99)
        mg=0.15*(hi-lo); out[:,i]=np.clip(out[:,i],max(0,lo-mg),hi+mg)
    return out

def save_submission(preds, y_train, name, out_dir):
    p=post_process(preds.copy(),y_train)
    sub=pd.DataFrame({"ID":np.arange(1,len(p)+1),"Glucose":p[:,0],"Sodium Acetate":p[:,1],"Magnesium Sulfate":p[:,2]})
    path=os.path.join(out_dir,f"submission_{name}.csv"); sub.to_csv(path,index=False)
    print(f"  -> Saved {path}"); return sub

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: 6 comprehensive comparison plots
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {'A':'#8ECAE6', 'B':'#219EBC', 'C':'#FFB703', 'D':'#FB8500'}
MODEL_LABELS = {
    'A': 'Vanilla Meta\n+ Vanilla FT',
    'B': 'Vanilla Meta\n+ PINN FT',
    'C': 'Physics Meta\n+ Vanilla FT',
    'D': 'Physics Meta\n+ PINN FT',
}
SHORT_LABELS = {'A':'Van-Meta + Van-FT', 'B':'Van-Meta + PINN-FT', 'C':'Phys-Meta + Van-FT', 'D':'Phys-Meta + PINN-FT'}

def plot_1_bar_chart(results, out_dir):
    """Bar chart: OOF R² per analyte per model."""
    fig,ax=plt.subplots(figsize=(12,6))
    x=np.arange(N_OUT); w=0.18
    for i,(key,label) in enumerate(SHORT_LABELS.items()):
        r2s=results[key]['r2s']
        bars=ax.bar(x+i*w-1.5*w, r2s, w, label=label, color=COLORS[key], edgecolor='white', linewidth=0.8)
        for bar,v in zip(bars,r2s): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.3f}', ha='center',va='bottom',fontsize=8,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([ANALYTE_COLS[i] for i in range(N_OUT)], fontsize=11)
    ax.set_ylabel('OOF R²', fontsize=12); ax.set_title('Per-Analyte OOF R² — 2×2 Factorial Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.set_ylim(bottom=min(0,min(min(results[k]['r2s']) for k in results)-0.05))
    ax.grid(axis='y',alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'plot1_r2_bar_chart.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot1_r2_bar_chart.png")

def plot_2_training_curves(results, out_dir):
    """Training loss curves: data loss convergence for vanilla vs PINN."""
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for key in ['A','B','C','D']:
        hists=results[key].get('hist',{})
        data_curves=hists.get('data',[])
        if data_curves:
            # Average across folds
            min_len=min(len(c) for c in data_curves)
            avg=np.mean([c[:min_len] for c in data_curves],0)
            axes[0].plot(avg, label=SHORT_LABELS[key], color=COLORS[key], linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Data Loss'); axes[0].set_title('Data Loss Convergence',fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # Physics loss (only B and D have it)
    for key in ['B','D']:
        hists=results[key].get('hist',{})
        phys_curves=hists.get('phys',[])
        if phys_curves:
            min_len=min(len(c) for c in phys_curves)
            avg=np.mean([c[:min_len] for c in phys_curves],0)
            axes[1].plot(avg, label=SHORT_LABELS[key], color=COLORS[key], linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Physics Loss'); axes[1].set_title('Physics Loss (PINN models only)',fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,'plot2_training_curves.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot2_training_curves.png")

def plot_3_scatter_pred_vs_actual(results, y_train, out_dir):
    """3×4 scatter grid: predicted vs actual for each analyte × model."""
    fig,axes=plt.subplots(N_OUT,4,figsize=(18,12))
    for col,(key,label) in enumerate(SHORT_LABELS.items()):
        oof=results[key]['oof']
        for row in range(N_OUT):
            ax=axes[row,col]; yt=y_train[:,row]; yp=oof[:,row]
            ax.scatter(yt,yp,c=COLORS[key],alpha=0.6,s=25,edgecolors='white',linewidth=0.3)
            lims=[min(yt.min(),yp.min())-0.5,max(yt.max(),yp.max())+0.5]
            ax.plot(lims,lims,'k--',alpha=0.4,linewidth=1)
            r2=r2_score(yt,yp); ax.text(0.05,0.92,f'R²={r2:.3f}',transform=ax.transAxes,fontsize=10,fontweight='bold',
                                          bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.8))
            if row==0: ax.set_title(label,fontsize=10,fontweight='bold')
            if col==0: ax.set_ylabel(ANALYTE_COLS[row],fontsize=10)
            if row==N_OUT-1: ax.set_xlabel('Actual',fontsize=9)
            ax.grid(alpha=0.2)
    plt.suptitle('Predicted vs Actual (OOF)',fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,'plot3_scatter_pred_vs_actual.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot3_scatter_pred_vs_actual.png")

def plot_4_improvement_heatmap(results, out_dir):
    """Heatmap: Δ R² improvement from adding physics at each stage."""
    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    # Effect of PINN fine-tuning (B-A, D-C)
    delta_ft = np.array([
        [results['B']['r2s'][t]-results['A']['r2s'][t] for t in range(N_OUT)],
        [results['D']['r2s'][t]-results['C']['r2s'][t] for t in range(N_OUT)]
    ])
    im=axes[0].imshow(delta_ft,cmap='RdYlGn',aspect='auto',vmin=-max(0.01,abs(delta_ft).max()),vmax=max(0.01,abs(delta_ft).max()))
    axes[0].set_yticks([0,1]); axes[0].set_yticklabels(['w/ Vanilla Meta','w/ Physics Meta'])
    axes[0].set_xticks(range(N_OUT)); axes[0].set_xticklabels(ANALYTE_NAMES)
    for i in range(2):
        for j in range(N_OUT): axes[0].text(j,i,f'{delta_ft[i,j]:+.4f}',ha='center',va='center',fontsize=11,fontweight='bold')
    axes[0].set_title('Δ R² from Adding Physics to Fine-Tuning\n(PINN − Vanilla FT)',fontweight='bold')
    plt.colorbar(im,ax=axes[0],shrink=0.8)
    # Effect of Physics Meta (C-A, D-B)
    delta_meta = np.array([
        [results['C']['r2s'][t]-results['A']['r2s'][t] for t in range(N_OUT)],
        [results['D']['r2s'][t]-results['B']['r2s'][t] for t in range(N_OUT)]
    ])
    im2=axes[1].imshow(delta_meta,cmap='RdYlGn',aspect='auto',vmin=-max(0.01,abs(delta_meta).max()),vmax=max(0.01,abs(delta_meta).max()))
    axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['w/ Vanilla FT','w/ PINN FT'])
    axes[1].set_xticks(range(N_OUT)); axes[1].set_xticklabels(ANALYTE_NAMES)
    for i in range(2):
        for j in range(N_OUT): axes[1].text(j,i,f'{delta_meta[i,j]:+.4f}',ha='center',va='center',fontsize=11,fontweight='bold')
    axes[1].set_title('Δ R² from Adding Physics to Meta-Learning\n(Phys-Meta − Vanilla Meta)',fontweight='bold')
    plt.colorbar(im2,ax=axes[1],shrink=0.8)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,'plot4_improvement_heatmap.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot4_improvement_heatmap.png")

def plot_5_residual_distributions(results, y_train, out_dir):
    """Residual histograms per model."""
    fig,axes=plt.subplots(1,N_OUT,figsize=(15,4.5))
    for t in range(N_OUT):
        ax=axes[t]; yt=y_train[:,t]
        for key in ['A','B','C','D']:
            resid=results[key]['oof'][:,t]-yt
            ax.hist(resid,bins=20,alpha=0.45,label=SHORT_LABELS[key],color=COLORS[key],edgecolor='white')
        ax.axvline(0,color='k',linestyle='--',alpha=0.5)
        ax.set_xlabel('Residual (Predicted − Actual)'); ax.set_title(ANALYTE_COLS[t],fontweight='bold')
        if t==0: ax.set_ylabel('Count')
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
    plt.suptitle('Residual Distributions (OOF)',fontsize=13,fontweight='bold')
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,'plot5_residual_distributions.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot5_residual_distributions.png")

def plot_6_meta_convergence(vanilla_meta_losses, physics_meta_losses, out_dir):
    """Meta-learning convergence: vanilla vs physics Reptile."""
    fig,ax=plt.subplots(figsize=(10,5))
    if vanilla_meta_losses: ax.plot(vanilla_meta_losses, label='Vanilla Reptile', color=COLORS['A'], linewidth=2)
    if physics_meta_losses: ax.plot(physics_meta_losses, label='Physics-Guided Reptile', color=COLORS['D'], linewidth=2)
    ax.set_xlabel('Meta-Epoch',fontsize=12); ax.set_ylabel('Inner-Loop Loss (post-adaptation)',fontsize=12)
    ax.set_title('Meta-Learning Convergence: Vanilla vs Physics-Guided Reptile',fontsize=14,fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(out_dir,'plot6_meta_convergence.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot6_meta_convergence.png")

def plot_7_summary_table(results, out_dir):
    """Summary comparison table as a figure."""
    fig,ax=plt.subplots(figsize=(12,4))
    ax.axis('off')
    col_labels=['Model','Meta-Learning','Fine-Tuning']+ANALYTE_NAMES+['Average R²']
    cell_data=[]
    for key in ['A','B','C','D']:
        meta='Vanilla' if key in ['A','B'] else 'Physics-Guided'
        ft='Vanilla' if key in ['A','C'] else 'PINN'
        r2s=results[key]['r2s']; avg=np.mean(r2s)
        cell_data.append([f'Model {key}',meta,ft]+[f'{r:.4f}' for r in r2s]+[f'{avg:.4f}'])
    table=ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.6)
    # Color header
    for j in range(len(col_labels)):
        table[0,j].set_facecolor('#2C3E50'); table[0,j].set_text_props(color='white',fontweight='bold')
    for i,key in enumerate(['A','B','C','D']):
        table[i+1,0].set_facecolor(COLORS[key])
    # Highlight best average
    avgs=[np.mean(results[k]['r2s']) for k in ['A','B','C','D']]
    best_idx=np.argmax(avgs)
    table[best_idx+1,len(col_labels)-1].set_facecolor('#2ECC71')
    table[best_idx+1,len(col_labels)-1].set_text_props(fontweight='bold')
    ax.set_title('v15 — Physics-Informed Neural Network: 2×2 Factorial Comparison',fontsize=14,fontweight='bold',pad=20)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir,'plot7_summary_table.png'),dpi=150,bbox_inches='tight'); plt.close()
    print("  -> Saved plot7_summary_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED); os.makedirs(OUT_DIR,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"v15 — PINN vs Vanilla × Physics-Meta vs Vanilla-Meta\nDevice: {device}")
    wns=make_grid(WN_LOW,WN_HIGH,WN_STEP); N_BINS=len(wns)

    # ── 0. Winning CSV
    win_path=find_winning_csv(); win_preds=None
    if win_path: print(f"\n[0] Winning CSV: {win_path}"); win_preds=load_winning_csv(win_path)
    else: print("\n[0] No winning CSV found.")

    # ── 1. Load data
    print("\n[1] Loading data ...")
    Xtr_raw,y_train=load_plate(os.path.join(DATA_DIR,"transfer_plate.csv"),True)
    Xte_raw,_=load_plate(os.path.join(DATA_DIR,"96_samples.csv"),False)
    Xtr=plate_to_grid(Xtr_raw,wns); Xte=plate_to_grid(Xte_raw,wns)
    print(f"  Train: {Xtr.shape}  Test: {Xte.shape}")

    # ── 1b. Physics spectra
    print("\n[1b] Physics features (NMF) ...")
    Xtr_phys=np.clip(apply_baseline(Xtr),0,None); Xte_phys=np.clip(apply_baseline(Xte),0,None)
    gmax=float(np.max(Xtr_phys))+1e-12
    Xtr_phys=(Xtr_phys/gmax).astype(np.float32); Xte_phys=(Xte_phys/gmax).astype(np.float32)
    phys_fe=PhysicsFeatureExtractor(N_OUT); phys_fe.fit(Xtr_phys)
    phys_train=phys_fe.transform(Xtr_phys); phys_test=phys_fe.transform(Xte_phys)

    # ── Load devices
    device_data=[]; device_data_phys=[]
    for fname in DEVICE_FILES:
        path=os.path.join(DATA_DIR,fname)
        if not os.path.exists(path): continue
        X_dev,y_dev=load_device(path,wns)
        if X_dev is None or len(X_dev)<K_SUPPORT: continue
        device_data.append((X_dev,y_dev))
        # Physics spectra for device data
        X_dev_phys=np.clip(apply_baseline(X_dev),0,None)
        X_dev_phys=(X_dev_phys/gmax).astype(np.float32)
        device_data_phys.append(X_dev_phys)
        print(f"    {fname:<25} {len(X_dev):4d} samples")
    use_meta=len(device_data)>0

    # ── 2. Preprocessing
    print("\n[2] Chemometrics preprocessing ...")
    prep=ChemometricsPreprocessor()
    all_raw=np.concatenate([Xtr]+[X for X,_ in device_data]) if use_meta else Xtr
    prep.fit_transform(all_raw)
    Xtr_pp=prep.transform(Xtr); Xte_pp=prep.transform(Xte)
    device_pp=[(prep.transform(X),y) for X,y in device_data] if use_meta else []

    # ── 3. Features
    print("\n[3] Feature engineering ...")
    fe=FeatureEngineer()
    feat_train=fe.build_features(Xtr_pp,Xtr,phys_feats=phys_train,fit=True)
    feat_test=fe.build_features(Xte_pp,Xte,phys_feats=phys_test,fit=False)

    # ── 4. Base ML models + Stacking
    print("\n[4] Multi-model training ...")
    configs=create_base_models(has_phys=True)
    all_oof,all_test,scores=train_base_models_cv(configs,feat_train,feat_test,y_train)
    print("\n[5] Stacking ...")
    stacked_test,stacked_oof=stack_models(all_oof,all_test,y_train,scores)

    # ══════════════════════════════════════════════════════════════════════════
    # 2×2 FACTORIAL EXPERIMENT
    # ══════════════════════════════════════════════════════════════════════════
    Z_train_nn=Xtr_pp.copy(); Z_test_nn=Xte_pp.copy()
    device_z=[(X.astype(np.float32),y) for X,y in device_pp] if use_meta else []
    tw=compute_target_weights(y_train)
    n_in=Z_train_nn.shape[1]

    # Build device_phys_z for physics meta-learning: (Z_pp, y, X_phys)
    device_phys_z=[]
    if use_meta:
        for i,(X_pp,y) in enumerate(device_pp):
            device_phys_z.append((X_pp.astype(np.float32), y, device_data_phys[i]))

    results = {}
    vanilla_meta_losses, physics_meta_losses = [], []

    # ── [Step 1] Vanilla Meta-Training ────────────────────────────────────────
    if use_meta and len(device_z) > 0:
        meta_vanilla = MetaModel(n_in, use_softplus=True).to(device)
        meta_vanilla, vanilla_meta_losses = meta_train_vanilla(meta_vanilla, device_z, device)
    else:
        meta_vanilla = MetaModel(n_in, use_softplus=True).to(device)
        print("  No device data → skipping meta-training (random init)")

    # ── [Step 2] Physics-Guided Meta-Training ─────────────────────────────────
    if use_meta and len(device_phys_z) > 0:
        meta_physics = MetaModel(n_in, use_softplus=True).to(device)
        meta_physics, physics_meta_losses = meta_train_physics(meta_physics, device_z, device_phys_z, N_BINS, device)
    else:
        meta_physics = MetaModel(n_in, use_softplus=True).to(device)
        print("  No device data → skipping physics meta-training (random init)")

    # ── [A] Vanilla Meta + Vanilla FT ─────────────────────────────────────────
    oof_A, test_A, hist_A = cv_run_model(meta_vanilla, Z_train_nn, y_train, Z_test_nn,
        Xtr_phys, Xte_phys, device, tw, use_pinn=False, label="[A] Vanilla Meta + Vanilla FT")
    r2s_A=[r2_score(y_train[:,i],oof_A[:,i]) for i in range(N_OUT)]
    results['A']={'oof':oof_A,'test':test_A,'r2s':r2s_A,'hist':hist_A}
    print(f"  → A avg R²: {np.mean(r2s_A):.4f}")

    # ── [B] Vanilla Meta + PINN FT ───────────────────────────────────────────
    oof_B, test_B, hist_B = cv_run_model(meta_vanilla, Z_train_nn, y_train, Z_test_nn,
        Xtr_phys, Xte_phys, device, tw, use_pinn=True, label="[B] Vanilla Meta + PINN FT")
    r2s_B=[r2_score(y_train[:,i],oof_B[:,i]) for i in range(N_OUT)]
    results['B']={'oof':oof_B,'test':test_B,'r2s':r2s_B,'hist':hist_B}
    print(f"  → B avg R²: {np.mean(r2s_B):.4f}")

    # ── [C] Physics Meta + Vanilla FT ─────────────────────────────────────────
    oof_C, test_C, hist_C = cv_run_model(meta_physics, Z_train_nn, y_train, Z_test_nn,
        Xtr_phys, Xte_phys, device, tw, use_pinn=False, label="[C] Physics Meta + Vanilla FT")
    r2s_C=[r2_score(y_train[:,i],oof_C[:,i]) for i in range(N_OUT)]
    results['C']={'oof':oof_C,'test':test_C,'r2s':r2s_C,'hist':hist_C}
    print(f"  → C avg R²: {np.mean(r2s_C):.4f}")

    # ── [D] Physics Meta + PINN FT ───────────────────────────────────────────
    oof_D, test_D, hist_D = cv_run_model(meta_physics, Z_train_nn, y_train, Z_test_nn,
        Xtr_phys, Xte_phys, device, tw, use_pinn=True, label="[D] Physics Meta + PINN FT")
    r2s_D=[r2_score(y_train[:,i],oof_D[:,i]) for i in range(N_OUT)]
    results['D']={'oof':oof_D,'test':test_D,'r2s':r2s_D,'hist':hist_D}
    print(f"  → D avg R²: {np.mean(r2s_D):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  2×2 FACTORIAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Glucose':>8} {'NaAc':>8} {'MgSO4':>8} {'Average':>8}")
    print(f"  {'-'*62}")
    for key in ['A','B','C','D']:
        r2s=results[key]['r2s']; avg=np.mean(r2s)
        print(f"  {SHORT_LABELS[key]:<30} {r2s[0]:8.4f} {r2s[1]:8.4f} {r2s[2]:8.4f} {avg:8.4f}")
    print(f"\n  Effect of PINN Fine-Tuning:")
    print(f"    With Vanilla Meta:  Δ = {np.mean(r2s_B)-np.mean(r2s_A):+.4f}  (B − A)")
    print(f"    With Physics Meta:  Δ = {np.mean(r2s_D)-np.mean(r2s_C):+.4f}  (D − C)")
    print(f"\n  Effect of Physics Meta-Learning:")
    print(f"    With Vanilla FT:    Δ = {np.mean(r2s_C)-np.mean(r2s_A):+.4f}  (C − A)")
    print(f"    With PINN FT:       Δ = {np.mean(r2s_D)-np.mean(r2s_B):+.4f}  (D − B)")
    print(f"\n  Full Physics Pipeline vs Baseline:")
    print(f"    Δ = {np.mean(r2s_D)-np.mean(r2s_A):+.4f}  (D − A)")

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATE ALL PLOTS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  GENERATING PLOTS\n{'='*70}")
    plot_1_bar_chart(results, OUT_DIR)
    plot_2_training_curves(results, OUT_DIR)
    plot_3_scatter_pred_vs_actual(results, y_train, OUT_DIR)
    plot_4_improvement_heatmap(results, OUT_DIR)
    plot_5_residual_distributions(results, y_train, OUT_DIR)
    plot_6_meta_convergence(vanilla_meta_losses, physics_meta_losses, OUT_DIR)
    plot_7_summary_table(results, OUT_DIR)

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE SUBMISSIONS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  SUBMISSIONS\n{'='*70}")
    # Best NN model
    best_key = max(results, key=lambda k: np.mean(results[k]['r2s']))
    print(f"\n  Best NN model: {SHORT_LABELS[best_key]} (avg R²={np.mean(results[best_key]['r2s']):.4f})")

    save_submission(stacked_test, y_train, "v15_stack", OUT_DIR)
    for key in ['A','B','C','D']:
        save_submission(results[key]['test'], y_train, f"v15_nn_{key}", OUT_DIR)

    # Ensemble: stack + best NN (auto-gated)
    best_nn_avg = np.mean(results[best_key]['r2s'])
    if best_nn_avg > NN_AUTO_GATE_R2:
        print(f"\n  Best NN INCLUDED in ensemble (R²={best_nn_avg:.4f})")
        nn_oof = results[best_key]['oof']; nn_test = results[best_key]['test']
        ens_test=np.zeros_like(stacked_test)
        for i in range(N_OUT):
            r=Ridge(alpha=0.1).fit(nn_oof[:,i:i+1],y_train[:,i])
            nn_oof_cal=r.predict(nn_oof[:,i:i+1]); nn_test_cal=r.predict(nn_test[:,i:i+1])
            bw,br2=1.0,-np.inf
            for ws in np.arange(0,1.01,0.05):
                r2=r2_score(y_train[:,i],ws*stacked_oof[:,i]+(1-ws)*nn_oof_cal.ravel())
                if r2>br2: br2,bw=r2,ws
            ens_test[:,i]=bw*stacked_test[:,i]+(1-bw)*nn_test_cal.ravel()
            print(f"  {ANALYTE_NAMES[i]}: w_stack={bw:.2f}")
    else:
        print(f"\n  Best NN EXCLUDED (R²={best_nn_avg:.4f})")
        ens_test=stacked_test.copy()
    save_submission(ens_test, y_train, "v15_ensemble", OUT_DIR)

    # Domain calibration + blends
    if win_preds is not None and win_preds.shape[0]==stacked_test.shape[0]:
        print(f"\n  --- Domain calibration + blends ---")
        for sn,sp in [('stack',stacked_test),('ensemble',ens_test),('nn_best',results[best_key]['test'])]:
            cal=DomainShiftCalibrator(); cal.fit_from_winning_csv(sp,win_preds)
            print(f"\n  Affine ({sn}):"); cal.report()
            save_submission(cal.transform(sp), y_train, f"v15_{sn}_affine", OUT_DIR)
        for alpha in [0.05,0.10,0.15,0.20,0.30]:
            blended=alpha*ens_test+(1-alpha)*win_preds
            save_submission(blended, y_train, f"v15_ensemble_win_{int(alpha*100):02d}", OUT_DIR)

    print(f"\n{'='*70}\n  v15 COMPLETE — {len(glob.glob(os.path.join(OUT_DIR,'*.png')))} plots + {len(glob.glob(os.path.join(OUT_DIR,'*.csv')))} submissions\n{'='*70}")

if __name__ == "__main__":
    main()
