"""Data loading + preprocessing for DIG 4 Bio Raman Transfer Learning Challenge.

This module supports:
- 8 device training CSVs (with wavenumber columns as header floats + label columns at the end)
- transfer_plate.csv (2 replicates per sample + labels at end)
- 96_samples.csv (2 replicates per sample, no labels)

The goal is to produce a consistent feature vector per sample by interpolating spectra
onto a shared wavenumber grid.

Notes
-----
- Device CSVs include explicit wavenumber headers (floats), so interpolation is exact.
- transfer/test plates do not include wavenumber headers. We assume their native grid is
  linearly spaced from ~65..3350 with 2048 points (a common assumption in public notebooks).
  If you have an authoritative grid, pass it via `--plate_wn_start/--plate_wn_end`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
class RamanDataset:
    X: np.ndarray  # (N, D) float32
    y: Optional[np.ndarray]  # (N, 3) float32 or None
    ids: Optional[np.ndarray]  # (N,) object/int ids, if available
    meta: Dict[str, np.ndarray]  # e.g., device_id
    wavenumbers: np.ndarray  # (D,) float32


def _safe_float_cols(cols: List[str]) -> np.ndarray:
    out = []
    for c in cols:
        try:
            out.append(float(str(c).strip()))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


def _spectrum_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Simple per-spectrum normalization to reduce device power/scale differences."""
    denom = np.maximum(np.max(np.abs(X), axis=1, keepdims=True), eps)
    return (X / denom).astype(np.float32)


def infer_shared_grid(
    data_dir: str,
    device_files: List[str] = DEVICE_FILES,
    wn_min_floor: Optional[float] = None,
    wn_max_ceil: Optional[float] = None,
    step: float = 1.0,
) -> np.ndarray:
    """Infer an intersection wavenumber grid across the device CSVs."""
    mins, maxs = [], []
    for fn in device_files:
        df_head = pd.read_csv(os.path.join(data_dir, fn), nrows=1)
        # spectral columns are everything except the last 5 metadata/label columns
        spec_cols = df_head.columns[:-5]
        wns = _safe_float_cols(list(spec_cols))
        wns = wns[np.isfinite(wns)]
        mins.append(float(np.min(wns)))
        maxs.append(float(np.max(wns)))

    lo = max(mins)
    hi = min(maxs)
    if wn_min_floor is not None:
        lo = max(lo, float(wn_min_floor))
    if wn_max_ceil is not None:
        hi = min(hi, float(wn_max_ceil))

    # inclusive grid
    grid = np.arange(lo, hi + 1e-9, step, dtype=np.float64)
    return grid.astype(np.float32)


def load_device_training(
    data_dir: str,
    grid_wns: np.ndarray,
    device_files: List[str] = DEVICE_FILES,
) -> RamanDataset:
    """Load and interpolate the 8 device training sets onto `grid_wns`.

    Assumptions (true for the provided competition data):
    - Device files have spectral columns as wavenumber floats.
    - Last 5 columns are: glucose, Na_acetate, Mg_SO4, MSM_present, fold_idx.
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_device_id: List[np.ndarray] = []

    for did, fn in enumerate(device_files):
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)

        spec_df = df.iloc[:, :-5]
        label_df = df.iloc[:, -5:-2]  # glucose, Na_acetate, Mg_SO4

        wns_all = _safe_float_cols(list(spec_df.columns))
        mask = np.isfinite(wns_all)
        wns = wns_all[mask]
        spec = spec_df.loc[:, mask].to_numpy(dtype=np.float64)

        # interpolate each row to shared grid
        X_interp = np.vstack([np.interp(grid_wns.astype(np.float64), wns, row) for row in spec]).astype(np.float64)
        X_norm = _spectrum_normalize(X_interp)

        y = label_df.to_numpy(dtype=np.float32)

        all_X.append(X_norm)
        all_y.append(y)
        all_device_id.append(np.full((len(y),), did, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    device_id = np.concatenate(all_device_id, axis=0)

    return RamanDataset(
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        ids=None,
        meta={"device_id": device_id},
        wavenumbers=grid_wns.astype(np.float32),
    )


def _clean_brackets_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_transfer_plate(
    data_dir: str,
    grid_wns: np.ndarray,
    plate_filename: str = "transfer_plate.csv",
    plate_wn_start: float = 65.0,
    plate_wn_end: float = 3350.0,
) -> RamanDataset:
    """Load transfer_plate.csv (2 replicates per sample), interpolate onto grid_wns."""
    path = os.path.join(data_dir, plate_filename)
    df = pd.read_csv(path)

    # Column 0 is sample id with NaNs on replicate rows
    sample_id = df.iloc[:, 0].ffill().astype(str).str.strip().to_numpy()

    # Next 2048 columns are spectral values (often with brackets on first replicate row)
    spec_df = df.iloc[:, 1 : 1 + 2048].copy()
    spec_df = spec_df.apply(_clean_brackets_to_float)
    X_2048 = spec_df.to_numpy(dtype=np.float64)

    # Average replicates (assumes exactly 2 rows per sample)
    if X_2048.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates), got {X_2048.shape[0]}")
    X_2048 = X_2048.reshape(-1, 2, 2048).mean(axis=1)
    ids = sample_id[::2]

    # labels are at the end
    # Note: file uses 'Magnesium Acetate (g/L)' even though the competition label is Magnesium Sulfate.
    y_cols = ["Glucose (g/L)", "Sodium Acetate (g/L)", "Magnesium Acetate (g/L)"]
    for c in y_cols:
        if c not in df.columns:
            raise ValueError(f"Missing label column '{c}' in {plate_filename}.")
    y_df = df.loc[:, y_cols].copy()
    y_df = y_df.ffill()
    y = y_df.to_numpy(dtype=np.float32).reshape(-1, 2, 3).mean(axis=1)

    # plate native grid (assumed)
    plate_wns = np.linspace(float(plate_wn_start), float(plate_wn_end), 2048, dtype=np.float64)

    # select overlap then interpolate
    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    m = (plate_wns >= lo) & (plate_wns <= hi)
    plate_wns_sel = plate_wns[m]
    X_sel = X_2048[:, m]

    X_interp = np.vstack([np.interp(grid_wns.astype(np.float64), plate_wns_sel, row) for row in X_sel]).astype(np.float64)
    X_norm = _spectrum_normalize(X_interp)

    return RamanDataset(
        X=X_norm.astype(np.float32),
        y=y.astype(np.float32),
        ids=ids,
        meta={},
        wavenumbers=grid_wns.astype(np.float32),
    )


def load_test_plate(
    data_dir: str,
    grid_wns: np.ndarray,
    test_filename: str = "96_samples.csv",
    plate_wn_start: float = 65.0,
    plate_wn_end: float = 3350.0,
) -> RamanDataset:
    """Load 96_samples.csv (2 replicates per sample), interpolate onto grid_wns."""
    path = os.path.join(data_dir, test_filename)
    df = pd.read_csv(path, header=None)

    sample_id = df.iloc[:, 0].ffill().astype(str).str.replace("sample", "", regex=False).str.strip().to_numpy()
    spec_df = df.iloc[:, 1:].copy()
    spec_df = spec_df.apply(_clean_brackets_to_float)
    X_2048 = spec_df.to_numpy(dtype=np.float64)

    if X_2048.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates), got {X_2048.shape[0]}")

    X_2048 = X_2048.reshape(-1, 2, 2048).mean(axis=1)
    ids = sample_id[::2]

    plate_wns = np.linspace(float(plate_wn_start), float(plate_wn_end), 2048, dtype=np.float64)
    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    m = (plate_wns >= lo) & (plate_wns <= hi)
    plate_wns_sel = plate_wns[m]
    X_sel = X_2048[:, m]

    X_interp = np.vstack([np.interp(grid_wns.astype(np.float64), plate_wns_sel, row) for row in X_sel]).astype(np.float64)
    X_norm = _spectrum_normalize(X_interp)

    return RamanDataset(
        X=X_norm.astype(np.float32),
        y=None,
        ids=ids,
        meta={},
        wavenumbers=grid_wns.astype(np.float32),
    )
