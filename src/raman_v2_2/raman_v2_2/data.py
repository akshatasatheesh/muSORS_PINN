"""Data loading + interpolation for DIG 4 Bio Raman Transfer Learning Challenge (v2).

Differences from v1
-------------------
- This module performs ONLY:
  - CSV parsing & cleaning
  - replicate averaging
  - interpolation to a shared wavenumber grid

It intentionally does NOT perform normalization or baseline removal.
Those are handled by `raman_v2.preprocess.PreprocessPipeline`, which is fit on
training data and applied consistently to val/test.
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
    target_names: Optional[List[str]] = None  # length 3 (matches sample_submission)


def _safe_float_cols(cols: List[str]) -> np.ndarray:
    out: List[float] = []
    for c in cols:
        try:
            out.append(float(str(c).strip()))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


def _clean_brackets_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("[", "", regex=False).str.replace("]", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def read_sample_submission_targets(data_dir: str) -> Tuple[str, List[str]]:
    """Return (id_col, target_cols) from sample_submission.csv."""
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    if sub.shape[1] < 2:
        raise ValueError("sample_submission.csv must have at least an ID column and one target column")
    id_col = str(sub.columns[0])
    target_cols = [str(c) for c in sub.columns[1:]]
    if len(target_cols) != 3:
        raise ValueError(f"Expected 3 target columns in sample_submission.csv, got {len(target_cols)}")
    return id_col, target_cols


def infer_shared_grid(
    data_dir: str,
    device_files: List[str] = DEVICE_FILES,
    wn_min_floor: Optional[float] = None,
    wn_max_ceil: Optional[float] = None,
    step: float = 1.0,
) -> np.ndarray:
    """Infer an intersection wavenumber grid across the device CSVs."""
    mins: List[float] = []
    maxs: List[float] = []
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

    grid = np.arange(lo, hi + 1e-9, step, dtype=np.float64)
    return grid.astype(np.float32)


def _match_label_columns(df_cols: List[str], target_names: List[str]) -> List[str]:
    """Try to map dataset-specific label columns to sample_submission target names.

    We do keyword-based matching so we can map:
      - glucose -> Glucose
      - Na_acetate -> Sodium Acetate
      - Mg_SO4 or Magnesium Acetate (g/L) -> Magnesium Sulfate

    Returns list of column names in df in the same order as target_names.
    """

    def norm(s: str) -> str:
        s = s.lower().strip()
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = s.replace("(", " ").replace(")", " ")
        s = s.replace("/", " ")
        s = " ".join(s.split())
        return s

    cand = [str(c) for c in df_cols]
    cand_norm = {c: norm(c) for c in cand}

    out: List[str] = []
    used = set()

    for t in target_names:
        tn = norm(t)
        chosen: Optional[str] = None

        if "glucose" in tn:
            for c in cand:
                if c in used:
                    continue
                if "glucose" in cand_norm[c]:
                    chosen = c
                    break

        elif "sodium" in tn and "acetate" in tn:
            for c in cand:
                if c in used:
                    continue
                cn = cand_norm[c]
                if ("sodium" in cn or cn.startswith("na") or " na" in cn) and "acetate" in cn:
                    chosen = c
                    break

        elif "magnesium" in tn:
            for c in cand:
                if c in used:
                    continue
                cn = cand_norm[c]
                if ("magnesium" in cn or cn.startswith("mg") or " mg" in cn):
                    chosen = c
                    break

        if chosen is None:
            # fallback: try partial token overlap
            tokens = [tok for tok in tn.split() if tok not in {"g", "l", "mg", "dl"}]
            best = None
            best_score = -1
            for c in cand:
                if c in used:
                    continue
                score = sum(tok in cand_norm[c] for tok in tokens)
                if score > best_score:
                    best_score = score
                    best = c
            chosen = best

        if chosen is None:
            raise ValueError(f"Could not map target '{t}' to any label column among: {cand}")

        used.add(chosen)
        out.append(chosen)

    return out


def load_device_training(
    data_dir: str,
    grid_wns: np.ndarray,
    device_files: List[str] = DEVICE_FILES,
) -> RamanDataset:
    """Load the 8 device training sets, interpolate onto `grid_wns`.

    Assumption (competition data): last 5 columns are:
      glucose, Na_acetate, Mg_SO4, MSM_present, fold_idx
    """

    _, target_names = read_sample_submission_targets(data_dir)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_device_id: List[np.ndarray] = []

    for did, fn in enumerate(device_files):
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)

        spec_df = df.iloc[:, :-5]
        label_candidates = list(df.columns[-5:-2])
        label_cols = _match_label_columns(label_candidates, target_names)
        y = df.loc[:, label_cols].to_numpy(dtype=np.float32)

        wns_all = _safe_float_cols(list(spec_df.columns))
        mask = np.isfinite(wns_all)
        wns = wns_all[mask]
        spec = spec_df.loc[:, mask].to_numpy(dtype=np.float64)

        # interpolate each spectrum to shared grid
        X_interp = np.vstack([np.interp(grid_wns.astype(np.float64), wns, row) for row in spec]).astype(np.float32)

        all_X.append(X_interp)
        all_y.append(y)
        all_device_id.append(np.full((len(y),), did, dtype=np.int64))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    device_id = np.concatenate(all_device_id, axis=0)

    return RamanDataset(
        X=X,
        y=y,
        ids=None,
        meta={"device_id": device_id},
        wavenumbers=grid_wns.astype(np.float32),
        target_names=target_names,
    )


def _plate_to_grid(
    X_2048: np.ndarray,
    grid_wns: np.ndarray,
    *,
    plate_wn_start: float,
    plate_wn_end: float,
) -> np.ndarray:
    """Interpolate 2048-point plate spectra onto grid_wns."""
    plate_wns = np.linspace(float(plate_wn_start), float(plate_wn_end), 2048, dtype=np.float64)

    lo, hi = float(grid_wns.min()), float(grid_wns.max())
    m = (plate_wns >= lo) & (plate_wns <= hi)

    plate_wns_sel = plate_wns[m]
    X_sel = X_2048[:, m].astype(np.float64)

    X_interp = np.vstack(
        [np.interp(grid_wns.astype(np.float64), plate_wns_sel, row) for row in X_sel]
    ).astype(np.float32)
    return X_interp


def load_transfer_plate(
    data_dir: str,
    grid_wns: np.ndarray,
    plate_filename: str = "transfer_plate.csv",
    plate_wn_start: float = 65.0,
    plate_wn_end: float = 3350.0,
) -> RamanDataset:
    """Load transfer_plate.csv (2 replicates per sample), interpolate onto grid_wns."""

    _, target_names = read_sample_submission_targets(data_dir)

    path = os.path.join(data_dir, plate_filename)
    df = pd.read_csv(path)

    # Column 0 is sample id with NaNs on replicate rows
    sample_id = df.iloc[:, 0].ffill().astype(str).str.strip().to_numpy()

    # Next 2048 columns are spectral values
    spec_df = df.iloc[:, 1 : 1 + 2048].copy()
    spec_df = spec_df.apply(_clean_brackets_to_float)
    X_2048 = spec_df.to_numpy(dtype=np.float64)

    if X_2048.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates), got {X_2048.shape[0]}")

    # average replicates
    X_2048 = X_2048.reshape(-1, 2, 2048).mean(axis=1)
    ids = sample_id[::2]

    # labels: find best matching columns
    # Candidate label cols are the last ~10 columns (safer), but we search entire df.
    label_cols = _match_label_columns(list(df.columns), target_names)
    y_df = df.loc[:, label_cols].copy().ffill()
    y = y_df.to_numpy(dtype=np.float32)
    y = y.reshape(-1, 2, 3).mean(axis=1)

    X_interp = _plate_to_grid(X_2048, grid_wns, plate_wn_start=plate_wn_start, plate_wn_end=plate_wn_end)

    return RamanDataset(
        X=X_interp.astype(np.float32),
        y=y.astype(np.float32),
        ids=ids,
        meta={},
        wavenumbers=grid_wns.astype(np.float32),
        target_names=target_names,
    )


def load_test_plate(
    data_dir: str,
    grid_wns: np.ndarray,
    test_filename: str = "96_samples.csv",
    plate_wn_start: float = 65.0,
    plate_wn_end: float = 3350.0,
) -> RamanDataset:
    """Load 96_samples.csv (2 replicates per sample), interpolate onto grid_wns."""

    _, target_names = read_sample_submission_targets(data_dir)

    path = os.path.join(data_dir, test_filename)
    df = pd.read_csv(path, header=None)

    sample_id = (
        df.iloc[:, 0]
        .ffill()
        .astype(str)
        .str.replace("sample", "", regex=False)
        .str.strip()
        .to_numpy()
    )

    spec_df = df.iloc[:, 1:].copy()
    spec_df = spec_df.apply(_clean_brackets_to_float)
    X_2048 = spec_df.to_numpy(dtype=np.float64)

    if X_2048.shape[0] % 2 != 0:
        raise ValueError(f"Expected even number of rows (2 replicates), got {X_2048.shape[0]}")

    X_2048 = X_2048.reshape(-1, 2, 2048).mean(axis=1)
    ids = sample_id[::2]

    X_interp = _plate_to_grid(X_2048, grid_wns, plate_wn_start=plate_wn_start, plate_wn_end=plate_wn_end)

    return RamanDataset(
        X=X_interp.astype(np.float32),
        y=None,
        ids=ids,
        meta={},
        wavenumbers=grid_wns.astype(np.float32),
        target_names=target_names,
    )
