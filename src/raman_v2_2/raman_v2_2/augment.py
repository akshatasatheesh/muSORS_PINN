"""Invariance augmentations for Raman spectra.

These transforms simulate common Raman "nuisances" that ideally should not change
concentration labels:
- intensity scaling
- baseline drift
- small wavenumber shifts
- optional additive noise

They are meant to be used with an *invariance loss*.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _poly_baseline(D: int, degree: int, rng: np.random.RandomState) -> np.ndarray:
    """Return a zero-mean, unit-variance polynomial baseline over [0,1]."""
    t = np.linspace(0.0, 1.0, D, dtype=np.float64)
    coeffs = rng.normal(0.0, 1.0, size=(degree + 1,)).astype(np.float64)
    baseline = np.zeros((D,), dtype=np.float64)
    for i, c in enumerate(coeffs):
        baseline += c * (t ** i)
    baseline -= baseline.mean()
    baseline /= (baseline.std() + 1e-12)
    return baseline.astype(np.float32)


def augment_spectra(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    rng: np.random.RandomState,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    baseline_amp: float = 0.03,
    baseline_degree: int = 2,
    max_shift: float = 1.5,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Apply random invariance augmentations.

    Parameters
    ----------
    X:
        (N, D) spectra on the shared *wavenumber* grid.
    wavenumbers:
        (D,) grid wavenumbers (must match X columns).
    rng:
        numpy RandomState (seeded by caller).
    scale_range:
        Multiplicative scaling factor range.
    baseline_amp:
        Baseline amplitude as a fraction of spectrum amplitude. If 0, disabled.
    baseline_degree:
        Polynomial degree for baseline.
    max_shift:
        Max absolute wavenumber shift in the same units as `wavenumbers`.
    noise_std:
        Additive Gaussian noise (in the same units as X). If 0, disabled.

    Returns
    -------
    X_aug: np.ndarray
        Augmented spectra (N, D) float32.
    """
    X = np.asarray(X, dtype=np.float32)
    wns = np.asarray(wavenumbers, dtype=np.float32)
    N, D = X.shape

    out = X.copy()

    # 1) intensity scaling
    if scale_range is not None:
        lo, hi = float(scale_range[0]), float(scale_range[1])
        scales = rng.uniform(lo, hi, size=(N, 1)).astype(np.float32)
        out *= scales

    # 2) baseline drift
    if baseline_amp and baseline_amp > 0.0:
        baseline = _poly_baseline(D, int(baseline_degree), rng)[None, :]
        amps = rng.uniform(-float(baseline_amp), float(baseline_amp), size=(N, 1)).astype(np.float32)
        # scale baseline by each spectrum's max abs to keep magnitude reasonable
        spec_amp = np.maximum(np.max(np.abs(out), axis=1, keepdims=True), 1e-6).astype(np.float32)
        out = out + amps * baseline * spec_amp

    # 3) small wavenumber shift
    if max_shift and max_shift > 0.0:
        shifts = rng.uniform(-float(max_shift), float(max_shift), size=(N,)).astype(np.float32)
        out_shifted = np.empty_like(out)
        # Evaluate f(wn - shift) on the original grid (equivalent to shifting peaks)
        for i in range(N):
            delta = float(shifts[i])
            # query points
            q = (wns - delta).astype(np.float32)
            out_shifted[i] = np.interp(q.astype(np.float64), wns.astype(np.float64), out[i].astype(np.float64)).astype(np.float32)
        out = out_shifted

    # 4) optional additive noise
    if noise_std and noise_std > 0.0:
        out = out + rng.normal(0.0, float(noise_std), size=out.shape).astype(np.float32)

    return out.astype(np.float32)
