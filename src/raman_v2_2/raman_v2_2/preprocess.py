"""Preprocessing pipeline (v2).

Implements a practical spectroscopy preprocessing stack inspired by
winning DIG4Bio solutions:

  Original -> MSC -> Baseline correction -> Smoothing -> Scaling

Key design goal
--------------
Fit any train-dependent parameters (MSC reference, scalers) on TRAIN only,
and then apply the same transforms to validation/test.

This module is dependency-light: it uses only numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# Optional acceleration: use SciPy's optimized Savitzky-Golay filter if available.
try:
    from scipy.signal import savgol_filter as _scipy_savgol_filter  # type: ignore

    _HAVE_SCIPY = True
except Exception:
    _scipy_savgol_filter = None
    _HAVE_SCIPY = False


def _as_float64(X: np.ndarray) -> np.ndarray:
    if X.dtype == np.float64:
        return X
    return X.astype(np.float64, copy=False)


# -------------------------
# Transform base class
# -------------------------
class Transform:
    """Minimal sklearn-like interface."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Transform":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def state_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Transform":
        raise NotImplementedError


# -------------------------
# 1) MSC
# -------------------------
@dataclass
class MSCTransform(Transform):
    """Multiplicative Scatter Correction.

    Fits a reference spectrum r (default: mean of X_train).

    For each spectrum x, solve x ≈ a + b r, then output (x - a) / b.
    """

    eps: float = 1e-12
    reference_: Optional[np.ndarray] = None  # (D,)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MSCTransform":
        X64 = _as_float64(X)
        self.reference_ = np.mean(X64, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise ValueError("MSCTransform must be fit() before transform().")

        X64 = _as_float64(X)
        r = self.reference_.astype(np.float64, copy=False)

        r_mean = float(r.mean())
        r_center = r - r_mean
        r_var_sum = float(np.sum(r_center ** 2)) + self.eps

        x_mean = X64.mean(axis=1, keepdims=True)
        # covariance with reference (up to a constant factor)
        num = np.sum((X64 - x_mean) * r_center[None, :], axis=1, keepdims=True)
        b = num / r_var_sum
        a = x_mean - b * r_mean

        X_msc = (X64 - a) / (b + self.eps)
        return X_msc.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "eps": float(self.eps),
            "reference_": None if self.reference_ is None else self.reference_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "MSCTransform":
        obj = cls(eps=float(state.get("eps", 1e-12)))
        ref = state.get("reference_", None)
        if ref is not None:
            obj.reference_ = np.array(ref, dtype=np.float32)
        return obj


# -------------------------
# 2) Baseline correction (polynomial detrend)
# -------------------------
@dataclass
class PolyBaselineCorrector(Transform):
    """Subtract a low-degree polynomial baseline.

    This is a fast, dependency-free baseline correction.
    It fits, for each spectrum x, a polynomial baseline over index positions.

    Implementation detail
    ---------------------
    We precompute a least-squares projection matrix so all spectra are corrected
    with fast matrix multiplications.
    """

    degree: int = 3
    # precomputed matrices
    V_: Optional[np.ndarray] = None      # (D, deg+1)
    P_T_: Optional[np.ndarray] = None    # (D, deg+1) = (V^T V)^{-1} V^T transposed

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PolyBaselineCorrector":
        D = int(X.shape[1])
        deg = int(self.degree)
        if deg < 0:
            raise ValueError("degree must be >= 0")

        # Use a scaled x-axis to improve conditioning
        x = np.linspace(-1.0, 1.0, D, dtype=np.float64)
        V = np.vander(x, N=deg + 1, increasing=True)  # (D, deg+1)

        # P = (V^T V)^{-1} V^T, shape (deg+1, D)
        # We'll store P^T to compute coefficients as X @ P^T.
        VT_V = V.T @ V
        inv = np.linalg.pinv(VT_V)
        P = inv @ V.T
        self.V_ = V.astype(np.float64)
        self.P_T_ = P.T.astype(np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.V_ is None or self.P_T_ is None:
            raise ValueError("PolyBaselineCorrector must be fit() before transform().")

        X64 = _as_float64(X)
        # coeffs: (N, deg+1)
        coeffs = X64 @ self.P_T_
        # baseline: (N, D)
        baseline = coeffs @ self.V_.T
        X_corr = X64 - baseline
        return X_corr.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "degree": int(self.degree),
            "V_": None if self.V_ is None else self.V_.astype(np.float32),
            "P_T_": None if self.P_T_ is None else self.P_T_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "PolyBaselineCorrector":
        obj = cls(degree=int(state.get("degree", 3)))
        if state.get("V_", None) is not None:
            obj.V_ = np.array(state["V_"], dtype=np.float32).astype(np.float64)
        if state.get("P_T_", None) is not None:
            obj.P_T_ = np.array(state["P_T_"], dtype=np.float32).astype(np.float64)
        return obj


# -------------------------
# 3) Savitzky-Golay smoothing (deriv=0)
# -------------------------
@dataclass
class SavGolSmoother(Transform):
    """Savitzky-Golay smoothing (dependency-free).

    This implements a standard SavGol filter for smoothing (derivative=0).
    We use reflect padding at the boundaries.

    Parameters
    ----------
    window_length: odd int
    polyorder: int < window_length
    """

    window_length: int = 11
    polyorder: int = 3

    # If SciPy is available we don't need to precompute coefficients;
    # we still keep coeffs_ for the numpy fallback.
    coeffs_: Optional[np.ndarray] = None  # (window_length,)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SavGolSmoother":
        wl = int(self.window_length)
        po = int(self.polyorder)
        if wl % 2 == 0 or wl < 3:
            raise ValueError("window_length must be an odd integer >= 3")
        if po >= wl:
            raise ValueError("polyorder must be < window_length")

        if not _HAVE_SCIPY:
            # Numpy fallback: precompute convolution coefficients
            m = (wl - 1) // 2
            x = np.arange(-m, m + 1, dtype=np.float64)
            A = np.vander(x, N=po + 1, increasing=True)  # (wl, po+1)
            # pseudoinverse: (po+1, wl)
            pinv = np.linalg.pinv(A)
            # smoothing at center is coefficient for c0
            coeffs = pinv[0]  # (wl,)
            self.coeffs_ = coeffs.astype(np.float64)
        else:
            self.coeffs_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X64 = _as_float64(X)
        wl = int(self.window_length)
        po = int(self.polyorder)

        if _HAVE_SCIPY and _scipy_savgol_filter is not None:
            # Fast path
            out = _scipy_savgol_filter(X64, window_length=wl, polyorder=po, deriv=0, axis=1, mode="mirror")
            return out.astype(np.float32)

        # Numpy fallback (slower)
        if self.coeffs_ is None:
            raise ValueError("SavGolSmoother must be fit() before transform() when SciPy is not available.")

        m = (wl - 1) // 2
        coeffs = self.coeffs_

        # reflect pad along feature axis
        Xpad = np.pad(X64, pad_width=((0, 0), (m, m)), mode="reflect")

        # Convolve each row; reverse coeffs for np.convolve
        k = coeffs[::-1]
        out = np.empty_like(X64)
        for i in range(X64.shape[0]):
            out[i] = np.convolve(Xpad[i], k, mode="valid")
        return out.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "window_length": int(self.window_length),
            "polyorder": int(self.polyorder),
            "coeffs_": None if self.coeffs_ is None else self.coeffs_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SavGolSmoother":
        obj = cls(window_length=int(state.get("window_length", 11)), polyorder=int(state.get("polyorder", 3)))
        if state.get("coeffs_", None) is not None:
            obj.coeffs_ = np.array(state["coeffs_"], dtype=np.float32).astype(np.float64)
        return obj


# -------------------------
# 4) Scaling
# -------------------------
@dataclass
class StandardScaler1D(Transform):
    """Feature-wise standardization fit on train, applied to all splits."""

    eps: float = 1e-12
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScaler1D":
        X64 = _as_float64(X)
        self.mean_ = X64.mean(axis=0)
        self.std_ = X64.std(axis=0)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler1D must be fit() before transform().")
        X64 = _as_float64(X)
        return ((X64 - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "eps": float(self.eps),
            "mean_": None if self.mean_ is None else self.mean_.astype(np.float32),
            "std_": None if self.std_ is None else self.std_.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "StandardScaler1D":
        obj = cls(eps=float(state.get("eps", 1e-12)))
        if state.get("mean_", None) is not None:
            obj.mean_ = np.array(state["mean_"], dtype=np.float32).astype(np.float64)
        if state.get("std_", None) is not None:
            obj.std_ = np.array(state["std_"], dtype=np.float32).astype(np.float64)
        return obj


@dataclass
class GlobalMaxScaler(Transform):
    """Scale by a single global max abs value computed on TRAIN."""

    eps: float = 1e-12
    max_abs_: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GlobalMaxScaler":
        X64 = _as_float64(X)
        self.max_abs_ = float(np.max(np.abs(X64)))
        if self.max_abs_ < self.eps:
            self.max_abs_ = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.max_abs_ is None:
            raise ValueError("GlobalMaxScaler must be fit() before transform().")
        X64 = _as_float64(X)
        return (X64 / (self.max_abs_ + self.eps)).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "eps": float(self.eps),
            "max_abs_": None if self.max_abs_ is None else float(self.max_abs_),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "GlobalMaxScaler":
        obj = cls(eps=float(state.get("eps", 1e-12)))
        obj.max_abs_ = state.get("max_abs_", None)
        if obj.max_abs_ is not None:
            obj.max_abs_ = float(obj.max_abs_)
        return obj


# -------------------------
# Pipeline
# -------------------------
@dataclass
class PreprocessPipeline:
    steps: List[Tuple[str, Transform]]

    def fit(self, X: np.ndarray) -> "PreprocessPipeline":
        cur = X
        for _, t in self.steps:
            cur = t.fit_transform(cur)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cur = X
        for _, t in self.steps:
            cur = t.transform(cur)
        return cur

    def state_dict(self) -> Dict[str, Any]:
        return {
            "steps": [(name, t.state_dict()) for name, t in self.steps],
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "PreprocessPipeline":
        steps: List[Tuple[str, Transform]] = []
        for name, tstate in state.get("steps", []):
            ttype = tstate.get("type")
            if ttype == "MSCTransform":
                t = MSCTransform.from_state_dict(tstate)
            elif ttype == "PolyBaselineCorrector":
                t = PolyBaselineCorrector.from_state_dict(tstate)
            elif ttype == "SavGolSmoother":
                t = SavGolSmoother.from_state_dict(tstate)
            elif ttype == "StandardScaler1D":
                t = StandardScaler1D.from_state_dict(tstate)
            elif ttype == "GlobalMaxScaler":
                t = GlobalMaxScaler.from_state_dict(tstate)
            else:
                raise ValueError(f"Unknown transform type: {ttype}")
            steps.append((name, t))
        return cls(steps=steps)


ScalingKind = Literal["standard", "global_max", "none"]
BaselineKind = Literal["poly", "none"]


def build_default_pipeline(
    *,
    use_msc: bool = True,
    baseline: BaselineKind = "poly",
    baseline_degree: int = 3,
    use_savgol: bool = True,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    scaling: ScalingKind = "standard",
) -> PreprocessPipeline:
    steps: List[Tuple[str, Transform]] = []

    if use_msc:
        steps.append(("msc", MSCTransform()))

    if baseline == "poly":
        steps.append(("baseline_poly", PolyBaselineCorrector(degree=int(baseline_degree))))

    if use_savgol:
        steps.append(("savgol", SavGolSmoother(window_length=int(savgol_window), polyorder=int(savgol_polyorder))))

    if scaling == "standard":
        steps.append(("scale_standard", StandardScaler1D()))
    elif scaling == "global_max":
        steps.append(("scale_global_max", GlobalMaxScaler()))
    elif scaling == "none":
        pass
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    return PreprocessPipeline(steps=steps)
