from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar

from data_io import TIME_COL, X_COL, Y_COL


@dataclass
class AlignmentResult:
    delta: float
    bias_x: float
    bias_y: float
    rmse_before: float
    rmse_after: float
    overlap_start: float
    overlap_end: float


def _prepare(df: pd.DataFrame, smooth_window: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = df[[TIME_COL, X_COL, Y_COL]].dropna().drop_duplicates(TIME_COL).sort_values(TIME_COL)
    t = clean[TIME_COL].to_numpy(dtype=float)
    x = clean[X_COL].to_numpy(dtype=float)
    y = clean[Y_COL].to_numpy(dtype=float)
    if smooth_window > 1:
        x = uniform_filter1d(x, size=smooth_window, mode="nearest")
        y = uniform_filter1d(y, size=smooth_window, mode="nearest")
    return t, x, y


def _splines(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[CubicSpline, CubicSpline]:
    return CubicSpline(t, x), CubicSpline(t, y)


def _overlap(t1: np.ndarray, t2: np.ndarray, delta: float) -> tuple[float, float]:
    return max(t1[0], t2[0] + delta), min(t1[-1], t2[-1] + delta)


def _sample_grid(start: float, end: float, step: float = 0.1) -> np.ndarray:
    if end <= start:
        return np.array([])
    n = int(np.floor((end - start) / step)) + 1
    return start + np.arange(n) * step


def rmse_for_delta(
    t1: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    delta: float,
    estimate_bias: bool,
) -> float:
    start, end = _overlap(t1, t2, delta)
    grid = _sample_grid(start, end, 0.2)
    if len(grid) < 10:
        return float("inf")
    s1x, s1y = _splines(t1, x1, y1)
    s2x, s2y = _splines(t2, x2, y2)
    dx = s2x(grid - delta) - s1x(grid)
    dy = s2y(grid - delta) - s1y(grid)
    if estimate_bias:
        dx = dx - np.nanmedian(dx)
        dy = dy - np.nanmedian(dy)
    return float(np.sqrt(np.nanmean(dx * dx + dy * dy)))


def align_pair(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    estimate_bias: bool,
    smooth_window: int = 1,
    search_margin: float = 5.0,
) -> AlignmentResult:
    t1, x1, y1 = _prepare(df1, smooth_window=smooth_window)
    t2, x2, y2 = _prepare(df2, smooth_window=smooth_window)

    start_delta = t1[0] - t2[0]
    end_delta = t1[-1] - t2[-1]
    center = float((start_delta + end_delta) / 2.0)
    half_width = abs(start_delta - end_delta) / 2.0 + max(search_margin, 120.0)
    lo, hi = center - half_width, center + half_width

    objective = lambda d: rmse_for_delta(t1, x1, y1, t2, x2, y2, float(d), estimate_bias)
    coarse = np.linspace(lo, hi, 201)
    coarse_scores = np.array([objective(d) for d in coarse])
    best = float(coarse[int(np.nanargmin(coarse_scores))])
    bracket_lo = max(lo, best - max(2.0, half_width / 20))
    bracket_hi = min(hi, best + max(2.0, half_width / 20))
    opt = minimize_scalar(objective, bounds=(bracket_lo, bracket_hi), method="bounded", options={"xatol": 1e-4})
    delta = float(opt.x)

    start, end = _overlap(t1, t2, delta)
    grid = _sample_grid(start, end, 0.1)
    s1x, s1y = _splines(t1, x1, y1)
    s2x, s2y = _splines(t2, x2, y2)
    diff_x = s2x(grid - delta) - s1x(grid)
    diff_y = s2y(grid - delta) - s1y(grid)
    bias_x = float(np.nanmedian(diff_x)) if estimate_bias else 0.0
    bias_y = float(np.nanmedian(diff_y)) if estimate_bias else 0.0
    rmse_before = float(np.sqrt(np.nanmean(diff_x * diff_x + diff_y * diff_y)))
    adj_x = diff_x - bias_x
    adj_y = diff_y - bias_y
    rmse_after = float(np.sqrt(np.nanmean(adj_x * adj_x + adj_y * adj_y)))

    return AlignmentResult(delta, bias_x, bias_y, rmse_before, rmse_after, float(start), float(end))


def resample_aligned(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    result: AlignmentResult,
    smooth_window: int = 1,
) -> pd.DataFrame:
    t1, x1, y1 = _prepare(df1, smooth_window=smooth_window)
    t2, x2, y2 = _prepare(df2, smooth_window=smooth_window)
    grid = _sample_grid(result.overlap_start, result.overlap_end, 0.1)
    s1x, s1y = _splines(t1, x1, y1)
    s2x, s2y = _splines(t2, x2, y2)
    x1g, y1g = s1x(grid), s1y(grid)
    x2g = s2x(grid - result.delta) - result.bias_x
    y2g = s2y(grid - result.delta) - result.bias_y
    return pd.DataFrame(
        {
            TIME_COL: np.round(grid, 3),
            "x1_aligned": x1g,
            "y1_aligned": y1g,
            "x2_aligned_corrected": x2g,
            "y2_aligned_corrected": y2g,
            X_COL: (x1g + x2g) / 2.0,
            Y_COL: (y1g + y2g) / 2.0,
        }
    )
