from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from alignment import AlignmentResult, _overlap, _prepare, _sample_grid, _splines, rmse_for_delta
from data_io import TIME_COL


def _paired_points(df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1, x1, y1 = _prepare(df1, smooth_window=smooth_window)
    t2, x2, y2 = _prepare(df2, smooth_window=smooth_window)
    grid = _sample_grid(result.overlap_start, result.overlap_end, 0.1)
    s1x, s1y = _splines(t1, x1, y1)
    s2x, s2y = _splines(t2, x2, y2)
    p1 = np.column_stack([s1x(grid), s1y(grid)])
    p2 = np.column_stack([s2x(grid - result.delta), s2y(grid - result.delta)])
    return grid, p1, p2


def delta_curve(df1: pd.DataFrame, df2: pd.DataFrame, estimate_bias: bool, smooth_window: int, n: int = 401) -> pd.DataFrame:
    t1, x1, y1 = _prepare(df1, smooth_window=smooth_window)
    t2, x2, y2 = _prepare(df2, smooth_window=smooth_window)
    start_delta = t1[0] - t2[0]
    end_delta = t1[-1] - t2[-1]
    center = float((start_delta + end_delta) / 2.0)
    half_width = abs(start_delta - end_delta) / 2.0 + 120.0
    deltas = np.linspace(center - half_width, center + half_width, n)
    scores = [rmse_for_delta(t1, x1, y1, t2, x2, y2, float(d), estimate_bias) for d in deltas]
    rows = []
    for d, s in zip(deltas, scores):
        start, end = _overlap(t1, t2, float(d))
        rows.append({"Delta(s)": float(d), "objective_rmse(m)": float(s), "overlap_duration(s)": float(max(0.0, end - start))})
    return pd.DataFrame(rows)


def validate_alignment(
    name: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    result: AlignmentResult,
    estimate_bias: bool,
    smooth_window: int,
) -> dict[str, object]:
    curve = delta_curve(df1, df2, estimate_bias, smooth_window)
    finite = curve[np.isfinite(curve["objective_rmse(m)"])].reset_index(drop=True)
    best_idx = int(finite["objective_rmse(m)"].idxmin())
    best = finite.loc[best_idx]
    vals = finite["objective_rmse(m)"].to_numpy()
    local_minima = []
    for i in range(1, len(vals) - 1):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
            local_minima.append((i, vals[i]))
    local_minima = sorted(local_minima, key=lambda x: x[1])
    second = local_minima[1][1] if len(local_minima) > 1 else np.inf
    best_score = float(best["objective_rmse(m)"])
    unique_clear = bool((second - best_score) > max(0.05, 0.05 * best_score) and best_idx not in (0, len(finite) - 1))
    duration = result.overlap_end - result.overlap_start
    points_10hz = int(np.floor(duration / 0.1)) + 1 if duration > 0 else 0
    improvement = (result.rmse_before - result.rmse_after) / result.rmse_before if result.rmse_before > 1e-12 else 0.0
    return {
        "数据": name,
        "Delta(s)": result.delta,
        "overlap_start": result.overlap_start,
        "overlap_end": result.overlap_end,
        "overlap_duration": duration,
        "overlap_points_10hz": points_10hz,
        "RMSE_before": result.rmse_before,
        "RMSE_after": result.rmse_after,
        "improvement_ratio": improvement,
        "objective_best_delta": float(best["Delta(s)"]),
        "objective_best_rmse": best_score,
        "objective_second_local_min_rmse": float(second) if np.isfinite(second) else np.nan,
        "unique_clear_minimum": unique_clear,
        "sufficient_overlap": bool(duration >= 60.0 and points_10hz >= 600),
    }


def residual_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int) -> pd.DataFrame:
    grid, p1, p2 = _paired_points(df1, df2, result, smooth_window)
    residual = p2 - p1
    corrected = residual - np.array([result.bias_x, result.bias_y])
    return pd.DataFrame(
        {
            TIME_COL: grid,
            "residual_x_before": residual[:, 0],
            "residual_y_before": residual[:, 1],
            "residual_x_after": corrected[:, 0],
            "residual_y_after": corrected[:, 1],
            "residual_norm_after": np.sqrt(np.sum(corrected * corrected, axis=1)),
        }
    )


def bootstrap_bias_test(
    name: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    result: AlignmentResult,
    smooth_window: int,
    block_size: int = 30,
    n_boot: int = 500,
    seed: int = 2026,
) -> dict[str, object]:
    residuals = residual_dataframe(df1, df2, result, smooth_window)[["residual_x_before", "residual_y_before"]].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(residuals)
    blocks = [residuals[i : min(i + block_size, n)] for i in range(0, n, block_size)]
    boot = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.vstack([blocks[i] for i in picked])[:n]
        boot.append(np.nanmedian(sample, axis=0))
    boot_arr = np.asarray(boot)
    bx_ci = np.percentile(boot_arr[:, 0], [2.5, 97.5])
    by_ci = np.percentile(boot_arr[:, 1], [2.5, 97.5])
    bias_norm = float(np.hypot(result.bias_x, result.bias_y))
    tau_b = float(max(0.3, 0.2 * result.rmse_after))
    ci_excludes_zero = bool((bx_ci[0] > 0 or bx_ci[1] < 0) and (by_ci[0] > 0 or by_ci[1] < 0))
    exists = bool(bias_norm > tau_b and ci_excludes_zero)
    return {
        "数据": name,
        "bias_x": result.bias_x,
        "bias_y": result.bias_y,
        "bias_norm": bias_norm,
        "RMSE_after": result.rmse_after,
        "tau_b": tau_b,
        "bx_ci_low": float(bx_ci[0]),
        "bx_ci_high": float(bx_ci[1]),
        "by_ci_low": float(by_ci[0]),
        "by_ci_high": float(by_ci[1]),
        "ci_excludes_zero_both_axes": ci_excludes_zero,
        "has_system_bias": exists,
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _fit_translation(p2: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    b = np.nanmedian(p2 - p1, axis=0)
    pred = p2 - b
    return pred, {"bias_x": float(b[0]), "bias_y": float(b[1])}


def _fit_rigid(p2: np.ndarray, p1: np.ndarray, scale: bool) -> tuple[np.ndarray, dict[str, float]]:
    c2 = p2.mean(axis=0)
    c1 = p1.mean(axis=0)
    q2 = p2 - c2
    q1 = p1 - c1
    u, _, vt = np.linalg.svd(q2.T @ q1)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    s = float(np.sum(q1 * (q2 @ r)) / np.sum(q2 * q2)) if scale else 1.0
    pred = s * (p2 @ r) + (c1 - s * (c2 @ r))
    angle = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    trans = c1 - s * (c2 @ r)
    return pred, {"scale": s, "rotation_deg": angle, "tx": float(trans[0]), "ty": float(trans[1])}


def compare_bias_models(name: str, df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int) -> list[dict[str, object]]:
    _, p1, p2 = _paired_points(df1, df2, result, smooth_window)
    baseline = _rmse(p2, p1)
    specs = [
        ("M1_translation", 2, lambda: _fit_translation(p2, p1)),
        ("M2_rotation_translation", 3, lambda: _fit_rigid(p2, p1, False)),
        ("M3_scale_rotation_translation", 4, lambda: _fit_rigid(p2, p1, True)),
    ]
    rows = []
    best_simple_rmse = None
    for model, k, fitter in specs:
        pred, params = fitter()
        rmse = _rmse(pred, p1)
        if model == "M1_translation":
            best_simple_rmse = rmse
        rows.append(
            {
                "数据": name,
                "model": model,
                "param_count": k,
                "RMSE": rmse,
                "improvement_ratio": (baseline - rmse) / baseline if baseline > 1e-12 else 0.0,
                "relative_gain_vs_M1": ((best_simple_rmse - rmse) / best_simple_rmse) if best_simple_rmse else 0.0,
                "params": params,
            }
        )
    best = min(rows, key=lambda r: r["RMSE"])
    keep_model = "M1_translation" if (rows[0]["RMSE"] - best["RMSE"]) / rows[0]["RMSE"] < 0.05 else best["model"]
    for row in rows:
        row["selected_model"] = keep_model
    return rows
