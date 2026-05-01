from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.signal import savgol_filter

from alignment import AlignmentResult, _overlap, _prepare, _sample_grid, _splines, align_pair, resample_aligned
from data_io import TIME_COL, X_COL, Y_COL
from kinematics import add_kinematics
from task_opt import generate_candidates, optimize_with_verification


def rmse_vec(res: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(res * res, axis=1))))


def mae_vec(res: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.sum(res * res, axis=1))))


def huber_vec(res: np.ndarray, delta: float = 1.0) -> float:
    norm = np.sqrt(np.sum(res * res, axis=1))
    loss = np.where(norm <= delta, 0.5 * norm * norm, delta * (norm - 0.5 * delta))
    return float(np.mean(loss))


def paired_points_at_delta(df1: pd.DataFrame, df2: pd.DataFrame, delta: float, smooth_window: int = 1, step: float = 0.1):
    t1, x1, y1 = _prepare(df1, smooth_window=smooth_window)
    t2, x2, y2 = _prepare(df2, smooth_window=smooth_window)
    start, end = _overlap(t1, t2, delta)
    grid = _sample_grid(start, end, step)
    if len(grid) < 5:
        return grid, np.empty((0, 2)), np.empty((0, 2))
    s1x, s1y = _splines(t1, x1, y1)
    s2x, s2y = _splines(t2, x2, y2)
    p1 = np.column_stack([s1x(grid), s1y(grid)])
    p2 = np.column_stack([s2x(grid - delta), s2y(grid - delta)])
    return grid, p1, p2


def delta_objective_diagnostics(name: str, df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int) -> tuple[pd.DataFrame, dict[str, object]]:
    deltas = np.linspace(result.delta - 5.0, result.delta + 5.0, 201)
    rows = []
    for delta in deltas:
        grid, p1, p2 = paired_points_at_delta(df1, df2, float(delta), smooth_window=smooth_window)
        if len(grid) < 5:
            continue
        raw = p2 - p1
        b = np.nanmedian(raw, axis=0)
        res = raw - b
        rows.append(
            {
                "dataset": name,
                "Delta": float(delta),
                "bx": float(b[0]),
                "by": float(b[1]),
                "RMSE": rmse_vec(res),
                "MAE": mae_vec(res),
                "Huber": huber_vec(res),
                "points": int(len(grid)),
            }
        )
    df = pd.DataFrame(rows)
    best = df.loc[df["RMSE"].idxmin()]
    sorted_df = df.sort_values("RMSE").reset_index(drop=True)
    def interp_increase(offset: float) -> float:
        val = np.interp(float(best["Delta"] + offset), df["Delta"], df["RMSE"])
        return float(val - best["RMSE"])
    idx = int(df["RMSE"].idxmin())
    if 0 < idx < len(df) - 1:
        h = df["Delta"].iloc[idx + 1] - df["Delta"].iloc[idx]
        curvature = float((df["RMSE"].iloc[idx + 1] - 2 * df["RMSE"].iloc[idx] + df["RMSE"].iloc[idx - 1]) / (h * h))
    else:
        curvature = np.nan
    summary = {
        "dataset": name,
        "best_delta": float(best["Delta"]),
        "best_RMSE": float(best["RMSE"]),
        "second_best_delta_gap": float(abs(sorted_df.loc[1, "Delta"] - sorted_df.loc[0, "Delta"])) if len(sorted_df) > 1 else np.nan,
        "curvature_around_optimum": curvature,
        "rmse_increase_delta_minus_1s": interp_increase(-1.0),
        "rmse_increase_delta_minus_0.5s": interp_increase(-0.5),
        "rmse_increase_delta_plus_0.5s": interp_increase(0.5),
        "rmse_increase_delta_plus_1s": interp_increase(1.0),
    }
    return df, summary


def _fit_bias_model(tn: np.ndarray, res: np.ndarray, model: str, k_segments: int = 5):
    if model == "fixed":
        b = np.nanmedian(res, axis=0)
        pred = np.repeat(b[None, :], len(res), axis=0)
        return pred, 2
    if model == "linear":
        x = np.column_stack([np.ones(len(tn)), tn])
        coef_x = np.linalg.lstsq(x, res[:, 0], rcond=None)[0]
        coef_y = np.linalg.lstsq(x, res[:, 1], rcond=None)[0]
        pred = np.column_stack([x @ coef_x, x @ coef_y])
        return pred, 4
    if model.startswith("piecewise"):
        bins = np.minimum((tn * k_segments).astype(int), k_segments - 1)
        pred = np.zeros_like(res)
        for k in range(k_segments):
            mask = bins == k
            if np.any(mask):
                pred[mask] = np.nanmedian(res[mask], axis=0)
        return pred, 2 * k_segments
    raise ValueError(model)


def bias_model_comparison(name: str, df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    grid, p1, p2 = paired_points_at_delta(df1, df2, result.delta, smooth_window=smooth_window)
    res = p2 - p1
    tn = (grid - grid.min()) / (grid.max() - grid.min())
    valid_mask = ((np.arange(len(grid)) // 50) % 5) == 4
    train_mask = ~valid_mask
    rows = []
    for model, kseg in [("fixed", 1), ("linear", 1), ("piecewise_3", 3), ("piecewise_5", 5), ("piecewise_10", 10)]:
        pred_train, pcount = _fit_bias_model(tn[train_mask], res[train_mask], "piecewise" if model.startswith("piecewise") else model, kseg)
        if model == "fixed":
            b = np.nanmedian(res[train_mask], axis=0)
            pred_all = np.repeat(b[None, :], len(res), axis=0)
        elif model == "linear":
            x_train = np.column_stack([np.ones(train_mask.sum()), tn[train_mask]])
            coef_x = np.linalg.lstsq(x_train, res[train_mask, 0], rcond=None)[0]
            coef_y = np.linalg.lstsq(x_train, res[train_mask, 1], rcond=None)[0]
            x_all = np.column_stack([np.ones(len(tn)), tn])
            pred_all = np.column_stack([x_all @ coef_x, x_all @ coef_y])
        else:
            bins_train = np.minimum((tn[train_mask] * kseg).astype(int), kseg - 1)
            bins_all = np.minimum((tn * kseg).astype(int), kseg - 1)
            pred_all = np.zeros_like(res)
            global_b = np.nanmedian(res[train_mask], axis=0)
            for k in range(kseg):
                mask_train_k = bins_train == k
                b = np.nanmedian(res[train_mask][mask_train_k], axis=0) if np.any(mask_train_k) else global_b
                pred_all[bins_all == k] = b
        err_train = res[train_mask] - pred_all[train_mask]
        err_valid = res[valid_mask] - pred_all[valid_mask]
        n = len(err_valid)
        bic = n * np.log(max(rmse_vec(err_valid) ** 2, 1e-12)) + pcount * np.log(max(n, 2))
        rows.append(
            {
                "dataset": name,
                "model": model,
                "param_count": pcount,
                "train_RMSE": rmse_vec(err_train),
                "valid_RMSE": rmse_vec(err_valid),
                "train_MAE": mae_vec(err_train),
                "valid_MAE": mae_vec(err_valid),
                "valid_Huber": huber_vec(err_valid),
                "BIC_valid": float(bic),
            }
        )
    comp = pd.DataFrame(rows)
    fixed_valid = float(comp.loc[comp["model"] == "fixed", "valid_RMSE"].iloc[0])
    best = comp.loc[comp["valid_RMSE"].idxmin()]
    decision = "fixed" if (fixed_valid - float(best["valid_RMSE"])) / fixed_valid < 0.05 else str(best["model"])
    slide_rows = []
    win = max(50, len(res) // 20)
    step = max(10, win // 3)
    for start in range(0, len(res) - win + 1, step):
        part = res[start : start + win]
        b = np.nanmedian(part, axis=0)
        slide_rows.append({"dataset": name, "time_center": float(np.mean(grid[start : start + win])), "bx": float(b[0]), "by": float(b[1]), "bias_norm": float(np.hypot(*b)), "window_points": win})
    return comp, pd.DataFrame(slide_rows), decision


def residual_acf(x: np.ndarray, max_lag: int = 200) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    denom = np.nansum(x * x)
    rows = []
    for lag in range(max_lag + 1):
        if lag == 0:
            rho = 1.0
        else:
            rho = float(np.nansum(x[:-lag] * x[lag:]) / denom) if denom > 0 else np.nan
        rows.append({"lag": lag, "seconds": lag * 0.1, "acf": rho})
    return pd.DataFrame(rows)


def block_bootstrap_ci(res: np.ndarray, block_length: int, n_boot: int = 1000, seed: int = 2026) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(res)
    blocks = [res[i : min(i + block_length, n)] for i in range(0, n, block_length)]
    vals = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.vstack([blocks[i] for i in picked])[:n]
        b = np.nanmedian(sample, axis=0)
        vals.append([b[0], b[1], np.hypot(b[0], b[1])])
    arr = np.asarray(vals)
    ci = np.percentile(arr, [2.5, 50, 97.5], axis=0)
    return {
        "block_length": block_length,
        "block_seconds": block_length * 0.1,
        "bx_ci_low": float(ci[0, 0]),
        "bx_median": float(ci[1, 0]),
        "bx_ci_high": float(ci[2, 0]),
        "by_ci_low": float(ci[0, 1]),
        "by_median": float(ci[1, 1]),
        "by_ci_high": float(ci[2, 1]),
        "bias_norm_ci_low": float(ci[0, 2]),
        "bias_norm_median": float(ci[1, 2]),
        "bias_norm_ci_high": float(ci[2, 2]),
    }


def bootstrap_analysis_attachment3(df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, smooth_window: int, n_boot: int = 1000):
    grid, p1, p2 = paired_points_at_delta(df1, df2, result.delta, smooth_window=smooth_window)
    res = p2 - p1
    norm = np.sqrt(np.sum((res - np.nanmedian(res, axis=0)) ** 2, axis=1))
    acf = residual_acf(norm, max_lag=200)
    auto_candidates = acf[(acf["lag"] > 0) & (acf["acf"].abs() < 0.1)]
    auto_len = int(max(5, auto_candidates["lag"].iloc[0] if not auto_candidates.empty else 20))
    blocks = sorted(set([5, 10, 20, auto_len]))
    rows = []
    bias_norm = float(np.hypot(result.bias_x, result.bias_y))
    tau_b = float(max(0.3, 0.2 * result.rmse_after))
    for bl in blocks:
        row = block_bootstrap_ci(res, bl, n_boot=n_boot)
        row.update(
            {
                "bias_norm": bias_norm,
                "tau_b": tau_b,
                "bias_norm_gt_tau": bool(bias_norm > tau_b),
                "bx_contains_zero": bool(row["bx_ci_low"] <= 0 <= row["bx_ci_high"]),
                "by_contains_zero": bool(row["by_ci_low"] <= 0 <= row["by_ci_high"]),
                "significant_fixed_bias": bool(bias_norm > tau_b and row["bias_norm_ci_low"] > tau_b and not (row["bx_ci_low"] <= 0 <= row["bx_ci_high"]) and not (row["by_ci_low"] <= 0 <= row["by_ci_high"])),
            }
        )
        rows.append(row)
    return acf, pd.DataFrame(rows), auto_len


def robust_alignment_objectives(name: str, df1: pd.DataFrame, df2: pd.DataFrame, base_delta: float, smooth_window: int):
    metrics: dict[str, Callable[[np.ndarray], float]] = {
        "RMSE": rmse_vec,
        "MAE": mae_vec,
        "Huber": huber_vec,
        "trimmed_RMSE": lambda r: float(np.sqrt(np.mean(np.sort(np.sum(r * r, axis=1))[: max(1, int(0.95 * len(r)))]))),
    }
    rows = []
    deltas = np.linspace(base_delta - 2.0, base_delta + 2.0, 161)
    for metric_name, func in metrics.items():
        best = None
        for d in deltas:
            _, p1, p2 = paired_points_at_delta(df1, df2, float(d), smooth_window=smooth_window)
            raw = p2 - p1
            b = np.nanmedian(raw, axis=0)
            res = raw - b
            val = func(res)
            if best is None or val < best[0]:
                best = (val, float(d), b, res)
        val, d, b, res = best
        rows.append({"dataset": name, "objective": metric_name, "Delta": d, "bx": float(b[0]), "by": float(b[1]), "RMSE_after": rmse_vec(res), "MAE_after": mae_vec(res), "Huber_after": huber_vec(res)})
    return pd.DataFrame(rows)


def engineering_threshold_sensitivity(result: AlignmentResult, bootstrap_row: pd.Series) -> pd.DataFrame:
    rows = []
    bias_norm = float(np.hypot(result.bias_x, result.bias_y))
    bootstrap_significant = bool(bootstrap_row["bias_norm_ci_low"] > 0 and not bootstrap_row["bx_contains_zero"] and not bootstrap_row["by_contains_zero"])
    for tau0 in [0.2, 0.3, 0.4, 0.5]:
        for alpha in [0.1, 0.2, 0.3]:
            tau = max(tau0, alpha * result.rmse_after)
            rows.append({"tau0": tau0, "alpha": alpha, "bias_norm": bias_norm, "tau_b": tau, "bias_norm_gt_tau": bool(bias_norm > tau), "bootstrap_significant": bootstrap_significant, "final_decision": bool(bias_norm > tau and bootstrap_significant)})
    return pd.DataFrame(rows)


def estimate_noise_variance_from_smoothing(fused: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, xcol, ycol in [("source1", "x1_aligned", "y1_aligned"), ("source2", "x2_aligned_corrected", "y2_aligned_corrected")]:
        x = fused[xcol].to_numpy(dtype=float)
        y = fused[ycol].to_numpy(dtype=float)
        win = min(81, len(fused) - 1 if len(fused) % 2 == 0 else len(fused))
        if win % 2 == 0:
            win -= 1
        xs = savgol_filter(x, win, 3, mode="interp")
        ys = savgol_filter(y, win, 3, mode="interp")
        var = float(np.var(x - xs) + np.var(y - ys))
        rows.append({"source": label, "variance_sum_xy": var})
    df = pd.DataFrame(rows)
    inv = 1 / np.maximum(df["variance_sum_xy"].to_numpy(), 1e-12)
    weights = inv / inv.sum()
    df["inverse_variance_weight"] = weights
    return df


def fuse_with_weight(fused_aligned: pd.DataFrame, w1: float) -> pd.DataFrame:
    out = fused_aligned.copy()
    w2 = 1.0 - w1
    out[X_COL] = w1 * out["x1_aligned"] + w2 * out["x2_aligned_corrected"]
    out[Y_COL] = w1 * out["y1_aligned"] + w2 * out["y2_aligned_corrected"]
    return out


def task_metrics_for_traj(traj: pd.DataFrame, targets: pd.DataFrame, max_tasks: int | None = None):
    cand = generate_candidates(traj, targets)
    selected, verification = optimize_with_verification(cand, traj, targets, max_tasks=max_tasks)
    ids = "|".join((selected["目标编号"].astype(str) + selected["任务"].astype(str)).tolist()) if not selected.empty else ""
    return cand, selected, {
        "feasible_candidate_count": int(len(cand)),
        "selected_task_count": int(len(selected)),
        "covered_target_count": int(selected["目标编号"].nunique()) if not selected.empty else 0,
        "normalized_margin_sum": float(selected["稳定裕度"].sum()) if not selected.empty and "稳定裕度" in selected else 0.0,
        "selected_task_ids": ids,
    }


def interpolation_fused(df1: pd.DataFrame, df2: pd.DataFrame, result: AlignmentResult, method: str) -> pd.DataFrame:
    t1, x1, y1 = _prepare(df1)
    t2, x2, y2 = _prepare(df2)
    grid = _sample_grid(result.overlap_start, result.overlap_end, 0.1)
    if method in {"cubic", "cubic_savgol"}:
        f = CubicSpline
    elif method == "pchip":
        f = PchipInterpolator
    else:
        f = lambda t, v: interp1d(t, v, kind="linear", fill_value="extrapolate", assume_sorted=True)
    p1x = f(t1, x1)(grid)
    p1y = f(t1, y1)(grid)
    p2x = f(t2, x2)(grid - result.delta) - result.bias_x
    p2y = f(t2, y2)(grid - result.delta) - result.bias_y
    out = pd.DataFrame({TIME_COL: np.round(grid, 3), "x1_aligned": p1x, "y1_aligned": p1y, "x2_aligned_corrected": p2x, "y2_aligned_corrected": p2y, X_COL: (p1x + p2x) / 2, Y_COL: (p1y + p2y) / 2})
    if method in {"linear_savgol", "cubic_savgol"}:
        win = min(81, len(out) - 1 if len(out) % 2 == 0 else len(out))
        if win % 2 == 0:
            win -= 1
        out[X_COL] = savgol_filter(out[X_COL], win, 3, mode="interp")
        out[Y_COL] = savgol_filter(out[Y_COL], win, 3, mode="interp")
    return out
