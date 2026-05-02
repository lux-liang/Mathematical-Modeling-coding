from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from alignment import AlignmentResult, align_pair, resample_aligned
from data_io import TIME_COL, X_COL, Y_COL
from plotting import _setup_font


@dataclass
class KalmanConfig:
    """Noise parameters for the multi-rate Kalman bias model."""

    name: str
    q_pos: float
    q_vel: float
    q_bias: float
    r1: float
    r2: float


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[[TIME_COL, X_COL, Y_COL]].dropna().drop_duplicates(TIME_COL).sort_values(TIME_COL).reset_index(drop=True)


def _event_table(df1: pd.DataFrame, df2: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Merge asynchronous source-1 and source-2 observations after shifting source 2 by delta."""

    a = _clean(df1).copy()
    b = _clean(df2).copy()
    a["source"] = 1
    b["source"] = 2
    b[TIME_COL] = b[TIME_COL] + delta
    start = max(float(a[TIME_COL].min()), float(b[TIME_COL].min()))
    end = min(float(a[TIME_COL].max()), float(b[TIME_COL].max()))
    obs = pd.concat([a, b], ignore_index=True)
    obs = obs[(obs[TIME_COL] >= start) & (obs[TIME_COL] <= end)].copy()
    obs = obs.sort_values([TIME_COL, "source"]).reset_index(drop=True)
    return obs


def _transition(dt: float) -> np.ndarray:
    f = np.eye(6)
    f[0, 2] = dt
    f[1, 3] = dt
    return f


def _process_noise(dt: float, cfg: KalmanConfig) -> np.ndarray:
    q = np.diag([cfg.q_pos, cfg.q_pos, cfg.q_vel, cfg.q_vel, cfg.q_bias, cfg.q_bias])
    return q * max(dt, 1e-3)


def _obs_model(source: int) -> np.ndarray:
    h = np.zeros((2, 6))
    h[0, 0] = 1.0
    h[1, 1] = 1.0
    if source == 2:
        h[0, 4] = 1.0
        h[1, 5] = 1.0
    return h


def _initial_state(obs: pd.DataFrame, baseline: AlignmentResult) -> tuple[np.ndarray, np.ndarray]:
    first = obs.iloc[0]
    x0 = np.zeros(6)
    if int(first["source"]) == 2:
        x0[0] = float(first[X_COL]) - baseline.bias_x
        x0[1] = float(first[Y_COL]) - baseline.bias_y
    else:
        x0[0] = float(first[X_COL])
        x0[1] = float(first[Y_COL])
    x0[4] = baseline.bias_x
    x0[5] = baseline.bias_y
    p0 = np.diag([10.0, 10.0, 5.0, 5.0, 5.0, 5.0])
    return x0, p0


def kalman_filter(df1: pd.DataFrame, df2: pd.DataFrame, delta: float, baseline: AlignmentResult, cfg: KalmanConfig) -> dict[str, object]:
    """Run a multi-rate Kalman filter with fixed source-2 bias states."""

    obs = _event_table(df1, df2, delta)
    if len(obs) < 5:
        raise ValueError("not enough overlapping observations for Kalman filtering")
    x, p = _initial_state(obs, baseline)
    xs_f, ps_f, xs_pred, ps_pred, fs = [], [], [], [], []
    nll = 0.0
    prev_t = float(obs.iloc[0][TIME_COL])
    eye = np.eye(6)
    for _, row in obs.iterrows():
        t = float(row[TIME_COL])
        dt = max(0.0, t - prev_t)
        f = _transition(dt)
        q = _process_noise(dt, cfg)
        x_pred = f @ x
        p_pred = f @ p @ f.T + q
        h = _obs_model(int(row["source"]))
        r_scale = cfg.r1 if int(row["source"]) == 1 else cfg.r2
        r = np.eye(2) * max(r_scale, 1e-6)
        z = np.array([float(row[X_COL]), float(row[Y_COL])])
        nu = z - h @ x_pred
        s = h @ p_pred @ h.T + r
        s_inv = np.linalg.pinv(s)
        k = p_pred @ h.T @ s_inv
        x = x_pred + k @ nu
        p = (eye - k @ h) @ p_pred @ (eye - k @ h).T + k @ r @ k.T
        sign, logdet = np.linalg.slogdet(s)
        nll += float(nu.T @ s_inv @ nu + (logdet if sign > 0 else 0.0))
        xs_pred.append(x_pred)
        ps_pred.append(p_pred)
        xs_f.append(x.copy())
        ps_f.append(p.copy())
        fs.append(f)
        prev_t = t
    return {
        "obs": obs,
        "x_filter": np.vstack(xs_f),
        "p_filter": np.stack(ps_f),
        "x_pred": np.vstack(xs_pred),
        "p_pred": np.stack(ps_pred),
        "f": fs,
        "nll": nll,
    }


def rts_smoother(filtered: dict[str, object]) -> np.ndarray:
    """Apply Rauch-Tung-Striebel smoothing to filtered states."""

    xf = np.asarray(filtered["x_filter"])
    pf = np.asarray(filtered["p_filter"])
    xp = np.asarray(filtered["x_pred"])
    pp = np.asarray(filtered["p_pred"])
    fs = filtered["f"]
    xs = xf.copy()
    ps = pf.copy()
    for k in range(len(xf) - 2, -1, -1):
        c = pf[k] @ fs[k + 1].T @ np.linalg.pinv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T
    return xs


def _state_to_10hz(obs: pd.DataFrame, xs: np.ndarray) -> pd.DataFrame:
    t = obs[TIME_COL].to_numpy(float)
    order = np.argsort(t)
    t = t[order]
    xs = xs[order]
    _, unique_idx = np.unique(t, return_index=True)
    unique_idx = np.sort(unique_idx)
    t = t[unique_idx]
    xs = xs[unique_idx]
    grid = np.round(np.arange(t[0], t[-1] + 1e-9, 0.1), 3)
    out = {TIME_COL: grid}
    for col, idx in [("X坐标(m)", 0), ("Y坐标(m)", 1), ("vx", 2), ("vy", 3), ("bias_x", 4), ("bias_y", 5)]:
        out[col] = np.interp(grid, t, xs[:, idx])
    return pd.DataFrame(out)


def _rmse_after(df1: pd.DataFrame, df2: pd.DataFrame, delta: float, bx: float, by: float, smooth_window: int = 5) -> tuple[float, float, float, float]:
    a = _clean(df1)
    b = _clean(df2)
    if smooth_window > 1:
        a = a.copy()
        b = b.copy()
        a[X_COL] = pd.Series(a[X_COL]).rolling(smooth_window, center=True, min_periods=1).mean()
        a[Y_COL] = pd.Series(a[Y_COL]).rolling(smooth_window, center=True, min_periods=1).mean()
        b[X_COL] = pd.Series(b[X_COL]).rolling(smooth_window, center=True, min_periods=1).mean()
        b[Y_COL] = pd.Series(b[Y_COL]).rolling(smooth_window, center=True, min_periods=1).mean()
    start = max(float(a[TIME_COL].min()), float(b[TIME_COL].min() + delta))
    end = min(float(a[TIME_COL].max()), float(b[TIME_COL].max() + delta))
    grid = np.arange(start, end + 1e-9, 0.1)
    s1x, s1y = CubicSpline(a[TIME_COL], a[X_COL]), CubicSpline(a[TIME_COL], a[Y_COL])
    s2x, s2y = CubicSpline(b[TIME_COL], b[X_COL]), CubicSpline(b[TIME_COL], b[Y_COL])
    before_x = s2x(grid - delta) - s1x(grid)
    before_y = s2y(grid - delta) - s1y(grid)
    after_x = before_x - bx
    after_y = before_y - by
    return (
        float(np.sqrt(np.mean(before_x**2 + before_y**2))),
        float(np.sqrt(np.mean(after_x**2 + after_y**2))),
        float(np.var(after_x)),
        float(np.var(after_y)),
    )


def run_kalman_bias_attachment2(df1: pd.DataFrame, df2: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate fixed source-2 bias after noise separation for attachment 2."""

    tables = out_dir / "tables"
    figs = out_dir / "figures"
    traj_dir = out_dir / "trajectories"
    for p in [tables, figs, traj_dir]:
        p.mkdir(parents=True, exist_ok=True)

    baseline = align_pair(df1, df2, estimate_bias=True, smooth_window=5)
    rmse_before, rmse_after, var_x, var_y = _rmse_after(df1, df2, baseline.delta, baseline.bias_x, baseline.bias_y)
    r_base = max(0.05, (var_x + var_y) / 2.0)
    configs = [
        KalmanConfig("low_process", 0.002, 0.010, 1e-7, r_base, r_base),
        KalmanConfig("balanced", 0.010, 0.050, 1e-6, r_base, r_base),
        KalmanConfig("smooth_bias", 0.030, 0.100, 1e-8, r_base * 1.2, r_base * 1.2),
    ]
    sens_rows = []
    best = None
    for cfg in configs:
        deltas = np.linspace(baseline.delta - 1.0, baseline.delta + 1.0, 21)
        best_for_cfg = None
        for delta in deltas:
            filtered = kalman_filter(df1, df2, float(delta), baseline, cfg)
            xf = np.asarray(filtered["x_filter"])
            bx, by = float(np.median(xf[:, 4])), float(np.median(xf[:, 5]))
            _before, after, vx, vy = _rmse_after(df1, df2, float(delta), bx, by)
            score = float(filtered["nll"])
            if best_for_cfg is None or score < best_for_cfg["nll"]:
                best_for_cfg = {
                    "config": cfg,
                    "Delta": float(delta),
                    "bx": bx,
                    "by": by,
                    "rmse_after": after,
                    "residual_var_x": vx,
                    "residual_var_y": vy,
                    "nll": score,
                    "filtered": filtered,
                }
        assert best_for_cfg is not None
        sens_rows.append(
            {
                "config_name": cfg.name,
                "q_pos": cfg.q_pos,
                "q_vel": cfg.q_vel,
                "q_bias": cfg.q_bias,
                "r1_x": cfg.r1,
                "r1_y": cfg.r1,
                "r2_x": cfg.r2,
                "r2_y": cfg.r2,
                "Delta": best_for_cfg["Delta"],
                "bx": best_for_cfg["bx"],
                "by": best_for_cfg["by"],
                "rmse_after": best_for_cfg["rmse_after"],
                "residual_var_x": best_for_cfg["residual_var_x"],
                "residual_var_y": best_for_cfg["residual_var_y"],
                "nll": best_for_cfg["nll"],
            }
        )
        if best is None or best_for_cfg["nll"] < best["nll"]:
            best = best_for_cfg
    assert best is not None
    xs = rts_smoother(best["filtered"])
    smoothed = _state_to_10hz(best["filtered"]["obs"], xs)
    smoothed.to_csv(traj_dir / "attachment2_kalman_rts_10hz.csv", index=False, encoding="utf-8-sig")
    bx_rts, by_rts = float(np.median(xs[:, 4])), float(np.median(xs[:, 5]))
    _before, after_rts, vx_rts, vy_rts = _rmse_after(df1, df2, best["Delta"], bx_rts, by_rts)

    summary = pd.DataFrame(
        [
            {
                "method": "RobustMedian",
                "Delta": baseline.delta,
                "bx": baseline.bias_x,
                "by": baseline.bias_y,
                "rmse_before": rmse_before,
                "rmse_after": rmse_after,
                "residual_var_x": var_x,
                "residual_var_y": var_y,
                "nll": np.nan,
                "notes": "baseline robust median after coarse-to-fine RMSE alignment",
            },
            {
                "method": "KalmanFilter",
                "Delta": best["Delta"],
                "bx": best["bx"],
                "by": best["by"],
                "rmse_before": rmse_before,
                "rmse_after": best["rmse_after"],
                "residual_var_x": best["residual_var_x"],
                "residual_var_y": best["residual_var_y"],
                "nll": best["nll"],
                "notes": f"best sensitivity config={best['config'].name}",
            },
            {
                "method": "KalmanRTS",
                "Delta": best["Delta"],
                "bx": bx_rts,
                "by": by_rts,
                "rmse_before": rmse_before,
                "rmse_after": after_rts,
                "residual_var_x": vx_rts,
                "residual_var_y": vy_rts,
                "nll": best["nll"],
                "notes": "RTS smoothed states using all observations",
            },
        ]
    )
    sensitivity = pd.DataFrame(sens_rows)
    summary.to_csv(tables / "kalman_bias_attachment2_summary.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(tables / "kalman_bias_sensitivity.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(tables / "kalman_qr_sensitivity.csv", index=False, encoding="utf-8-sig")
    diagnostics = pd.DataFrame(
        [
            {
                "dataset": "附件2",
                "diagnostic": "robust_residual_variance_x",
                "value": var_x,
                "unit": "m^2",
                "interpretation": "fixed-bias-corrected residual variance in x direction",
            },
            {
                "dataset": "附件2",
                "diagnostic": "robust_residual_variance_y",
                "value": var_y,
                "unit": "m^2",
                "interpretation": "fixed-bias-corrected residual variance in y direction",
            },
            {
                "dataset": "附件2",
                "diagnostic": "residual_sigma_norm",
                "value": float(np.sqrt(var_x + var_y)),
                "unit": "m",
                "interpretation": "nominal random observation-noise scale after fixed-bias correction",
            },
            {
                "dataset": "附件2",
                "diagnostic": "r_base",
                "value": r_base,
                "unit": "m^2",
                "interpretation": "base measurement-noise covariance used by sensitivity configurations",
            },
            {
                "dataset": "附件2",
                "diagnostic": "rts_rmse_gain_vs_robust",
                "value": float(rmse_after - after_rts),
                "unit": "m",
                "interpretation": "small gain confirms RTS is validation/smoothing rather than a large RMSE-reduction model",
            },
        ]
    )
    diagnostics.to_csv(tables / "attachment2_noise_residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    _plot_kalman_attachment2(df1, df2, baseline, smoothed, figs)
    return summary, sensitivity, smoothed


def _save_both(fig: plt.Figure, path_png: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def _plot_kalman_attachment2(df1: pd.DataFrame, df2: pd.DataFrame, baseline: AlignmentResult, smoothed: pd.DataFrame, figs: Path) -> None:
    _setup_font()
    fused = resample_aligned(df1, df2, baseline, smooth_window=5)
    summary_path = figs.parent / "tables" / "kalman_bias_attachment2_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.bar(summary["method"], summary["rmse_after"], color=["#6B7280", "#356EA9", "#2A9D8F"])
        ax.set_xlabel("方法")
        ax.set_ylabel("校正后 RMSE / m")
        ax.set_title("附件2三种方法 RMSE 对比")
        ax.set_ylim(0, max(0.9, float(summary["rmse_after"].max()) * 1.18))
        for i, val in enumerate(summary["rmse_after"]):
            ax.text(i, float(val) + 0.015, f"{float(val):.3f}", ha="center", va="bottom", fontsize=9)
        _save_both(fig, figs / "attachment2_method_rmse_compare.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(df1[X_COL], df1[Y_COL], color="#9CA3AF", lw=0.8, alpha=0.65, label="方式1原始")
    ax.plot(df2[X_COL], df2[Y_COL], color="#C08497", lw=0.8, alpha=0.55, label="方式2原始")
    ax.plot(fused[X_COL], fused[Y_COL], color="#356EA9", lw=1.2, label="鲁棒校正融合")
    ax.plot(smoothed[X_COL], smoothed[Y_COL], color="#2A9D8F", lw=1.6, label="Kalman-RTS平滑")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("附件2对齐、鲁棒校正与 RTS 平滑轨迹")
    ax.legend(frameon=False, fontsize=8)
    _save_both(fig, figs / "attachment2_trajectory_kalman_rts.png")

    a = _clean(df1)
    b = _clean(df2)
    grid = fused[TIME_COL].to_numpy(float)
    s1x, s1y = CubicSpline(a[TIME_COL], a[X_COL]), CubicSpline(a[TIME_COL], a[Y_COL])
    s2x, s2y = CubicSpline(b[TIME_COL], b[X_COL]), CubicSpline(b[TIME_COL], b[Y_COL])
    before = np.column_stack([s2x(grid - baseline.delta) - s1x(grid), s2y(grid - baseline.delta) - s1y(grid)])
    robust = np.column_stack([before[:, 0] - baseline.bias_x, before[:, 1] - baseline.bias_y])
    interp_x = np.interp(grid, smoothed[TIME_COL], smoothed[X_COL])
    interp_y = np.interp(grid, smoothed[TIME_COL], smoothed[Y_COL])
    kalman = np.column_stack([interp_x - s1x(grid), interp_y - s1y(grid)])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharex=True, sharey=True)
    for ax, arr, title, color in [
        (axes[0], before, "去噪前残差", "#9CA3AF"),
        (axes[1], robust, "固定偏差校正后", "#356EA9"),
        (axes[2], kalman, "Kalman平滑后", "#2A9D8F"),
    ]:
        ax.scatter(arr[:, 0], arr[:, 1], s=6, alpha=0.35, color=color)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.axvline(0, color="#555555", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("残差X / m")
    axes[0].set_ylabel("残差Y / m")
    _save_both(fig, figs / "attachment2_residual_distribution_compare.png")

    # Keep the old filenames as compatibility aliases for the writing repository.
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(df1[X_COL], df1[Y_COL], color="#9CA3AF", lw=0.8, alpha=0.65, label="方式1原始")
    ax.plot(df2[X_COL], df2[Y_COL], color="#C08497", lw=0.8, alpha=0.55, label="方式2原始")
    ax.plot(fused[X_COL], fused[Y_COL], color="#356EA9", lw=1.2, label="鲁棒校正融合")
    ax.plot(smoothed[X_COL], smoothed[Y_COL], color="#2A9D8F", lw=1.6, label="RTS 平滑")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X 坐标 / m")
    ax.set_ylabel("Y 坐标 / m")
    ax.set_title("附件2 Kalman-RTS 去噪与固定偏差估计")
    ax.legend(frameon=False, fontsize=8)
    _save_both(fig, figs / "attachment2_kalman_denoising.png")
