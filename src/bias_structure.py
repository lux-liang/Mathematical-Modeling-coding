from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alignment import align_pair
from data_io import TIME_COL, X_COL, Y_COL
from kinematics import add_kinematics
from plotting import _setup_font


def _clean(df: pd.DataFrame, smooth_window: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    part = df[[TIME_COL, X_COL, Y_COL]].dropna().drop_duplicates(TIME_COL).sort_values(TIME_COL)
    t = part[TIME_COL].to_numpy(float)
    x = part[X_COL].to_numpy(float)
    y = part[Y_COL].to_numpy(float)
    if smooth_window > 1:
        x = uniform_filter1d(x, smooth_window, mode="nearest")
        y = uniform_filter1d(y, smooth_window, mode="nearest")
    return t, x, y


def _residual_frame(df1: pd.DataFrame, df2: pd.DataFrame, delta: float, smooth_window: int = 7) -> pd.DataFrame:
    t1, x1, y1 = _clean(df1, smooth_window)
    t2, x2, y2 = _clean(df2, smooth_window)
    start = max(t1[0], t2[0] + delta)
    end = min(t1[-1], t2[-1] + delta)
    grid = np.arange(start, end + 1e-9, 0.2)
    s1x, s1y = CubicSpline(t1, x1), CubicSpline(t1, y1)
    s2x, s2y = CubicSpline(t2, x2), CubicSpline(t2, y2)
    x1g, y1g = s1x(grid), s1y(grid)
    x2g, y2g = s2x(grid - delta), s2y(grid - delta)
    fused = pd.DataFrame({TIME_COL: grid, X_COL: (x1g + x2g) / 2, Y_COL: (y1g + y2g) / 2})
    fused = add_kinematics(fused, smooth_window=41)
    theta = np.arctan2(fused["vy"].to_numpy(float), fused["vx"].to_numpy(float))
    speed = fused["speed"].to_numpy(float)
    dt = float(np.median(np.diff(grid)))
    heading_rate = np.gradient(theta, dt)
    curvature = np.abs(heading_rate) / np.maximum(speed, 1e-6)
    return pd.DataFrame(
        {
            TIME_COL: grid,
            "residual_x": x2g - x1g,
            "residual_y": y2g - y1g,
            "t_norm": (grid - grid.min()) / max(grid.max() - grid.min(), 1e-9),
            "x": fused[X_COL],
            "y": fused[Y_COL],
            "v": speed,
            "a": fused["acceleration"],
            "theta": theta,
            "cos_theta": np.cos(theta),
            "sin_theta": np.sin(theta),
            "curvature": curvature,
        }
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _blocked_folds(n: int, k: int = 5) -> list[np.ndarray]:
    k = min(k, max(3, n // 30))
    edges = np.linspace(0, n, k + 1, dtype=int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k) if edges[i + 1] > edges[i]]


def _fit_predict(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, object, int]:
    if model_name == "M0":
        return np.zeros((len(x_test), 2)), None, 0
    if model_name == "M1":
        mean = y_train.mean(axis=0)
        return np.repeat(mean[None, :], len(x_test), axis=0), mean, 2
    if model_name == "M2":
        reg = LinearRegression().fit(x_train[:, [0]], y_train)
        return reg.predict(x_test[:, [0]]), reg, 4
    if model_name == "M3":
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x_train, y_train)
        return reg.predict(x_test), reg, 16
    if model_name == "M4":
        reg = make_pipeline(StandardScaler(), KernelRidge(alpha=2.0, kernel="rbf", gamma=0.35)).fit(x_train, y_train)
        return reg.predict(x_test), reg, len(x_train)
    raise ValueError(model_name)


def run_attachment3_bias_structure(df1: pd.DataFrame, df2: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare fixed, time-varying, state-related, and nonlinear residual-bias models."""

    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    baseline = align_pair(df1, df2, estimate_bias=True, smooth_window=7)
    data = _residual_frame(df1, df2, baseline.delta, smooth_window=7)
    features = ["t_norm", "x", "y", "v", "a", "cos_theta", "sin_theta", "curvature"]
    x = data[features].to_numpy(float)
    y = data[["residual_x", "residual_y"]].to_numpy(float)
    folds = _blocked_folds(len(data), 5)
    descriptions = {
        "M0": "no systematic bias, residual prediction is zero",
        "M1": "constant translation bias",
        "M2": "linear time-varying bias",
        "M3": "state-related linear Ridge bias",
        "M4": "nonlinear RBF kernel-ridge bias with blocked CV",
    }
    rows = []
    full_models = {}
    m0_cv = None
    m1_cv = None
    for name in ["M0", "M1", "M2", "M3", "M4"]:
        pred_train, model, n_params = _fit_predict(name, x, y, x)
        train_rmse = _rmse(y, pred_train)
        cv_preds = np.zeros_like(y)
        for test_idx in folds:
            train_mask = np.ones(len(data), dtype=bool)
            train_mask[test_idx] = False
            pred, _, _ = _fit_predict(name, x[train_mask], y[train_mask], x[test_idx])
            cv_preds[test_idx] = pred
        cv_rmse = _rmse(y, cv_preds)
        cv_rmse_x = float(np.sqrt(mean_squared_error(y[:, 0], cv_preds[:, 0])))
        cv_rmse_y = float(np.sqrt(mean_squared_error(y[:, 1], cv_preds[:, 1])))
        if name == "M0":
            m0_cv = cv_rmse
        if name == "M1":
            m1_cv = cv_rmse
        imp0 = 0.0 if not m0_cv else (m0_cv - cv_rmse) / m0_cv
        imp1 = 0.0 if not m1_cv else (m1_cv - cv_rmse) / m1_cv
        if name == "M4" and train_rmse < cv_rmse * 0.75:
            conclusion = "nonlinear training gain, CV suggests overfit risk"
        elif name in {"M2", "M3", "M4"} and imp1 >= 0.05:
            conclusion = "structured non-fixed systematic bias supported"
        elif name == "M1" and imp0 >= 0.05:
            conclusion = "constant translation explains part of residual"
        else:
            conclusion = "not primary explanatory model"
        rows.append(
            {
                "model_name": name,
                "description": descriptions[name],
                "n_params": n_params,
                "train_rmse": train_rmse,
                "cv_rmse": cv_rmse,
                "cv_rmse_x": cv_rmse_x,
                "cv_rmse_y": cv_rmse_y,
                "improvement_vs_M0": imp0,
                "improvement_vs_M1": imp1,
                "conclusion": conclusion,
            }
        )
        full_models[name] = model
    comparison = pd.DataFrame(rows)
    comparison.to_csv(tables / "attachment3_bias_structure_comparison.csv", index=False, encoding="utf-8-sig")

    coeff_rows = []
    model_m3 = full_models["M3"]
    if model_m3 is not None:
        ridge = model_m3.named_steps["ridge"]
        for axis, coefs in zip(["x", "y"], ridge.coef_):
            for feature, coef in zip(features, coefs):
                coeff_rows.append({"model_name": "M3", "axis": axis, "feature": feature, "coefficient": float(coef)})
    coeffs = pd.DataFrame(coeff_rows)
    coeffs.to_csv(tables / "attachment3_bias_feature_coefficients.csv", index=False, encoding="utf-8-sig")
    data.to_csv(tables / "attachment3_residual_feature_frame.csv", index=False, encoding="utf-8-sig")
    _plot_bias_structure(data, comparison, figs)
    return comparison, coeffs, data


def _save_both(fig: plt.Figure, path_png: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def _plot_bias_structure(data: pd.DataFrame, comparison: pd.DataFrame, figs: Path) -> None:
    _setup_font()
    t = data[TIME_COL].to_numpy(float)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True)
    for ax, col, color in [(axes[0], "residual_x", "#356EA9"), (axes[1], "residual_y", "#D97732")]:
        y = data[col].to_numpy(float)
        ax.scatter(t, y, s=5, alpha=0.25, color=color)
        trend = pd.Series(y).rolling(61, center=True, min_periods=5).mean().to_numpy()
        ax.plot(t, trend, color="#111827", lw=1.0, label="滚动趋势")
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_ylabel(f"{col} / m")
        ax.legend(frameon=False)
    axes[1].set_xlabel("时间 / s")
    fig.suptitle("附件3残差分量随时间变化")
    _save_both(fig, figs / "attachment3_residual_time_trend_clean.png")

    step = max(1, len(data) // 120)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    part = data.iloc[::step]
    ax.plot(data["x"], data["y"], color="#9CA3AF", lw=0.9, label="状态估计轨迹")
    norm = np.sqrt(part["residual_x"] ** 2 + part["residual_y"] ** 2)
    q = ax.quiver(
        part["x"],
        part["y"],
        part["residual_x"],
        part["residual_y"],
        norm,
        angles="xy",
        scale_units="xy",
        scale=14,
        cmap="viridis",
        width=0.003,
        alpha=0.78,
    )
    ax.scatter(data["x"].iloc[0], data["y"].iloc[0], marker="o", s=42, color="#2A9D8F", label="起点")
    ax.scatter(data["x"].iloc[-1], data["y"].iloc[-1], marker="s", s=42, color="#B94A48", label="终点")
    cbar = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("残差范数 / m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("附件3残差向量场")
    ax.legend(frameon=False)
    _save_both(fig, figs / "attachment3_residual_vector_field_clean.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(comparison))
    ax.bar(x - 0.18, comparison["train_rmse"], width=0.34, color="#356EA9", label="训练RMSE")
    ax.bar(x + 0.18, comparison["cv_rmse"], width=0.34, color="#2A9D8F", label="块交叉验证RMSE")
    ax.set_xticks(x, comparison["model_name"])
    ax.set_ylabel("RMSE / m")
    ax.set_title("附件3系统偏差模型比较")
    ax.legend(frameon=False)
    _save_both(fig, figs / "attachment3_bias_model_cv_compare_clean.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))
    pairs = [("v", "速度 / (m/s)"), ("a", "加速度 / (m/s²)"), ("theta", "航向角 / rad"), ("curvature", "曲率近似")]
    for ax, (col, label) in zip(axes.ravel(), pairs):
        ax.scatter(data[col], data["residual_x"], s=5, alpha=0.25, color="#356EA9", label="rx")
        ax.scatter(data[col], data["residual_y"], s=5, alpha=0.25, color="#D97732", label="ry")
        ax.set_xlabel(label)
        ax.set_ylabel("残差 / m")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("附件3残差与运动状态关系")
    _save_both(fig, figs / "attachment3_residual_state_relation_clean.png")

    # Compatibility aliases used by the writing repository before the final cleanup.
    for old, new in [
        ("attachment3_residual_time_trend.png", "attachment3_residual_time_trend_clean.png"),
        ("attachment3_residual_vector_field.png", "attachment3_residual_vector_field_clean.png"),
        ("attachment3_bias_model_comparison.png", "attachment3_bias_model_cv_compare_clean.png"),
        ("attachment3_residual_vs_state.png", "attachment3_residual_state_relation_clean.png"),
    ]:
        src_png = figs / new
        if src_png.exists():
            (figs / old).write_bytes(src_png.read_bytes())
        src_pdf = src_png.with_suffix(".pdf")
        if src_pdf.exists():
            (figs / old).with_suffix(".pdf").write_bytes(src_pdf.read_bytes())
