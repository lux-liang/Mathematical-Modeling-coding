from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alignment import AlignmentResult
from data_io import TIME_COL, X_COL, Y_COL


def _setup_font() -> None:
    warnings.filterwarnings("ignore", message="Glyph .* missing from font.*", category=UserWarning)
    font_candidates = [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    family = None
    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            fm.fontManager.addfont(str(path))
            family = fm.FontProperties(fname=str(path)).get_name()
            break
    if family:
        plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_raw(data: dict[str, pd.DataFrame], title: str, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(7, 5))
    for sheet, df in data.items():
        plt.plot(df[X_COL], df[Y_COL], label=sheet, linewidth=1.2)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_aligned(fused: pd.DataFrame, title: str, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(7, 5))
    plt.plot(fused["x1_aligned"], fused["y1_aligned"], label="方式1", linewidth=1.0)
    plt.plot(fused["x2_aligned_corrected"], fused["y2_aligned_corrected"], label="方式2校正后", linewidth=1.0)
    plt.plot(fused[X_COL], fused[Y_COL], label="融合", linewidth=1.5)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_series(df: pd.DataFrame, col: str, title: str, ylabel: str, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(8, 3.8))
    plt.plot(df[TIME_COL], df[col], linewidth=1.2)
    plt.title(title)
    plt.xlabel("时间(s)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_tasks(traj: pd.DataFrame, targets: pd.DataFrame, tasks: pd.DataFrame, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(7, 5))
    plt.plot(traj[X_COL], traj[Y_COL], color="#555555", linewidth=1.0, label="附件3融合轨迹")
    colors = {"射击": "#d62728", "拍照": "#1f77b4"}
    for task_type, part in targets.groupby("任务"):
        plt.scatter(part[X_COL], part[Y_COL], s=35, label=f"{task_type}目标", color=colors.get(task_type))
    if not tasks.empty:
        chosen = targets.merge(tasks[["目标编号", "任务"]].drop_duplicates(), left_on=["编号", "任务"], right_on=["目标编号", "任务"])
        plt.scatter(chosen[X_COL], chosen[Y_COL], marker="*", s=120, color="#ffbf00", label="已选任务目标")
    plt.axis("equal")
    plt.title("最终任务点分布")
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_residuals(residuals: pd.DataFrame, title: str, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(5.5, 5))
    plt.scatter(residuals["residual_x_after"], residuals["residual_y_after"], s=8, alpha=0.45)
    plt.axhline(0, color="#777777", linewidth=0.8)
    plt.axvline(0, color="#777777", linewidth=0.8)
    plt.title(title)
    plt.xlabel("残差X(m)")
    plt.ylabel("残差Y(m)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_fused_trajectory(df: pd.DataFrame, title: str, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(7, 5))
    plt.plot(df[X_COL], df[Y_COL], linewidth=1.4, color="#2c5aa0")
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_task_timeline(candidates: pd.DataFrame, tasks: pd.DataFrame, path: Path) -> None:
    _setup_font()
    plt.figure(figsize=(9, 3.8))
    if not candidates.empty:
        y = [0 if t == "射击" else 1 for t in candidates["任务"]]
        plt.scatter(candidates["exec_time"], y, s=8, alpha=0.25, label="可行候选")
    if not tasks.empty:
        y_sel = [0 if t == "射击" else 1 for t in tasks["任务"]]
        plt.scatter(tasks["任务执行时刻(s)"], y_sel, s=55, marker="*", color="#d62728", label="最终选择")
        for _, row in tasks.iterrows():
            yy = 0 if row["任务"] == "射击" else 1
            plt.hlines(yy, row["开始准备时刻(s)"], row["任务执行时刻(s)"], color="#d62728", linewidth=2)
    plt.yticks([0, 1], ["射击", "拍照"])
    plt.xlabel("时间(s)")
    plt.title("可行任务窗口时间轴")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_task_feasibility_heatmap(candidates: pd.DataFrame, path: Path, bin_seconds: float = 10.0) -> None:
    _setup_font()
    if candidates.empty:
        return
    work = candidates.copy()
    work["target_task"] = work["目标编号"].astype(str) + work["任务"].astype(str)
    start = float(np.floor(work["exec_time"].min() / bin_seconds) * bin_seconds)
    end = float(np.ceil(work["exec_time"].max() / bin_seconds) * bin_seconds)
    bins = np.arange(start, end + bin_seconds, bin_seconds)
    if len(bins) < 2:
        bins = np.array([start, start + bin_seconds])
    work["time_bin"] = pd.cut(work["exec_time"], bins=bins, right=False, include_lowest=True)
    heat = work.pivot_table(
        index="target_task",
        columns="time_bin",
        values="normalized_margin",
        aggfunc="max",
        observed=False,
    )
    target_order = sorted(heat.index, key=lambda v: (str(v)[0], int("".join(ch for ch in str(v) if ch.isdigit()) or 0)))
    heat = heat.reindex(target_order)
    values = heat.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    fig_width = max(9.0, min(16.0, 0.28 * heat.shape[1] + 4.0))
    fig_height = max(5.5, min(12.0, 0.22 * heat.shape[0] + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.cm.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0)

    centers = [interval.left + bin_seconds / 2 for interval in heat.columns]
    step = max(1, len(centers) // 12)
    ax.set_xticks(np.arange(len(centers))[::step])
    ax.set_xticklabels([f"{centers[i]:.0f}" for i in range(0, len(centers), step)], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("任务执行时刻(s)")
    ax.set_ylabel("目标与任务类型")
    ax.set_title("目标-时间可行性热力图")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("最佳稳定裕度")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
