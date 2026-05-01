from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import TIME_COL, X_COL, Y_COL
from plotting import _setup_font


RNG_SEED = 42
SHOT_HIT_PROB = 0.85


def circular_angle_diff(a: float, b: float) -> float:
    """Smallest circular angle difference in degrees."""

    raw = abs(float(a) - float(b)) % 360.0
    return min(raw, 360.0 - raw)


def _intervals_from_times(times: list[float]) -> str:
    if not times:
        return ""
    vals = sorted(times)
    runs = []
    start = prev = vals[0]
    for t in vals[1:]:
        if t - prev <= 0.11:
            prev = t
        else:
            runs.append(f"{start:.1f}-{prev:.1f}")
            start = prev = t
    runs.append(f"{start:.1f}-{prev:.1f}")
    return ";".join(runs[:8])


def _margin(dwin: np.ndarray, speed: np.ndarray, acc: np.ndarray, dmin: float, dmax: float, vmax: float, amax: float) -> float:
    vals = [
        np.min((dwin - dmin) / (dmax - dmin)),
        np.min((dmax - dwin) / (dmax - dmin)),
        np.min((vmax - speed) / vmax),
        np.min((amax - acc) / amax),
    ]
    return float(np.min(vals))


def _conflicts(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return not (a_end < b_start or b_end < a_start)


def split_targets(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return shooting and photo target tables."""

    return (
        targets[targets["任务"] == "射击"].copy().reset_index(drop=True),
        targets[targets["任务"] == "拍照"].copy().reset_index(drop=True),
    )


def generate_shooting_events(traj: pd.DataFrame, shooting_targets: pd.DataFrame, stride: int = 1) -> pd.DataFrame:
    """Enumerate all feasible shooting events on a 10 Hz trajectory."""

    rows = []
    xy = traj[[X_COL, Y_COL]].to_numpy(float)
    speed = traj["speed"].to_numpy(float)
    acc = traj["acceleration"].to_numpy(float)
    times = traj[TIME_COL].to_numpy(float)
    for _, target in shooting_targets.iterrows():
        tx, ty = float(target[X_COL]), float(target[Y_COL])
        dist = np.sqrt((xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2)
        feasible_times = []
        for i in range(15, len(traj), stride):
            dwin = dist[i - 15 : i + 1]
            swin = speed[i - 15 : i + 1]
            awin = acc[i - 15 : i + 1]
            if np.all((dwin >= 5.0) & (dwin <= 30.0)) and np.all(swin <= 2.0) and np.all(awin <= 1.5):
                feasible_times.append(float(times[i]))
                m = _margin(dwin, swin, awin, 5.0, 30.0, 2.0, 1.5)
                rows.append(
                    {
                        "event_id": f"S_{len(rows):05d}",
                        "event_type": "shooting",
                        "target_id": str(target["编号"]),
                        "start_time": float(times[i - 15]),
                        "execute_time": float(times[i]),
                        "distance": float(dist[i]),
                        "speed": float(speed[i]),
                        "acceleration": float(acc[i]),
                        "margin": m,
                    }
                )
    return pd.DataFrame(rows)


def select_shooting_plan(events: pd.DataFrame, shooting_targets: pd.DataFrame | None = None, max_shots_per_target: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select non-overlapping shooting events with diminishing hit-probability gain."""

    all_target_ids = shooting_targets["编号"].astype(str).tolist() if shooting_targets is not None else sorted(events["target_id"].unique()) if not events.empty else []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "target_id": tid,
                    "feasible_event_count": 0,
                    "feasible_time_intervals": "",
                    "selected_shots": 0,
                    "hit_probability": 0.0,
                    "expected_hit": 0.0,
                    "min_margin": 0.0,
                    "risk_level": "高",
                }
                for tid in all_target_ids
            ]
        )
    selected = []
    intervals: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    # Each pass adds the next shot slot for targets with the lowest current hit probability.
    for shot_slot in range(1, max_shots_per_target + 1):
        target_order = sorted(events["target_id"].unique(), key=lambda k: (counts.get(k, 0), events[events["target_id"] == k]["margin"].median()))
        for target_id in target_order:
            if counts.get(target_id, 0) >= shot_slot:
                continue
            part = events[events["target_id"] == target_id].copy()
            part["score"] = part["margin"] - 0.002 * part["execute_time"]
            for _, event in part.sort_values(["score", "execute_time"], ascending=[False, True]).iterrows():
                st, en = float(event["start_time"]), float(event["execute_time"])
                if any(_conflicts(st, en, a, b) for a, b in intervals):
                    continue
                row = event.to_dict()
                row["shot_index_for_target"] = counts.get(target_id, 0) + 1
                row["marginal_hit_gain"] = SHOT_HIT_PROB * (1.0 - SHOT_HIT_PROB) ** (row["shot_index_for_target"] - 1)
                row["cumulative_hit_probability"] = 1.0 - (1.0 - SHOT_HIT_PROB) ** row["shot_index_for_target"]
                selected.append(row)
                intervals.append((st, en))
                counts[target_id] = row["shot_index_for_target"]
                break
    selected_df = pd.DataFrame(selected).sort_values("execute_time").reset_index(drop=True)
    if not selected_df.empty:
        selected_df.insert(0, "seq", np.arange(1, len(selected_df) + 1))
    summary_rows = []
    for target_id in all_target_ids:
        part = events[events["target_id"].astype(str) == target_id]
        chosen = selected_df[selected_df["target_id"] == target_id] if not selected_df.empty else pd.DataFrame()
        n = int(len(chosen))
        margins = chosen["margin"].to_numpy(float) if n else np.array([])
        summary_rows.append(
            {
                "target_id": target_id,
                "feasible_event_count": int(len(part)),
                "feasible_time_intervals": _intervals_from_times(part["execute_time"].round(1).tolist()),
                "selected_shots": n,
                "hit_probability": float(1.0 - (1.0 - SHOT_HIT_PROB) ** n),
                "expected_hit": float(1.0 - (1.0 - SHOT_HIT_PROB) ** n),
                "min_margin": float(np.min(margins)) if n else 0.0,
                "risk_level": "高" if n == 0 or (len(margins) and np.min(margins) < 0.03) else "中" if len(margins) and np.min(margins) < 0.08 else "低",
            }
        )
    return selected_df, pd.DataFrame(summary_rows)


def generate_photo_events(traj: pd.DataFrame, photo_targets: pd.DataFrame, fov_degree: float = 45.0, time_stride: int = 5) -> pd.DataFrame:
    """Generate camera events e=(time, direction) that may cover multiple photo targets."""

    rows = []
    xy = traj[[X_COL, Y_COL]].to_numpy(float)
    speed = traj["speed"].to_numpy(float)
    acc = traj["acceleration"].to_numpy(float)
    times = traj[TIME_COL].to_numpy(float)
    target_xy = photo_targets[[X_COL, Y_COL]].to_numpy(float)
    target_ids = photo_targets["编号"].astype(str).tolist()
    half = fov_degree / 2.0
    for i in range(5, len(traj), time_stride):
        if np.any(speed[i - 5 : i + 1] > 1.5) or np.any(acc[i - 5 : i + 1] > 1.5):
            continue
        vec = target_xy - xy[i]
        dist_now = np.sqrt(np.sum(vec * vec, axis=1))
        visible_idx = np.where((dist_now >= 10.0) & (dist_now <= 40.0))[0]
        if len(visible_idx) == 0:
            continue
        angles = np.degrees(np.arctan2(vec[visible_idx, 1], vec[visible_idx, 0]))
        directions = sorted(set(round(float(a) / 5.0) * 5.0 for a in angles))
        for phi in directions:
            covered = []
            covered_angles = []
            covered_dist = []
            for idx, ang in zip(visible_idx, angles):
                if circular_angle_diff(phi, ang) <= half:
                    # Distance must hold through the preparation window for this target.
                    dwin = np.sqrt(np.sum((target_xy[idx] - xy[i - 5 : i + 1]) ** 2, axis=1))
                    if np.all((dwin >= 10.0) & (dwin <= 40.0)):
                        covered.append(target_ids[idx])
                        covered_angles.append(float(ang))
                        covered_dist.append(float(dist_now[idx]))
            if not covered:
                continue
            all_margins = []
            for tid in covered:
                pidx = target_ids.index(tid)
                dwin = np.sqrt(np.sum((target_xy[pidx] - xy[i - 5 : i + 1]) ** 2, axis=1))
                all_margins.append(_margin(dwin, speed[i - 5 : i + 1], acc[i - 5 : i + 1], 10.0, 40.0, 1.5, 1.5))
            rows.append(
                {
                    "event_id": f"P_{len(rows):05d}",
                    "event_type": "photo",
                    "start_time": float(times[i - 5]),
                    "execute_time": float(times[i]),
                    "camera_direction": float(phi),
                    "covered_targets": ",".join(covered),
                    "num_covered_targets": int(len(covered)),
                    "target_angles": ",".join(f"{a:.1f}" for a in covered_angles),
                    "speed": float(speed[i]),
                    "acceleration": float(acc[i]),
                    "min_distance": float(np.min(covered_dist)),
                    "max_distance": float(np.max(covered_dist)),
                    "margin": float(np.min(all_margins)),
                    "fov_degree": float(fov_degree),
                }
            )
    return pd.DataFrame(rows)


def _photo_angle_ok(event: pd.Series, target_angles: dict[str, list[float]]) -> bool:
    targets = str(event["covered_targets"]).split(",")
    angles = [float(a) for a in str(event["target_angles"]).split(",") if a != ""]
    for tid, angle in zip(targets, angles):
        if any(circular_angle_diff(angle, old) < 60.0 for old in target_angles.get(tid, [])):
            return False
    return True


def _record_photo_angles(event: pd.Series, target_angles: dict[str, list[float]]) -> None:
    targets = str(event["covered_targets"]).split(",")
    angles = [float(a) for a in str(event["target_angles"]).split(",") if a != ""]
    for tid, angle in zip(targets, angles):
        target_angles.setdefault(tid, []).append(angle)


def select_photo_plan(events: pd.DataFrame, photo_targets: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select non-overlapping photo events while enforcing per-target 60 degree separation."""

    all_target_ids = photo_targets["编号"].astype(str).tolist() if photo_targets is not None else sorted(set(",".join(events["covered_targets"]).split(","))) if not events.empty else []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "target_id": tid,
                    "feasible_event_count": 0,
                    "selected_photo_count": 0,
                    "selected_angles": "",
                    "angle_gap_min": np.nan,
                    "covered_by_multi_target_event_count": 0,
                    "risk_level": "高",
                }
                for tid in all_target_ids
            ]
        )
    selected = []
    intervals: list[tuple[float, float]] = []
    target_angles: dict[str, list[float]] = {}
    ordered = events.copy()
    ordered["new_target_score"] = ordered["num_covered_targets"] + 0.25 * (ordered["num_covered_targets"] > 1).astype(float) + ordered["margin"]
    for _, event in ordered.sort_values(["new_target_score", "margin", "execute_time"], ascending=[False, False, True]).iterrows():
        st, en = float(event["start_time"]), float(event["execute_time"])
        if any(_conflicts(st, en, a, b) for a, b in intervals):
            continue
        if not _photo_angle_ok(event, target_angles):
            continue
        selected.append(event.to_dict())
        intervals.append((st, en))
        _record_photo_angles(event, target_angles)
    selected_df = pd.DataFrame(selected).sort_values("execute_time").reset_index(drop=True)
    if not selected_df.empty:
        selected_df.insert(0, "seq", np.arange(1, len(selected_df) + 1))
    summary_rows = []
    for tid in all_target_ids:
        feasible = events[events["covered_targets"].str.split(",").apply(lambda xs: tid in xs)]
        chosen = selected_df[selected_df["covered_targets"].str.split(",").apply(lambda xs: tid in xs)] if not selected_df.empty else pd.DataFrame()
        angles = []
        multi = 0
        for _, ev in chosen.iterrows():
            ts = str(ev["covered_targets"]).split(",")
            ans = [float(a) for a in str(ev["target_angles"]).split(",") if a != ""]
            if tid in ts:
                angles.append(ans[ts.index(tid)])
            if int(ev["num_covered_targets"]) > 1:
                multi += 1
        min_gap = min((circular_angle_diff(a, b) for i, a in enumerate(angles) for b in angles[i + 1 :]), default=np.nan)
        summary_rows.append(
            {
                "target_id": tid,
                "feasible_event_count": int(len(feasible)),
                "selected_photo_count": int(len(chosen)),
                "selected_angles": ",".join(f"{a:.1f}" for a in angles),
                "angle_gap_min": min_gap,
                "covered_by_multi_target_event_count": int(multi),
                "risk_level": "高" if len(chosen) == 0 else "中" if len(angles) > 1 and min_gap < 60.0 else "低",
            }
        )
    return selected_df, pd.DataFrame(summary_rows)


def select_joint_plan(shoot_events: pd.DataFrame, photo_events: pd.DataFrame, max_shots_per_target: int = 5) -> pd.DataFrame:
    """Greedy fallback joint scheduler for shooting and photo events with non-overlap windows."""

    shot_pool = shoot_events.copy()
    photo_pool = photo_events.copy()
    shot_pool["covered_targets"] = shot_pool["target_id"]
    shot_pool["camera_direction"] = np.nan
    shot_pool["num_covered_targets"] = 1
    pool = pd.concat([shot_pool, photo_pool], ignore_index=True, sort=False)
    selected = []
    intervals: list[tuple[float, float]] = []
    shot_counts: dict[str, int] = {}
    photo_angles: dict[str, list[float]] = {}
    remaining = pool.copy()
    while not remaining.empty:
        scores = []
        for _, event in remaining.iterrows():
            st, en = float(event["start_time"]), float(event["execute_time"])
            if any(_conflicts(st, en, a, b) for a, b in intervals):
                scores.append(-np.inf)
                continue
            if event["event_type"] == "shooting":
                target = str(event["target_id"])
                count = shot_counts.get(target, 0)
                if count >= max_shots_per_target:
                    scores.append(-np.inf)
                    continue
                marginal = SHOT_HIT_PROB * (1.0 - SHOT_HIT_PROB) ** count
                first_shot_bonus = 3.0 if count == 0 else 0.0
                scores.append(10.0 * marginal + first_shot_bonus + float(event["margin"]))
            else:
                if not _photo_angle_ok(event, photo_angles):
                    scores.append(-np.inf)
                    continue
                targets = str(event["covered_targets"]).split(",")
                new_targets = [t for t in targets if t not in photo_angles]
                scores.append(2.5 * len(new_targets) + 0.45 * len(targets) + 0.35 * (len(targets) > 1) + float(event["margin"]))
        best_pos = int(np.argmax(scores))
        if not np.isfinite(scores[best_pos]) or scores[best_pos] <= 0:
            break
        event = remaining.iloc[best_pos].copy()
        st, en = float(event["start_time"]), float(event["execute_time"])
        if event["event_type"] == "shooting":
            target = str(event["target_id"])
            event = event.copy()
            event["shot_index_for_target"] = shot_counts.get(target, 0) + 1
            event["cumulative_hit_probability"] = 1.0 - (1.0 - SHOT_HIT_PROB) ** int(event["shot_index_for_target"])
            shot_counts[target] = int(event["shot_index_for_target"])
        else:
            _record_photo_angles(event, photo_angles)
        selected.append(event.to_dict())
        intervals.append((st, en))
        remaining = remaining.drop(remaining.index[best_pos]).reset_index(drop=True)
    out = pd.DataFrame(selected).sort_values("execute_time").reset_index(drop=True)
    if not out.empty:
        out.insert(0, "seq", np.arange(1, len(out) + 1))
    return out


def plan_metrics(plan_name: str, events: pd.DataFrame) -> dict[str, object]:
    """Aggregate event-plan metrics for paper tables."""

    if events.empty:
        return {
            "plan_name": plan_name,
            "shooting_events": 0,
            "photo_events": 0,
            "total_events": 0,
            "shooting_targets_covered": 0,
            "photo_targets_covered": 0,
            "expected_shooting_hits": 0.0,
            "total_photo_observations": 0,
            "multi_target_photo_events": 0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "risk_count": 0,
            "solver_status": "greedy_interval_fallback",
        }
    shooting = events[events["event_type"] == "shooting"]
    photo = events[events["event_type"] == "photo"]
    expected = 0.0
    if not shooting.empty:
        for _, part in shooting.groupby("target_id"):
            expected += 1.0 - (1.0 - SHOT_HIT_PROB) ** len(part)
    photo_targets = set()
    observations = 0
    if not photo.empty:
        for val in photo["covered_targets"]:
            parts = str(val).split(",")
            observations += len(parts)
            photo_targets.update(parts)
    return {
        "plan_name": plan_name,
        "shooting_events": int(len(shooting)),
        "photo_events": int(len(photo)),
        "total_events": int(len(events)),
        "shooting_targets_covered": int(shooting["target_id"].nunique()) if not shooting.empty else 0,
        "photo_targets_covered": int(len(photo_targets)),
        "expected_shooting_hits": float(expected),
        "total_photo_observations": int(observations),
        "multi_target_photo_events": int((photo["num_covered_targets"] > 1).sum()) if not photo.empty else 0,
        "mean_margin": float(events["margin"].mean()),
        "min_margin": float(events["margin"].min()),
        "risk_count": int((events["margin"] < 0.03).sum()),
        "solver_status": "greedy_interval_fallback",
    }


def robustness_table(events: pd.DataFrame) -> pd.DataFrame:
    """Nominal robustness proxy table for selected joint events."""

    rows = []
    if events.empty:
        return pd.DataFrame()
    for _, row in events.iterrows():
        margin = float(row["margin"])
        pass_rate = 1.0 if margin >= 0.08 else 0.75 if margin >= 0.03 else 0.5
        worst = max(0.0, margin - 0.03)
        risk = "高" if pass_rate < 0.7 else "中" if pass_rate < 0.9 else "低"
        rows.append(
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "target_or_targets": row.get("target_id", row.get("covered_targets", "")) if row["event_type"] == "shooting" else row.get("covered_targets", ""),
                "nominal_margin": margin,
                "scenario_pass_rate": pass_rate,
                "worst_margin": worst,
                "risk_level": risk,
                "notes": "proxy robustness from nominal margin; full multi-scenario migration left explicit",
            }
        )
    return pd.DataFrame(rows)


def run_event_task_optimization(traj: pd.DataFrame, targets: pd.DataFrame, out_dir: Path, fov_main: float = 45.0) -> dict[str, pd.DataFrame]:
    """Run pure shooting, pure photo, and joint event-level task plans."""

    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    shooting_targets, photo_targets = split_targets(targets)
    shoot_events = generate_shooting_events(traj, shooting_targets, stride=1)
    pure_shoot, shooting_summary = select_shooting_plan(shoot_events, shooting_targets)
    photo_events = generate_photo_events(traj, photo_targets, fov_degree=fov_main, time_stride=5)
    pure_photo, photo_summary = select_photo_plan(photo_events, photo_targets)
    joint = select_joint_plan(shoot_events, photo_events)
    robustness = robustness_table(joint)

    fov_rows = []
    for fov in [30.0, 45.0, 60.0]:
        events = generate_photo_events(traj, photo_targets, fov_degree=fov, time_stride=5)
        selected, _summary = select_photo_plan(events, photo_targets)
        metrics = plan_metrics(f"pure_photo_fov_{int(fov)}", selected)
        fov_rows.append(
            {
                "fov_degree": fov,
                "selected_events": metrics["photo_events"],
                "covered_targets": metrics["photo_targets_covered"],
                "total_target_observations": metrics["total_photo_observations"],
                "multi_target_events": metrics["multi_target_photo_events"],
                "mean_margin": metrics["mean_margin"],
            }
        )

    comparison = pd.DataFrame(
        [
            plan_metrics("pure_shooting", pure_shoot),
            plan_metrics("pure_photo", pure_photo),
            plan_metrics("joint", joint),
        ]
    )
    shoot_events.to_csv(tables / "shooting_candidate_events.csv", index=False, encoding="utf-8-sig")
    pure_shoot.to_csv(tables / "shooting_selected_events.csv", index=False, encoding="utf-8-sig")
    shooting_summary.to_csv(tables / "shooting_target_summary.csv", index=False, encoding="utf-8-sig")
    photo_events.to_csv(tables / "photo_candidate_events.csv", index=False, encoding="utf-8-sig")
    pure_photo.to_csv(tables / "photo_selected_events.csv", index=False, encoding="utf-8-sig")
    photo_summary.to_csv(tables / "photo_target_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fov_rows).to_csv(tables / "photo_fov_sensitivity.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(tables / "joint_selected_events.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(tables / "task_plan_comparison.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(tables / "joint_task_robustness.csv", index=False, encoding="utf-8-sig")
    _plot_task_outputs(traj, targets, pure_shoot, pure_photo, joint, shooting_summary, figs)
    return {
        "shooting_selected": pure_shoot,
        "shooting_summary": shooting_summary,
        "photo_selected": pure_photo,
        "photo_summary": photo_summary,
        "joint_selected": joint,
        "comparison": comparison,
        "robustness": robustness,
    }


def _save_both(fig: plt.Figure, path_png: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def _plot_task_outputs(traj: pd.DataFrame, targets: pd.DataFrame, shooting: pd.DataFrame, photo: pd.DataFrame, joint: pd.DataFrame, shooting_summary: pd.DataFrame, figs: Path) -> None:
    _setup_font()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not shooting_summary.empty:
        order = shooting_summary.sort_values("target_id")
        ax.bar(order["target_id"], order["hit_probability"], color="#356EA9")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("至少命中一次概率")
    ax.set_title("射击目标命中概率分布")
    ax.tick_params(axis="x", rotation=45)
    _save_both(fig, figs / "shooting_hit_probability_distribution.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    ax.plot(traj[X_COL], traj[Y_COL], color="#9CA3AF", lw=0.9, label="融合轨迹")
    shot_t = targets[targets["任务"] == "射击"]
    photo_t = targets[targets["任务"] == "拍照"]
    ax.scatter(shot_t[X_COL], shot_t[Y_COL], marker="^", color="#B94A48", s=34, label="射击目标")
    ax.scatter(photo_t[X_COL], photo_t[Y_COL], marker="s", color="#356EA9", s=30, label="拍照目标")
    times = traj[TIME_COL].to_numpy(float)
    if not joint.empty:
        for _, ev in joint.iterrows():
            idx = int(np.argmin(np.abs(times - float(ev["execute_time"]))))
            color = "#B94A48" if ev["event_type"] == "shooting" else "#2A9D8F"
            ax.scatter(traj.iloc[idx][X_COL], traj.iloc[idx][Y_COL], marker="*", s=80, color=color, edgecolor="#111827", linewidth=0.3)
            if ev["event_type"] == "photo" and int(ev.get("num_covered_targets", 1)) > 1:
                for tid in str(ev["covered_targets"]).split(","):
                    target = photo_t[photo_t["编号"].astype(str) == tid]
                    if not target.empty:
                        ax.plot([traj.iloc[idx][X_COL], target.iloc[0][X_COL]], [traj.iloc[idx][Y_COL], target.iloc[0][Y_COL]], color="#2A9D8F", lw=0.5, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("联合任务空间分布与多目标同拍连线")
    ax.legend(frameon=False, fontsize=8)
    _save_both(fig, figs / "joint_task_spatial_distribution.png")

    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.16 * len(joint) + 1.5)))
    if not joint.empty:
        labels = []
        y = np.arange(len(joint))[::-1]
        for yy, (_, ev) in zip(y, joint.sort_values("execute_time").iterrows()):
            label = str(ev["target_id"]) if ev["event_type"] == "shooting" else str(ev["covered_targets"])
            labels.append(label)
            color = "#B94A48" if ev["event_type"] == "shooting" else "#2A9D8F"
            ax.barh(yy, float(ev["execute_time"]) - float(ev["start_time"]), left=float(ev["start_time"]), height=0.55, color=color, alpha=0.75)
            ax.plot(float(ev["execute_time"]), yy, marker="*", color="#111827", ms=5)
        ax.set_yticks(y, labels)
    ax.set_xlabel("时间 / s")
    ax.set_title("联合任务事件级 Gantt 图")
    _save_both(fig, figs / "joint_task_gantt.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not photo.empty:
        vals = photo["num_covered_targets"].value_counts().sort_index()
        ax.bar(vals.index.astype(str), vals.values, color="#2A9D8F")
    ax.set_xlabel("单次拍照覆盖目标数")
    ax.set_ylabel("事件数")
    ax.set_title("拍照事件多目标覆盖分布")
    _save_both(fig, figs / "photo_multi_target_coverage.png")
