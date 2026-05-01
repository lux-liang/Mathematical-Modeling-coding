from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from data_io import TIME_COL, X_COL, Y_COL


def _angle_diff(a: float, b: float) -> float:
    raw = abs(float(a) - float(b)) % 360.0
    return min(raw, 360.0 - raw)


def _task_params(task: str) -> tuple[int, float, float, float, float]:
    if task == "射击":
        return 16, 5.0, 30.0, 2.0, 1.5
    return 6, 10.0, 40.0, 1.5, 1.5


def _normalized_margin(
    dwin: np.ndarray,
    speed: np.ndarray,
    acc: np.ndarray,
    dmin: float,
    dmax: float,
    vmax: float,
    amax: float,
) -> tuple[float, str]:
    values = {
        "distance_lower": float(np.min((dwin - dmin) / (dmax - dmin))),
        "distance_upper": float(np.min((dmax - dwin) / (dmax - dmin))),
        "speed": float(np.min((vmax - speed) / vmax)),
        "acceleration": float(np.min((amax - acc) / amax)),
    }
    bottleneck = min(values, key=values.get)
    return float(values[bottleneck]), bottleneck


def _window_ok(df: pd.DataFrame, idx: int, points: int, max_speed: float, max_acc: float, dist: np.ndarray, dmin: float, dmax: float) -> bool:
    start = idx - points + 1
    if start < 0:
        return False
    win = df.iloc[start : idx + 1]
    dwin = dist[start : idx + 1]
    return bool(
        np.all(dwin >= dmin)
        and np.all(dwin <= dmax)
        and np.all(win["speed"].to_numpy() <= max_speed)
        and np.all(win["acceleration"].to_numpy() <= max_acc)
    )


def generate_candidates(traj: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xy = traj[[X_COL, Y_COL]].to_numpy(dtype=float)
    for _, target in targets.iterrows():
        tx, ty = float(target[X_COL]), float(target[Y_COL])
        diff = np.column_stack([tx - xy[:, 0], ty - xy[:, 1]])
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        task = target["任务"]
        window_points, dmin, dmax, vmax, amax = _task_params(task)
        feasible = [
            i for i in range(len(traj))
            if _window_ok(traj, i, window_points, vmax, amax, dist, dmin, dmax)
        ]
        if not feasible:
            continue
        for i in feasible[:: max(1, len(feasible) // 30)]:
            angle = math.degrees(math.atan2(ty - xy[i, 1], tx - xy[i, 0]))
            start = i - window_points + 1
            win = traj.iloc[start : i + 1]
            dwin = dist[start : i + 1]
            margin, bottleneck = _normalized_margin(
                dwin,
                win["speed"].to_numpy(dtype=float),
                win["acceleration"].to_numpy(dtype=float),
                dmin,
                dmax,
                vmax,
                amax,
            )
            if margin < 0:
                continue
            rows.append(
                {
                    "candidate_id": len(rows),
                    "目标编号": target["编号"],
                    "任务": task,
                    "start_time": float(traj.iloc[i - window_points + 1][TIME_COL]),
                    "exec_time": float(traj.iloc[i][TIME_COL]),
                    "distance": float(dist[i]),
                    "angle_deg": float(angle),
                    "speed": float(traj.iloc[i]["speed"]),
                    "acceleration": float(traj.iloc[i]["acceleration"]),
                    "margin": float(margin),
                    "normalized_margin": float(margin),
                    "bottleneck_constraint": bottleneck,
                    "window_points": window_points,
                    "dmin": dmin,
                    "dmax": dmax,
                    "vmax": vmax,
                    "amax": amax,
                }
            )
    return pd.DataFrame(rows)


def _format_selected(selected: list[pd.Series]) -> pd.DataFrame:
    out_rows = []
    for i, cand in enumerate(sorted(selected, key=lambda c: float(c["exec_time"])), 1):
        out_rows.append(
            {
                "序号": i,
                "目标编号": cand["目标编号"],
                "任务": cand["任务"],
                "开始准备时刻(s)": round(float(cand["start_time"]), 2),
                "任务执行时刻(s)": round(float(cand["exec_time"]), 2),
                "距离(m)": round(float(cand["distance"]), 3),
                "方向角(deg)": round(float(cand["angle_deg"]), 2),
                "速度(m/s)": round(float(cand["speed"]), 3),
                "加速度(m/s2)": round(float(cand["acceleration"]), 3),
                "稳定裕度": round(float(cand["margin"]), 3),
            }
        )
    return pd.DataFrame(out_rows)


def select_tasks(candidates: pd.DataFrame, max_tasks: int | None = None, min_gap: float = 0.1) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["序号", "目标编号", "任务", "开始准备时刻(s)", "任务执行时刻(s)"])
    selected = []
    photo_angles: dict[str, list[float]] = {}
    used_shots: set[str] = set()
    ordered = candidates.sort_values(["exec_time", "margin"], ascending=[True, False])
    for _, cand in ordered.iterrows():
        if max_tasks is not None and len(selected) >= max_tasks:
            break
        if selected and cand["start_time"] < selected[-1]["exec_time"] + min_gap:
            continue
        target = str(cand["目标编号"])
        if cand["任务"] == "射击":
            if target in used_shots:
                continue
            used_shots.add(target)
        else:
            angles = photo_angles.setdefault(target, [])
            if any(_angle_diff(float(cand["angle_deg"]), a) < 60.0 for a in angles):
                continue
            angles.append(float(cand["angle_deg"]))
        selected.append(cand)
    return _format_selected(selected)


def optimize_tasks(candidates: pd.DataFrame, max_tasks: int | None = None) -> pd.DataFrame:
    selected, _coverage, _raw = optimize_tasks_with_diagnostics(candidates, max_tasks=max_tasks)
    return selected


def optimize_tasks_with_diagnostics(candidates: pd.DataFrame, max_tasks: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return _format_selected([]), pd.DataFrame(), pd.DataFrame()
    cand = candidates.sort_values(["exec_time", "margin"], ascending=[True, False]).reset_index(drop=True)
    n = len(cand)
    if max_tasks is None:
        max_tasks = n

    target_ids = sorted(cand["目标编号"].astype(str).unique())
    target_var = {tid: n + i for i, tid in enumerate(target_ids)}
    m = n + len(target_ids)

    constraints = []
    lb = []
    ub = []

    row = np.zeros(m)
    row[:n] = 1.0
    constraints.append(row)
    lb.append(0.0)
    ub.append(float(max_tasks))

    for i in range(n):
        for j in range(i + 1, n):
            # Preparing/executing windows cannot overlap.
            if not (cand.loc[i, "exec_time"] < cand.loc[j, "start_time"] or cand.loc[j, "exec_time"] < cand.loc[i, "start_time"]):
                row = np.zeros(m)
                row[i] = 1.0
                row[j] = 1.0
                constraints.append(row)
                lb.append(0.0)
                ub.append(1.0)

    for target, part in cand[cand["任务"] == "射击"].groupby("目标编号"):
        row = np.zeros(m)
        row[part.index.to_numpy()] = 1.0
        constraints.append(row)
        lb.append(0.0)
        ub.append(1.0)

    photo = cand[cand["任务"] == "拍照"]
    for target, part in photo.groupby("目标编号"):
        idxs = part.index.to_list()
        for a, i in enumerate(idxs):
            for j in idxs[a + 1 :]:
                if _angle_diff(float(cand.loc[i, "angle_deg"]), float(cand.loc[j, "angle_deg"])) < 60.0:
                    row = np.zeros(m)
                    row[i] = 1.0
                    row[j] = 1.0
                    constraints.append(row)
                    lb.append(0.0)
                    ub.append(1.0)

    for i, target in enumerate(cand["目标编号"].astype(str)):
        row = np.zeros(m)
        row[i] = 1.0
        row[target_var[target]] = -1.0
        constraints.append(row)
        lb.append(-np.inf)
        ub.append(0.0)
    for target in target_ids:
        idxs = cand.index[cand["目标编号"].astype(str) == target].to_numpy()
        row = np.zeros(m)
        row[target_var[target]] = 1.0
        row[idxs] = -1.0
        constraints.append(row)
        lb.append(-np.inf)
        ub.append(0.0)

    margins = cand["margin"].to_numpy(dtype=float)
    if np.nanmax(margins) > np.nanmin(margins):
        margin_score = (margins - np.nanmin(margins)) / (np.nanmax(margins) - np.nanmin(margins))
    else:
        margin_score = np.zeros_like(margins)
    angle_sector_score = np.where(cand["任务"].to_numpy() == "拍照", 1.0, 0.0)
    # SciPy minimizes; negative coefficients maximize lexicographic-like score.
    c = np.zeros(m)
    c[:n] = -(1_000_000.0 + 1_000.0 * angle_sector_score + margin_score)
    for tid, idx in target_var.items():
        c[idx] = -100_000.0

    integrality = np.ones(m)
    bounds = Bounds(np.zeros(m), np.ones(m))
    lc = LinearConstraint(np.vstack(constraints), np.asarray(lb), np.asarray(ub))
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=lc, options={"time_limit": 30})
    if not result.success or result.x is None:
        fallback = select_tasks(candidates, max_tasks=max_tasks)
        return fallback, pd.DataFrame(), pd.DataFrame()
    selected_idx = np.where(result.x[:n] > 0.5)[0]
    selected = [cand.loc[i] for i in selected_idx]
    coverage_rows = []
    for target in target_ids:
        idxs = cand.index[cand["目标编号"].astype(str) == target].to_numpy()
        num_sel = int(np.sum(result.x[idxs] > 0.5))
        task_type = str(cand.loc[idxs[0], "任务"]) if len(idxs) else ""
        z_val = int(result.x[target_var[target]] > 0.5)
        coverage_rows.append(
            {
                "target_id": target,
                "task_type": task_type,
                "z_k": z_val,
                "number_selected_candidates": num_sel,
                "covered_or_not": bool(num_sel > 0),
                "z_matches_selection": bool(z_val == (num_sel > 0)),
            }
        )
    raw = cand.copy()
    raw["selected"] = False
    raw.loc[selected_idx, "selected"] = True
    return _format_selected(selected), pd.DataFrame(coverage_rows), raw


def verify_selected_tasks(traj: pd.DataFrame, targets: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    if tasks.empty:
        return pd.DataFrame()
    rows = []
    target_lookup = targets.set_index(["编号", "任务"])
    xy = traj[[X_COL, Y_COL]].to_numpy(dtype=float)
    times = traj[TIME_COL].to_numpy(dtype=float)
    prev_exec = -np.inf
    photo_angles: dict[str, list[float]] = {}
    shot_seen: set[str] = set()
    for _, task in tasks.sort_values("任务执行时刻(s)").iterrows():
        target_id = str(task["目标编号"])
        task_type = str(task["任务"])
        tx = float(target_lookup.loc[(target_id, task_type), X_COL])
        ty = float(target_lookup.loc[(target_id, task_type), Y_COL])
        exec_time = float(task["任务执行时刻(s)"])
        idx = int(np.argmin(np.abs(times - exec_time)))
        if task_type == "射击":
            window_points, dmin, dmax, vmax, amax = 16, 5.0, 30.0, 2.0, 1.5
        else:
            window_points, dmin, dmax, vmax, amax = 6, 10.0, 40.0, 1.5, 1.5
        start = idx - window_points + 1
        dist = np.sqrt((xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2)
        if start < 0:
            win = traj.iloc[0 : idx + 1]
            dwin = dist[0 : idx + 1]
        else:
            win = traj.iloc[start : idx + 1]
            dwin = dist[start : idx + 1]
        angle = math.degrees(math.atan2(ty - xy[idx, 1], tx - xy[idx, 0]))
        pass_distance = bool(len(dwin) == window_points and np.all(dwin >= dmin) and np.all(dwin <= dmax))
        pass_speed = bool(len(win) == window_points and np.all(win["speed"].to_numpy() <= vmax))
        pass_acc = bool(len(win) == window_points and np.all(win["acceleration"].to_numpy() <= amax))
        if task_type == "射击":
            pass_angle = target_id not in shot_seen
            shot_seen.add(target_id)
        else:
            angles = photo_angles.setdefault(target_id, [])
            pass_angle = not any(_angle_diff(angle, a) < 60.0 for a in angles)
            angles.append(angle)
        prep_start = float(times[start]) if start >= 0 else float(times[0])
        pass_conflict = bool(prep_start > prev_exec or not np.isfinite(prev_exec))
        prev_exec = max(prev_exec, exec_time)
        margin, bottleneck = _normalized_margin(
            dwin,
            win["speed"].to_numpy(dtype=float),
            win["acceleration"].to_numpy(dtype=float),
            dmin,
            dmax,
            vmax,
            amax,
        )
        rows.append(
            {
                "target_id": target_id,
                "task_type": task_type,
                "time": exec_time,
                "prep_start": prep_start,
                "distance_min_window": float(np.min(dwin)),
                "distance_max_window": float(np.max(dwin)),
                "speed_max_window": float(win["speed"].max()),
                "acc_max_window": float(win["acceleration"].max()),
                "angle_deg": angle,
                "pass_distance": pass_distance,
                "pass_speed": pass_speed,
                "pass_acc": pass_acc,
                "pass_angle": pass_angle,
                "pass_conflict": pass_conflict,
                "stability_margin": margin,
                "bottleneck_constraint": bottleneck,
                "pass_all": bool(pass_distance and pass_speed and pass_acc and pass_angle and pass_conflict),
            }
        )
    return pd.DataFrame(rows)


def optimize_with_verification(candidates: pd.DataFrame, traj: pd.DataFrame, targets: pd.DataFrame, max_tasks: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = candidates.copy()
    for _ in range(10):
        selected = optimize_tasks(working, max_tasks=max_tasks)
        verification = verify_selected_tasks(traj, targets, selected)
        if verification.empty or verification["pass_all"].all():
            return selected, verification
        bad_keys = set(zip(verification.loc[~verification["pass_all"], "target_id"], verification.loc[~verification["pass_all"], "time"].round(2)))
        mask = [
            (str(row["目标编号"]), round(float(row["exec_time"]), 2)) not in bad_keys
            for _, row in working.iterrows()
        ]
        working = working.loc[mask].reset_index(drop=True)
    selected = optimize_tasks(working, max_tasks=max_tasks)
    return selected, verify_selected_tasks(traj, targets, selected)


def compare_task_solutions(greedy: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    def metrics(df: pd.DataFrame, name: str) -> dict[str, object]:
        if df.empty:
            return {"solution": name, "task_count": 0, "covered_targets": 0, "photo_count": 0, "stability_margin_sum": 0.0}
        return {
            "solution": name,
            "task_count": int(len(df)),
            "covered_targets": int(df["目标编号"].nunique()),
            "photo_count": int((df["任务"] == "拍照").sum()),
            "stability_margin_sum": float(df.get("稳定裕度", pd.Series(dtype=float)).sum()),
        }
    return pd.DataFrame([metrics(greedy, "greedy"), metrics(optimized, "optimized")])


def photo_angle_check(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    photo = tasks[tasks["任务"] == "拍照"].copy() if not tasks.empty else pd.DataFrame()
    for target, part in photo.groupby("目标编号"):
        part = part.sort_values("任务执行时刻(s)").reset_index(drop=True)
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                theta_i = float(part.loc[i, "方向角(deg)"])
                theta_j = float(part.loc[j, "方向角(deg)"])
                diff = _angle_diff(theta_i, theta_j)
                rows.append(
                    {
                        "target_id": target,
                        "selected_task_i": int(part.loc[i, "序号"]),
                        "selected_task_j": int(part.loc[j, "序号"]),
                        "theta_i": theta_i,
                        "theta_j": theta_j,
                        "circular_angle_diff": diff,
                        "pass_or_not": bool(diff >= 60.0),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "target_id",
            "selected_task_i",
            "selected_task_j",
            "theta_i",
            "theta_j",
            "circular_angle_diff",
            "pass_or_not",
        ],
    )
