from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

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
        if t - prev <= 0.25:
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


def _risk_penalty(margin: float) -> float:
    return 1.0 if margin < 0.03 else 0.35 if margin < 0.08 else 0.0


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


def _joint_pool(shoot_events: pd.DataFrame, photo_events: pd.DataFrame, min_margin: float = -np.inf) -> pd.DataFrame:
    shot_pool = shoot_events.copy()
    photo_pool = photo_events.copy()
    if not shot_pool.empty:
        shot_pool["covered_targets"] = shot_pool["target_id"]
        shot_pool["camera_direction"] = np.nan
        shot_pool["num_covered_targets"] = 1
        shot_pool["target_angles"] = ""
    pool = pd.concat([shot_pool, photo_pool], ignore_index=True, sort=False)
    if pool.empty:
        return pool
    pool = pool[pool["margin"].astype(float) >= min_margin].copy()
    pool = pool.sort_values(["execute_time", "event_type", "margin"], ascending=[True, True, False]).reset_index(drop=True)
    return pool


def _select_joint_plan_milp(
    shoot_events: pd.DataFrame,
    photo_events: pd.DataFrame,
    photo_targets: pd.DataFrame,
    max_shots_per_target: int = 5,
    min_margin: float = -np.inf,
    risk_weight: float = 0.20,
    time_limit: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pool = _joint_pool(shoot_events, photo_events, min_margin=min_margin)
    pruned_by_margin = int(len(shoot_events) + len(photo_events) - len(pool))
    if pool.empty:
        return pd.DataFrame(), {
            "solver": "scipy.milp",
            "status": "empty_candidate_pool",
            "objective_value": 0.0,
            "mip_gap": np.nan,
            "runtime_sec": 0.0,
            "n_variables": 0,
            "n_constraints": 0,
            "n_candidate_events": 0,
            "n_time_conflict_edges": 0,
            "n_photo_angle_conflict_edges": 0,
            "n_pruned_by_margin": pruned_by_margin,
            "used_fallback": False,
        }

    n_events = len(pool)
    shooting_target_ids = sorted(shoot_events["target_id"].astype(str).unique()) if not shoot_events.empty else []
    photo_target_ids = photo_targets["编号"].astype(str).tolist()
    y_index: dict[tuple[str, int], int] = {}
    idx = n_events
    for tid in shooting_target_ids:
        for m in range(1, max_shots_per_target + 1):
            y_index[(tid, m)] = idx
            idx += 1
    u_index = {}
    for tid in photo_target_ids:
        u_index[tid] = idx
        idx += 1
    n_vars = idx

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []
    time_conflict_edges = 0
    photo_angle_conflict_edges = 0

    def add_constraint(coeffs: dict[int, float], lower: float, upper: float) -> None:
        row = np.zeros(n_vars)
        for pos, val in coeffs.items():
            row[pos] = val
        rows.append(row)
        lb.append(lower)
        ub.append(upper)

    # Time-window conflicts.
    for i in range(n_events):
        for j in range(i + 1, n_events):
            if _conflicts(float(pool.loc[i, "start_time"]), float(pool.loc[i, "execute_time"]), float(pool.loc[j, "start_time"]), float(pool.loc[j, "execute_time"])):
                add_constraint({i: 1.0, j: 1.0}, 0.0, 1.0)
                time_conflict_edges += 1

    # Shooting count equals selected marginal slots; y_m is ordered.
    for tid in shooting_target_ids:
        event_idxs = pool.index[(pool["event_type"] == "shooting") & (pool["target_id"].astype(str) == tid)].tolist()
        coeff = {i: 1.0 for i in event_idxs}
        for m in range(1, max_shots_per_target + 1):
            coeff[y_index[(tid, m)]] = coeff.get(y_index[(tid, m)], 0.0) - 1.0
        add_constraint(coeff, 0.0, 0.0)
        for m in range(2, max_shots_per_target + 1):
            add_constraint({y_index[(tid, m)]: 1.0, y_index[(tid, m - 1)]: -1.0}, -np.inf, 0.0)

    # Photo coverage variables cannot be set without at least one selected covering event.
    for tid in photo_target_ids:
        cover_idxs = []
        for i, ev in pool[pool["event_type"] == "photo"].iterrows():
            if tid in str(ev["covered_targets"]).split(","):
                cover_idxs.append(i)
        coeff = {u_index[tid]: 1.0}
        for i in cover_idxs:
            coeff[i] = coeff.get(i, 0.0) - 1.0
        add_constraint(coeff, -np.inf, 0.0)

    # Same-target photo angles must differ by at least 60 degrees.
    photo_part = pool[pool["event_type"] == "photo"]
    for tid in photo_target_ids:
        ev_angles = []
        for i, ev in photo_part.iterrows():
            targets = str(ev["covered_targets"]).split(",")
            angles = [float(a) for a in str(ev["target_angles"]).split(",") if a != ""]
            if tid in targets and len(angles) == len(targets):
                ev_angles.append((i, angles[targets.index(tid)]))
        for a_pos, (i, ai) in enumerate(ev_angles):
            for j, aj in ev_angles[a_pos + 1 :]:
                if circular_angle_diff(ai, aj) < 60.0:
                    add_constraint({i: 1.0, j: 1.0}, 0.0, 1.0)
                    photo_angle_conflict_edges += 1

    c = np.zeros(n_vars)
    # SciPy minimizes. Coefficients below maximize the stated weighted utility.
    m1, m2, m3, m4, m5 = 1000.0, 10.0, 2.0, 0.25, risk_weight
    for (tid, m), pos in y_index.items():
        c[pos] = -m1 * SHOT_HIT_PROB * (1.0 - SHOT_HIT_PROB) ** (m - 1)
    for pos in u_index.values():
        c[pos] = -m1
    for i, ev in pool.iterrows():
        margin = float(ev["margin"])
        if ev["event_type"] == "photo":
            covered_count = int(ev.get("num_covered_targets", 1))
            c[i] += -(m2 * covered_count + m3 * max(0, covered_count - 1))
        c[i] += -(m4 * margin) + m5 * _risk_penalty(margin)

    integrality = np.ones(n_vars)
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb), np.asarray(ub)) if rows else None
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    status = {
        "solver": "scipy.milp",
        "status": str(getattr(result, "message", "")) if getattr(result, "success", False) else f"failed:{getattr(result, 'message', '')}",
        "objective_value": float(-result.fun) if getattr(result, "fun", None) is not None else np.nan,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "runtime_sec": float(getattr(result, "time", np.nan)),
        "n_variables": int(n_vars),
        "n_constraints": int(len(rows)),
        "n_candidate_events": int(n_events),
        "n_shooting_events": int((pool["event_type"] == "shooting").sum()),
        "n_photo_events": int((pool["event_type"] == "photo").sum()),
        "n_time_conflict_edges": int(time_conflict_edges),
        "n_photo_angle_conflict_edges": int(photo_angle_conflict_edges),
        "n_pruned_by_margin": int(pruned_by_margin),
        "weight_M1_target_completion": 1000.0,
        "weight_M2_photo_observation": 10.0,
        "weight_M3_multi_target_photo": 2.0,
        "weight_M4_margin": 0.25,
        "weight_M5_risk": float(risk_weight),
        "used_fallback": False,
    }
    if not getattr(result, "success", False) or result.x is None:
        fallback = select_joint_plan(shoot_events, photo_events, max_shots_per_target=max_shots_per_target)
        status["used_fallback"] = True
        return fallback, status

    selected_idx = np.where(result.x[:n_events] > 0.5)[0]
    out = pool.loc[selected_idx].copy().sort_values("execute_time").reset_index(drop=True)
    if not out.empty:
        shot_counts: dict[str, int] = {}
        shot_indices = []
        cum_probs = []
        gains = []
        for _, ev in out.iterrows():
            if ev["event_type"] == "shooting":
                tid = str(ev["target_id"])
                shot_counts[tid] = shot_counts.get(tid, 0) + 1
                m = shot_counts[tid]
                shot_indices.append(m)
                gains.append(SHOT_HIT_PROB * (1.0 - SHOT_HIT_PROB) ** (m - 1))
                cum_probs.append(1.0 - (1.0 - SHOT_HIT_PROB) ** m)
            else:
                shot_indices.append(np.nan)
                gains.append(np.nan)
                cum_probs.append(np.nan)
        out.insert(0, "seq", np.arange(1, len(out) + 1))
        out["shot_index_for_target"] = shot_indices
        out["marginal_hit_gain"] = gains
        out["cumulative_hit_probability"] = cum_probs
    return out, status


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


def plan_metrics(plan_name: str, events: pd.DataFrame, solver_status: str = "heuristic") -> dict[str, object]:
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
            "photo_observations": 0,
            "multi_target_photo_events": 0,
            "combined_target_utility": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "risk_count": 0,
            "solver_status": solver_status,
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
    photo_targets_covered = int(len(photo_targets))
    return {
        "plan_name": plan_name,
        "shooting_events": int(len(shooting)),
        "photo_events": int(len(photo)),
        "total_events": int(len(events)),
        "shooting_targets_covered": int(shooting["target_id"].nunique()) if not shooting.empty else 0,
        "photo_targets_covered": photo_targets_covered,
        "expected_shooting_hits": float(expected),
        "photo_observations": int(observations),
        "multi_target_photo_events": int((photo["num_covered_targets"] > 1).sum()) if not photo.empty else 0,
        "combined_target_utility": float(expected + photo_targets_covered),
        "mean_margin": float(events["margin"].mean()),
        "min_margin": float(events["margin"].min()),
        "risk_count": int((events["margin"] < 0.03).sum()),
        "solver_status": solver_status,
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


def summarize_shooting_targets(shoot_events: pd.DataFrame, selected: pd.DataFrame, shooting_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_shoot = selected[selected["event_type"] == "shooting"] if not selected.empty else pd.DataFrame()
    for tid in shooting_targets["编号"].astype(str):
        feasible = shoot_events[shoot_events["target_id"].astype(str) == tid] if not shoot_events.empty else pd.DataFrame()
        chosen = selected_shoot[selected_shoot["target_id"].astype(str) == tid] if not selected_shoot.empty else pd.DataFrame()
        n = int(len(chosen))
        hit = float(1.0 - (1.0 - SHOT_HIT_PROB) ** n)
        gains = [SHOT_HIT_PROB * (1.0 - SHOT_HIT_PROB) ** (m - 1) for m in range(1, n + 1)]
        if len(feasible) == 0:
            reason = "no_feasible_window"
        elif n == 0 and float(feasible["margin"].max()) < 0.03:
            reason = "low_margin"
        elif n == 0:
            reason = "time_conflict"
        else:
            reason = ""
        min_margin = float(chosen["margin"].min()) if n else np.nan
        rows.append(
            {
                "target_id": tid,
                "feasible_event_count": int(len(feasible)),
                "feasible_time_intervals": _intervals_from_times(feasible["execute_time"].round(1).tolist()) if len(feasible) else "",
                "selected_shots": n,
                "hit_probability": hit,
                "expected_hit_contribution": hit,
                "first_shot_gain": float(gains[0]) if gains else 0.0,
                "last_shot_gain": float(gains[-1]) if gains else 0.0,
                "min_selected_margin": min_margin,
                "risk_level": "高" if n == 0 or (n and min_margin < 0.03) else "中" if n and min_margin < 0.08 else "低",
                "reason_if_unselected": reason,
            }
        )
    return pd.DataFrame(rows)


def summarize_photo_targets(photo_events: pd.DataFrame, selected: pd.DataFrame, photo_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_photo = selected[selected["event_type"] == "photo"] if not selected.empty else pd.DataFrame()
    for tid in photo_targets["编号"].astype(str):
        feasible = photo_events[photo_events["covered_targets"].str.split(",").apply(lambda xs: tid in xs)] if not photo_events.empty else pd.DataFrame()
        chosen = selected_photo[selected_photo["covered_targets"].str.split(",").apply(lambda xs: tid in xs)] if not selected_photo.empty else pd.DataFrame()
        angles = []
        event_ids = []
        multi = False
        for _, ev in chosen.iterrows():
            targets = str(ev["covered_targets"]).split(",")
            vals = [float(a) for a in str(ev["target_angles"]).split(",") if a != ""]
            if tid in targets and len(vals) == len(targets):
                angles.append(vals[targets.index(tid)])
            event_ids.append(str(ev["event_id"]))
            multi = multi or int(ev.get("num_covered_targets", 1)) > 1
        min_gap = min((circular_angle_diff(a, b) for i, a in enumerate(angles) for b in angles[i + 1 :]), default=np.nan)
        if len(feasible) == 0:
            reason = "no_feasible_window"
        elif len(chosen) == 0 and float(feasible["margin"].max()) < 0.03:
            reason = "low_margin"
        elif len(chosen) == 0:
            reason = "time_conflict"
        else:
            reason = ""
        rows.append(
            {
                "target_id": tid,
                "feasible_event_count": int(len(feasible)),
                "selected_photo_count": int(len(chosen)),
                "selected_angles": ",".join(f"{a:.1f}" for a in angles),
                "min_angle_gap": min_gap,
                "covered_by_events": ",".join(event_ids),
                "covered_by_multi_target_event": bool(multi),
                "risk_level": "高" if len(chosen) == 0 else "中" if len(angles) > 1 and min_gap < 60.0 else "低",
                "reason_if_unselected": reason,
            }
        )
    return pd.DataFrame(rows)


def conservative_compare(joint: pd.DataFrame, conservative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, events in [("A_task_utility_first", joint), ("B_margin_first", conservative)]:
        metrics = plan_metrics(name, events, solver_status="scipy.milp")
        robust = robustness_table(events)
        rows.append(
            {
                "plan_name": name,
                "total_events": metrics["total_events"],
                "expected_shooting_hits": metrics["expected_shooting_hits"],
                "photo_targets_covered": metrics["photo_targets_covered"],
                "combined_target_utility": metrics["combined_target_utility"],
                "mean_margin": metrics["mean_margin"],
                "min_margin": metrics["min_margin"],
                "worst_margin": float(robust["worst_margin"].min()) if not robust.empty else 0.0,
                "scenario_pass_rate": float(robust["scenario_pass_rate"].mean()) if not robust.empty else 0.0,
                "risk_count": metrics["risk_count"],
            }
        )
    return pd.DataFrame(rows)


def candidate_generation_stats(traj: pd.DataFrame, shooting_targets: pd.DataFrame, photo_targets: pd.DataFrame, shoot_events: pd.DataFrame, photo_events: pd.DataFrame, solver_status: dict[str, object]) -> pd.DataFrame:
    """Summarize event enumeration and pruning for the paper."""

    n = len(traj)
    shooting_raw_checks = max(0, n - 15) * len(shooting_targets)
    photo_time_points = len(range(5, n, 5))
    photo_raw_checks = photo_time_points * len(photo_targets)
    rows = [
        {
            "stage": "shooting_window_enumeration",
            "raw_checks": int(shooting_raw_checks),
            "feasible_events": int(len(shoot_events)),
            "pruned_count": int(shooting_raw_checks - len(shoot_events)),
            "notes": "target-time checks pruned by distance, speed, acceleration, and 1.5s preparation-window constraints",
        },
        {
            "stage": "photo_window_direction_enumeration",
            "raw_checks": int(photo_raw_checks),
            "feasible_events": int(len(photo_events)),
            "pruned_count": int(photo_raw_checks - len(photo_events)),
            "notes": "target-time checks generate direction events only when distance, FOV, speed, acceleration, and 0.5s window constraints hold",
        },
        {
            "stage": "joint_milp_pool",
            "raw_checks": int(len(shoot_events) + len(photo_events)),
            "feasible_events": int(solver_status.get("n_candidate_events", 0)),
            "pruned_count": int(solver_status.get("n_pruned_by_margin", 0)),
            "notes": "events entering the main MILP after optional margin filtering",
        },
        {
            "stage": "time_conflict_edges",
            "raw_checks": int(solver_status.get("n_candidate_events", 0) * max(0, solver_status.get("n_candidate_events", 0) - 1) / 2),
            "feasible_events": int(solver_status.get("n_time_conflict_edges", 0)),
            "pruned_count": 0,
            "notes": "pairwise event-window overlap constraints q_i + q_j <= 1",
        },
        {
            "stage": "photo_angle_conflict_edges",
            "raw_checks": int(solver_status.get("n_photo_events", 0) * max(0, solver_status.get("n_photo_events", 0) - 1) / 2),
            "feasible_events": int(solver_status.get("n_photo_angle_conflict_edges", 0)),
            "pruned_count": 0,
            "notes": "same-photo-target circular angle differences below 60 degrees",
        },
    ]
    return pd.DataFrame(rows)


def risk_tradeoff_curve(shoot_events: pd.DataFrame, photo_events: pd.DataFrame, photo_targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    configs = [
        ("A0_max_utility", -np.inf, 0.20),
        ("A1_light_margin", 0.01, 0.75),
        ("A2_balanced", 0.02, 1.50),
        ("B_margin_first", 0.03, 3.00),
        ("B2_strict_margin", 0.05, 5.00),
    ]
    rows = []
    plans: dict[str, pd.DataFrame] = {}
    for name, min_margin, risk_weight in configs:
        plan, status = _select_joint_plan_milp(
            shoot_events,
            photo_events,
            photo_targets,
            min_margin=min_margin,
            risk_weight=risk_weight,
            time_limit=45.0,
        )
        metrics = plan_metrics(name, plan, solver_status="scipy.milp" if not status["used_fallback"] else "fallback")
        robust = robustness_table(plan)
        rows.append(
            {
                "plan_name": name,
                "min_margin_filter": min_margin if np.isfinite(min_margin) else -1.0,
                "risk_weight": risk_weight,
                "total_events": metrics["total_events"],
                "expected_shooting_hits": metrics["expected_shooting_hits"],
                "photo_targets_covered": metrics["photo_targets_covered"],
                "combined_target_utility": metrics["combined_target_utility"],
                "mean_margin": metrics["mean_margin"],
                "min_margin": metrics["min_margin"],
                "scenario_pass_rate": float(robust["scenario_pass_rate"].mean()) if not robust.empty else 0.0,
                "risk_count": metrics["risk_count"],
                "objective_value": status["objective_value"],
                "solver_status": status["status"],
            }
        )
        plans[name] = plan
    return pd.DataFrame(rows), plans


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
    joint, solver_status = _select_joint_plan_milp(shoot_events, photo_events, photo_targets, min_margin=-np.inf, risk_weight=0.20)
    conservative, conservative_status = _select_joint_plan_milp(shoot_events, photo_events, photo_targets, min_margin=0.03, risk_weight=3.00)
    robustness = robustness_table(joint)
    shooting_summary = summarize_shooting_targets(shoot_events, joint, shooting_targets)
    photo_summary = summarize_photo_targets(photo_events, joint, photo_targets)
    candidate_stats = candidate_generation_stats(traj, shooting_targets, photo_targets, shoot_events, photo_events, solver_status)
    tradeoff, tradeoff_plans = risk_tradeoff_curve(shoot_events, photo_events, photo_targets)

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
                "total_target_observations": metrics["photo_observations"],
                "multi_target_events": metrics["multi_target_photo_events"],
                "mean_margin": metrics["mean_margin"],
            }
        )

    comparison = pd.DataFrame(
        [
            plan_metrics("pure_shooting", pure_shoot, solver_status="shooting_greedy_slot"),
            plan_metrics("pure_photo", pure_photo, solver_status="photo_greedy_set_packing"),
            plan_metrics("joint", joint, solver_status="scipy.milp" if not solver_status["used_fallback"] else "greedy_interval_fallback"),
        ]
    )
    solver_df = pd.DataFrame([solver_status | {"plan_name": "joint"}, conservative_status | {"plan_name": "joint_conservative"}])
    conservative_df = conservative_compare(joint, conservative)
    solver_weights = pd.DataFrame(
        [
            {
                "parameter": "M1_target_completion",
                "value": 1000.0,
                "role": "expected shooting hits and photo target coverage",
            },
            {
                "parameter": "M2_photo_observation",
                "value": 10.0,
                "role": "additional photo observations after first coverage",
            },
            {
                "parameter": "M3_multi_target_photo",
                "value": 2.0,
                "role": "bonus for one photo covering more than one target",
            },
            {
                "parameter": "M4_margin",
                "value": 0.25,
                "role": "nominal normalized stability margin",
            },
            {
                "parameter": "M5_risk_A",
                "value": 0.20,
                "role": "risk penalty in task-utility-first plan",
            },
            {
                "parameter": "M5_risk_B",
                "value": 3.00,
                "role": "risk penalty in margin-first conservative plan",
            },
        ]
    )
    shoot_events.to_csv(tables / "shooting_candidate_events.csv", index=False, encoding="utf-8-sig")
    pure_shoot.to_csv(tables / "shooting_selected_events.csv", index=False, encoding="utf-8-sig")
    shooting_summary.to_csv(tables / "shooting_target_summary.csv", index=False, encoding="utf-8-sig")
    photo_events.to_csv(tables / "photo_candidate_events.csv", index=False, encoding="utf-8-sig")
    pure_photo.to_csv(tables / "photo_selected_events.csv", index=False, encoding="utf-8-sig")
    photo_summary.to_csv(tables / "photo_target_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fov_rows).to_csv(tables / "photo_fov_sensitivity.csv", index=False, encoding="utf-8-sig")
    candidate_stats.to_csv(tables / "candidate_generation_stats.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(tables / "joint_selected_events.csv", index=False, encoding="utf-8-sig")
    conservative.to_csv(tables / "joint_selected_events_conservative.csv", index=False, encoding="utf-8-sig")
    for name, plan in tradeoff_plans.items():
        plan.to_csv(tables / f"joint_selected_events_{name}.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(tables / "task_plan_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(tables / "task_plan_comparison_v2.csv", index=False, encoding="utf-8-sig")
    solver_df.to_csv(tables / "joint_solver_status.csv", index=False, encoding="utf-8-sig")
    solver_weights.to_csv(tables / "milp_weight_parameters.csv", index=False, encoding="utf-8-sig")
    conservative_df.to_csv(tables / "joint_plan_conservative_compare.csv", index=False, encoding="utf-8-sig")
    tradeoff.to_csv(tables / "joint_risk_tradeoff_curve.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(tables / "joint_task_robustness.csv", index=False, encoding="utf-8-sig")
    _plot_task_outputs(traj, targets, pure_shoot, pure_photo, joint, shooting_summary, figs, tradeoff)
    return {
        "shooting_selected": pure_shoot,
        "shooting_summary": shooting_summary,
        "photo_selected": pure_photo,
        "photo_summary": photo_summary,
        "joint_selected": joint,
        "joint_conservative": conservative,
        "comparison": comparison,
        "robustness": robustness,
        "solver_status": solver_df,
        "conservative_compare": conservative_df,
        "candidate_stats": candidate_stats,
        "risk_tradeoff": tradeoff,
    }


def _save_both(fig: plt.Figure, path_png: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def _plot_task_outputs(traj: pd.DataFrame, targets: pd.DataFrame, shooting: pd.DataFrame, photo: pd.DataFrame, joint: pd.DataFrame, shooting_summary: pd.DataFrame, figs: Path, tradeoff: pd.DataFrame | None = None) -> None:
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

    if tradeoff is not None and not tradeoff.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        colors = np.where(tradeoff["risk_count"].to_numpy(int) > 0, "#B94A48", "#2A9D8F")
        ax.scatter(tradeoff["risk_count"], tradeoff["combined_target_utility"], s=90, c=colors, edgecolor="#111827", linewidth=0.5)
        for _, row in tradeoff.iterrows():
            ax.annotate(str(row["plan_name"]), (row["risk_count"], row["combined_target_utility"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("高风险事件数")
        ax.set_ylabel("综合目标效用")
        ax.set_title("联合方案收益-风险折中曲线")
        ax.grid(alpha=0.2)
        _save_both(fig, figs / "joint_risk_tradeoff_curve.png")
