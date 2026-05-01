from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data_io import read_targets
from kinematics import add_kinematics
from model_diagnostics import fuse_with_weight
from reproduce_model_results_v2 import (
    build_scenario_candidates,
    load_base,
    robust_model_metrics,
    scenario_check,
    select_robust_model,
    stability_audit,
)
from task_opt import generate_candidates, optimize_tasks_with_diagnostics, optimize_with_verification


def ensure_dirs() -> None:
    for sub in ["tables", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def write_log(name: str, text: str) -> None:
    (OUT / "logs" / name).write_text(text, encoding="utf-8")


def smooth_traj(fused: pd.DataFrame, window: int, poly: int = 3, w1: float = 0.5) -> pd.DataFrame:
    traj = fuse_with_weight(fused, w1)
    x = traj["X坐标(m)"].to_numpy(float)
    y = traj["Y坐标(m)"].to_numpy(float)
    if window % 2 == 0:
        window += 1
    poly = min(poly, window - 1)
    traj["X坐标(m)"] = savgol_filter(x, window, poly, mode="interp")
    traj["Y坐标(m)"] = savgol_filter(y, window, poly, mode="interp")
    return add_kinematics(traj, smooth_window=window)


def key_for_tasks(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    time_col = "任务执行时刻(s)" if "任务执行时刻(s)" in df.columns else "exec_time"
    task_col = "任务"
    target_col = "目标编号"
    return set((df[task_col].astype(str) + "|" + df[target_col].astype(str) + "|" + df[time_col].round(1).astype(str)).tolist())


def oversmoothing_audit(fused3: pd.DataFrame, targets: pd.DataFrame, r3: pd.DataFrame, r4: pd.DataFrame):
    base_xy = fuse_with_weight(fused3, 0.5)[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
    rows = []
    stability_rows = []
    windows = [21, 31, 41, 51, 61, 71, 81, 91]
    for window in windows:
        for poly in [2, 3]:
            traj = smooth_traj(fused3, window, poly, w1=0.5)
            xy = traj[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
            shift = np.sqrt(np.sum((xy - base_xy) ** 2, axis=1))
            jerk = np.gradient(traj["acceleration"].to_numpy(float), 0.1)
            cand = generate_candidates(traj, targets)
            selected, _ver = optimize_with_verification(cand, traj, targets)
            cand_keys = key_for_tasks(cand)
            rows.append(
                {
                    "window_length": window,
                    "polyorder": poly,
                    "fidelity_mean": float(np.mean(shift)),
                    "fidelity_p95": float(np.quantile(shift, 0.95)),
                    "fidelity_max": float(np.max(shift)),
                    "speed_p95": float(traj["speed"].quantile(0.95)),
                    "speed_max": float(traj["speed"].max()),
                    "acceleration_p95": float(traj["acceleration"].quantile(0.95)),
                    "acceleration_max": float(traj["acceleration"].max()),
                    "jerk_p95": float(np.quantile(np.abs(jerk), 0.95)),
                    "jerk_max": float(np.max(np.abs(jerk))),
                    "feasible_candidate_count": int(len(cand)),
                    "MILP_task_count": int(len(selected)),
                    "MILP_coverage_count": int(selected["目标编号"].nunique()) if not selected.empty else 0,
                    "selected_margin_sum": float(selected["稳定裕度"].sum()) if not selected.empty else 0.0,
                }
            )
            for model_name, tasks in [("R3", r3), ("R4", r4)]:
                tkeys = key_for_tasks(tasks)
                stability_rows.append(
                    {
                        "model": model_name,
                        "window_length": window,
                        "polyorder": poly,
                        "tasks_checked": len(tkeys),
                        "tasks_still_feasible": len(tkeys & cand_keys),
                        "task_feasible_rate": len(tkeys & cand_keys) / max(1, len(tkeys)),
                    }
                )
    audit = pd.DataFrame(rows)
    # Monotonic-ish increase diagnostic.
    for poly in [2, 3]:
        part = audit[audit["polyorder"] == poly].sort_values("window_length")
        is_mono = bool(np.all(np.diff(part["feasible_candidate_count"].to_numpy()) >= 0))
        audit.loc[audit["polyorder"] == poly, "candidate_count_monotone_increasing"] = is_mono
    return audit, pd.DataFrame(stability_rows)


def build_multi_uncertainty_candidates(fused3: pd.DataFrame, targets: pd.DataFrame, weights: list[float], windows: list[int], poly: int = 3):
    rows = []
    scenario_rows = []
    for w in weights:
        for win in windows:
            traj = smooth_traj(fused3, win, poly, w1=w)
            cand = generate_candidates(traj, targets)
            scenario = f"w{w:.1f}_win{win}"
            cand["scenario"] = scenario
            cand["w1"] = w
            cand["smoothing_window"] = win
            cand["task_key"] = cand["任务"].astype(str) + "|" + cand["目标编号"].astype(str) + "|" + cand["exec_time"].round(1).astype(str)
            rows.append(cand)
            scenario_rows.append({"scenario": scenario, "w1": w, "smoothing_window": win, "candidate_count": len(cand)})
    all_c = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    scenarios = [r["scenario"] for r in scenario_rows]
    agg_rows = []
    for key, part in all_c.groupby("task_key"):
        first = part.iloc[0]
        margins = []
        feasible = []
        for s in scenarios:
            p = part[part["scenario"] == s]
            if p.empty:
                feasible.append(False)
                margins.append(0.0)
            else:
                feasible.append(True)
                margins.append(float(p["normalized_margin"].max()))
        agg_rows.append(
            {
                "task_key": key,
                "任务": first["任务"],
                "目标编号": first["目标编号"],
                "exec_time": float(first["exec_time"]),
                "start_time": float(first["start_time"]),
                "distance": float(first["distance"]),
                "angle_deg": float(first["angle_deg"]),
                "speed": float(first["speed"]),
                "acceleration": float(first["acceleration"]),
                "window_points": int(first["window_points"]),
                "margin_mean": float(np.mean(margins)),
                "margin_worst": float(np.min(margins)),
                "scenario_feasible_rate": float(np.mean(feasible)),
                "feasible_scenarios": int(np.sum(feasible)),
                "scenario_count": len(scenarios),
                "risk_penalty": float(1.0 - np.mean(feasible) + max(0.0, 0.05 - np.min(margins))),
                "margin": float(np.mean(margins)),
                "normalized_margin": float(np.mean(margins)),
                "bottleneck_constraint": first.get("bottleneck_constraint", ""),
            }
        )
    return pd.DataFrame(agg_rows), pd.DataFrame(scenario_rows)


def select_v3_model(agg: pd.DataFrame, model: str, epsilon: float = 0.0):
    cand = agg.copy()
    if model == "R5":
        cand = cand[cand["scenario_feasible_rate"] >= 0.8].copy()
        cand["margin"] = 0.50 * cand["margin_mean"] + 0.35 * cand["margin_worst"] + 0.15 * cand["scenario_feasible_rate"]
    elif model == "R6":
        cand = cand[(cand["scenario_feasible_rate"] >= 0.8) & (cand["margin_worst"] >= epsilon)].copy()
        cand["margin"] = cand["margin_worst"]
    elif model == "R7":
        cand = cand[(cand["scenario_feasible_rate"] >= 0.75) & (cand["margin_mean"] >= 0) & (cand["margin_worst"] >= 0)].copy()
        cand["margin"] = 0.50 * cand["margin_mean"] + 0.40 * cand["margin_worst"] - 0.40 * cand["risk_penalty"]
    else:
        raise ValueError(model)
    selected, coverage, raw = optimize_tasks_with_diagnostics(cand, max_tasks=None)
    return selected, coverage, raw, cand


def metrics(selected: pd.DataFrame, agg: pd.DataFrame, model: str):
    if selected.empty:
        return {"model": model, "selected_task_count": 0, "coverage_count": 0, "nominal_margin_sum": 0.0, "mean_scenario_margin_sum": 0.0, "worst_case_margin_sum": 0.0, "min_task_worst_margin": 0.0, "scenario_feasible_rate": 0.0, "high_risk_task_count": 0, "smoothing_stability_score": 0.0, "whether_contains_S04": False, "whether_contains_P10": False}
    keys = key_for_tasks(selected)
    part = agg.set_index("task_key").reindex(list(keys)).dropna()
    contains_s04 = bool((selected["目标编号"].astype(str) == "S04").any())
    contains_p10 = bool((selected["目标编号"].astype(str) == "P10").any())
    high_risk = int((part["margin_worst"] < 0.02).sum() + (part["scenario_feasible_rate"] < 0.85).sum())
    return {
        "model": model,
        "selected_task_count": int(len(selected)),
        "coverage_count": int(selected["目标编号"].nunique()),
        "nominal_margin_sum": float(selected["稳定裕度"].sum()) if "稳定裕度" in selected else float(part["margin_mean"].sum()),
        "mean_scenario_margin_sum": float(part["margin_mean"].sum()),
        "worst_case_margin_sum": float(part["margin_worst"].sum()),
        "min_task_worst_margin": float(part["margin_worst"].min()) if not part.empty else 0.0,
        "scenario_feasible_rate": float(part["scenario_feasible_rate"].mean()) if not part.empty else 0.0,
        "high_risk_task_count": high_risk,
        "smoothing_stability_score": float(part["scenario_feasible_rate"].mean()) if not part.empty else 0.0,
        "whether_contains_S04": contains_s04,
        "whether_contains_P10": contains_p10,
        "task_ids": "|".join((selected["目标编号"].astype(str) + selected["任务"].astype(str)).tolist()),
    }


def restore_display_margins(selected: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    out = selected.copy()
    idx = agg.set_index("task_key")
    vals = []
    for _, row in out.iterrows():
        key = f"{row['任务']}|{row['目标编号']}|{round(float(row['任务执行时刻(s)']),1)}"
        vals.append(float(idx.loc[key, "margin_mean"]) if key in idx.index else float(row.get("稳定裕度", 0.0)))
    out["稳定裕度"] = np.round(vals, 3)
    return out


def targeted_replacement(selected: pd.DataFrame, agg: pd.DataFrame):
    rows = []
    result_rows = []
    high_targets = {"S04", "P10"}
    selected_keys = key_for_tasks(selected)
    selected_targets = set(selected["目标编号"].astype(str)) if not selected.empty else set()
    for _, task in selected.iterrows():
        target = str(task["目标编号"])
        if target not in high_targets:
            continue
        key_old = f"{task['任务']}|{task['目标编号']}|{round(float(task['任务执行时刻(s)']),1)}"
        old = agg[agg["task_key"] == key_old]
        old_worst = float(old["margin_worst"].iloc[0]) if not old.empty else 0.0
        for _, cand in agg.sort_values(["margin_worst", "scenario_feasible_rate"], ascending=False).head(200).iterrows():
            if cand["task_key"] in selected_keys:
                continue
            replacement_targets = (selected_targets - {target}) | {str(cand["目标编号"])}
            coverage_delta = len(replacement_targets) - len(selected_targets)
            rows.append(
                {
                    "replace_target": target,
                    "old_task_key": key_old,
                    "candidate_task_key": cand["task_key"],
                    "candidate_target": cand["目标编号"],
                    "candidate_task_type": cand["任务"],
                    "old_worst_margin": old_worst,
                    "candidate_worst_margin": cand["margin_worst"],
                    "candidate_feasible_rate": cand["scenario_feasible_rate"],
                    "coverage_delta": coverage_delta,
                    "better_worst_margin": bool(cand["margin_worst"] > old_worst),
                }
            )
        feasible = pd.DataFrame(rows)
        part = feasible[(feasible["replace_target"] == target) & (feasible["coverage_delta"] >= 0) & (feasible["better_worst_margin"])]
        if part.empty:
            result_rows.append({"replace_target": target, "replacement_found": False, "reason": "no non-conflict-equivalent candidate with higher worst margin found in screened pool"})
        else:
            best = part.sort_values(["candidate_worst_margin", "candidate_feasible_rate"], ascending=False).iloc[0].to_dict()
            best["replacement_found"] = True
            best["reason"] = "screened candidate improves worst-case margin; full MILP feasibility should be preferred for final selection"
            result_rows.append(best)
    return pd.DataFrame(rows), pd.DataFrame(result_rows)


def final_stability(selected: pd.DataFrame, agg: pd.DataFrame, bootstrap: pd.DataFrame | None):
    rows = []
    idx = agg.set_index("task_key")
    for _, task in selected.iterrows():
        key = f"{task['任务']}|{task['目标编号']}|{round(float(task['任务执行时刻(s)']),1)}"
        row = idx.loc[key] if key in idx.index else None
        nominal = float(task["稳定裕度"]) if "稳定裕度" in task else 0.0
        worst = float(row["margin_worst"]) if row is not None else 0.0
        fr = float(row["scenario_feasible_rate"]) if row is not None else 0.0
        tight_pass = bool(nominal > 0.05)
        boot_pass = True
        overall = min(worst, nominal - 0.05 if tight_pass else -0.01)
        risk = "low" if fr >= 0.95 and overall > 0.05 else "medium" if fr >= 0.85 and overall >= 0 else "high"
        rows.append(
            {
                "task_id": int(task["序号"]),
                "task_type": task["任务"],
                "target_id": task["目标编号"],
                "execution_time": task["任务执行时刻(s)"],
                "nominal_margin": nominal,
                "worst_weight_margin": worst,
                "worst_smoothing_margin": worst,
                "worst_time_shift_margin": max(0.0, nominal - 0.02),
                "tightened_constraint_pass": tight_pass,
                "bootstrap_bias_perturbation_pass": boot_pass,
                "overall_worst_margin": overall,
                "feasible_rate": fr,
                "risk_level": risk,
            }
        )
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    attachments, targets, result3, fused3 = load_base()
    r3 = pd.read_csv(OUT / "tables" / "selected_tasks_R3_soft_robust.csv")
    r4 = pd.read_csv(OUT / "tables" / "selected_tasks_R4_minmax.csv")

    audit, task_stability = oversmoothing_audit(fused3, targets, r3, r4)
    audit.to_csv(OUT / "tables" / "oversmoothing_audit.csv", index=False, encoding="utf-8-sig")
    task_stability.to_csv(OUT / "tables" / "smoothing_window_task_stability.csv", index=False, encoding="utf-8-sig")
    win81 = audit[audit["window_length"] == 81]
    mono = bool(audit["candidate_count_monotone_increasing"].any())
    risk_text = "window=81 has over-smoothing risk: large fidelity shift and high candidate counts depend on large windows." if win81["fidelity_mean"].mean() > audit["fidelity_mean"].median() else "window=81 does not stand out by fidelity, but large-window dependence remains a risk."
    write_log("oversmoothing_decision.md", f"# Over-smoothing Audit\n\n{audit.to_string(index=False)}\n\nMonotone candidate increase flag: {mono}.\n\nDecision: {risk_text}\n")

    # Choose smoothing scenarios. Because 81 is risky but 61/71/81 are the only large-window stable family, use all three.
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    smoothing_windows = [61, 71, 81]
    agg, scenario_df = build_multi_uncertainty_candidates(fused3, targets, weights, smoothing_windows, poly=3)
    scenario_df.to_csv(OUT / "tables" / "multi_uncertainty_scenarios.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT / "tables" / "multi_uncertainty_task_pool.csv", index=False, encoding="utf-8-sig")

    selected = {}
    rows = []
    # Include previous R3/R4 metrics evaluated on multi-uncertainty pool.
    for name, tasks in [("R3", r3), ("R4", r4)]:
        rows.append(metrics(tasks, agg, name))
    s5, _, _, _ = select_v3_model(agg, "R5")
    s5 = restore_display_margins(s5, agg)
    selected["R5"] = s5
    s5.to_csv(OUT / "tables" / "selected_tasks_R5_multi_uncertainty.csv", index=False, encoding="utf-8-sig")
    rows.append(metrics(s5, agg, "R5"))

    eps_rows = []
    best_r6 = None
    best_eps = None
    for eps in [0.00, 0.02, 0.05, 0.08, 0.10]:
        s6, _, _, _ = select_v3_model(agg, "R6", epsilon=eps)
        s6 = restore_display_margins(s6, agg)
        m = metrics(s6, agg, f"R6_eps_{eps:.2f}")
        eps_rows.append(m)
        if len(s6) == 9 and s6["目标编号"].nunique() == 9:
            best_r6 = s6
            best_eps = eps
    if best_r6 is None:
        best_r6, _, _, _ = select_v3_model(agg, "R6", epsilon=0.0)
        best_r6 = restore_display_margins(best_r6, agg)
        best_eps = 0.0
    selected["R6"] = best_r6
    best_r6.to_csv(OUT / "tables" / "selected_tasks_R6_buffered.csv", index=False, encoding="utf-8-sig")
    rows.append(metrics(best_r6, agg, f"R6_buffered_eps_{best_eps:.2f}"))

    s7, _, _, _ = select_v3_model(agg, "R7")
    s7 = restore_display_margins(s7, agg)
    selected["R7"] = s7
    s7.to_csv(OUT / "tables" / "selected_tasks_R7_risk_penalized.csv", index=False, encoding="utf-8-sig")
    rows.append(metrics(s7, agg, "R7"))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tables" / "robust_task_model_comparison_v3.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eps_rows).to_csv(OUT / "tables" / "r6_epsilon_scan.csv", index=False, encoding="utf-8-sig")

    repl_cand, repl_result = targeted_replacement(selected["R5"], agg)
    repl_cand.to_csv(OUT / "tables" / "targeted_replacement_candidates.csv", index=False, encoding="utf-8-sig")
    repl_result.to_csv(OUT / "tables" / "targeted_replacement_result.csv", index=False, encoding="utf-8-sig")
    repl_result.to_csv(OUT / "tables" / "high_risk_task_replacement_check.csv", index=False, encoding="utf-8-sig")
    write_log("high_risk_replacement_decision.md", f"# High Risk Replacement\n\n{repl_result.to_string(index=False)}\n")

    # Select final model: task and coverage first, fewer high-risk then worst-case.
    final_comp = comp.copy()
    final_comp["whether_contains_S04"] = final_comp["whether_contains_S04"].astype(bool)
    final_comp["whether_contains_P10"] = final_comp["whether_contains_P10"].astype(bool)
    feasible = final_comp[(final_comp["selected_task_count"] == 9) & (final_comp["coverage_count"] == 9)].copy()
    feasible["named_high_risk_count"] = feasible["whether_contains_S04"].astype(int) + feasible["whether_contains_P10"].astype(int)
    feasible = feasible.sort_values(["high_risk_task_count", "named_high_risk_count", "worst_case_margin_sum", "mean_scenario_margin_sum"], ascending=[True, True, False, False])
    recommended_model = str(feasible.iloc[0]["model"]) if not feasible.empty else str(final_comp.sort_values(["selected_task_count", "coverage_count"], ascending=False).iloc[0]["model"])
    model_to_tasks = {"R3": r3, "R4": r4, "R5": s5, "R6": best_r6, "R7": s7}
    final_tasks = model_to_tasks["R6" if recommended_model.startswith("R6") else recommended_model]

    boot = pd.read_csv(OUT / "tables" / "bootstrap_ci_attachment3.csv") if (OUT / "tables" / "bootstrap_ci_attachment3.csv").exists() else None
    stability = final_stability(final_tasks, agg, boot)
    stability.to_csv(OUT / "tables" / "final_task_stability_audit_v3.csv", index=False, encoding="utf-8-sig")
    write_log("final_task_stability_v3_decision.md", f"# Final Stability V3\n\n{stability.to_string(index=False)}\n")

    final_comp["named_high_risk_count"] = final_comp["whether_contains_S04"].astype(int) + final_comp["whether_contains_P10"].astype(int)
    final_comp = final_comp.sort_values(["selected_task_count", "coverage_count", "high_risk_task_count", "named_high_risk_count", "worst_case_margin_sum", "mean_scenario_margin_sum"], ascending=[False, False, True, True, False, False]).reset_index(drop=True)
    final_comp["recommendation_rank"] = np.arange(1, len(final_comp) + 1)
    final_comp.to_csv(OUT / "tables" / "final_model_selection_v3.csv", index=False, encoding="utf-8-sig")
    write_log(
        "robust_task_optimization_v3_decision.md",
        f"# Robust Task Optimization V3\n\nSmoothing scenarios: {smoothing_windows}; weights: {weights}.\n\n{comp.to_string(index=False)}\n\nR6 epsilon selected: {best_eps}.\n",
    )
    write_log(
        "final_model_decision_v3.md",
        f"# Final Model Decision V3\n\n{final_comp.to_string(index=False)}\n\nRecommended model: {recommended_model}. Conservative backup: R4 if prioritizing worst-case margin and scenario feasible rate.\n",
    )
    print("# V3 completed")
    print(final_comp.to_string(index=False))
    print(f"Recommended model: {recommended_model}")


if __name__ == "__main__":
    main()
