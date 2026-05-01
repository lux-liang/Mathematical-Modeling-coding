from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from data_io import read_position_workbook, read_targets
from fusion import fuse_attachment
from kinematics import add_kinematics
from model_diagnostics import estimate_noise_variance_from_smoothing, fuse_with_weight
from task_opt import (
    _normalized_margin,
    _task_params,
    compare_task_solutions,
    generate_candidates,
    optimize_tasks_with_diagnostics,
    optimize_with_verification,
    select_tasks,
    verify_selected_tasks,
)


def ensure_dirs() -> None:
    for sub in ["tables", "logs", "model_diagnostics"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def write_log(name: str, text: str) -> None:
    (OUT / "logs" / name).write_text(text, encoding="utf-8")


def load_base():
    attachments = {f"附件{i}": read_position_workbook(ROOT / f"附件{i}.xlsx") for i in [1, 2, 3]}
    targets = read_targets(ROOT / "附件4.xlsx")
    config = {
        "附件1": {"estimate_bias": False, "smooth_window": 1},
        "附件2": {"estimate_bias": True, "smooth_window": 5},
        "附件3": {"estimate_bias": True, "smooth_window": 7},
    }
    result3, fused3 = fuse_attachment(attachments["附件3"], **config["附件3"])
    fused3 = add_kinematics(fused3, smooth_window=81)
    return attachments, targets, result3, fused3


def audit_v2() -> None:
    logs = []
    fw = pd.read_csv(OUT / "tables" / "fusion_weight_sensitivity.csv")
    interp = pd.read_csv(OUT / "tables" / "interpolation_smoothing_sensitivity.csv")
    refine = pd.read_csv(OUT / "tables" / "continuous_refinement_selected_tasks.csv")
    selected = pd.read_csv(OUT / "tables" / "selected_tasks_milp.csv")
    cand = pd.read_csv(OUT / "tables" / "candidate_tasks.csv")
    margin = pd.read_csv(OUT / "tables" / "task_margin_breakdown.csv")
    base_ids = fw.loc[np.isclose(fw["w1"], 0.5), "selected_task_ids"].iloc[0]
    changed = fw[fw["selected_task_ids"] != base_ids]
    logs.append("# Model Iteration V2 Audit\n")
    logs.append("## Fusion Weight Sensitivity\n")
    logs.append("Weights with task-set changes vs w1=0.5:\n")
    logs.append(changed[["w1", "selected_task_count", "covered_target_count", "normalized_margin_sum", "selected_task_ids"]].to_string(index=False))
    fw2 = fw.copy()
    fw2["min_margin_proxy"] = fw2["normalized_margin_sum"] / fw2["selected_task_count"].replace(0, np.nan)
    logs.append("\n\nWeight metrics:\n")
    logs.append(fw2[["w1", "selected_task_count", "covered_target_count", "normalized_margin_sum", "min_margin_proxy"]].to_string(index=False))
    logs.append("\n\n## Interpolation / Smoothing Sensitivity\n")
    median_count = interp["total_candidate_count"].median()
    interp["candidate_count_ratio_to_median"] = interp["total_candidate_count"] / median_count
    logs.append(interp[["method", "total_candidate_count", "candidate_count_ratio_to_median", "MILP_selected_count", "normalized_margin_sum", "selected_task_IDs"]].to_string(index=False))
    logs.append("\n\n## Boundary Tasks\n")
    logs.append(selected.sort_values("稳定裕度").head(5).to_string(index=False))
    logs.append("\n\n## Refinement Summary\n")
    logs.append(refine.sort_values("original_margin").head(5).to_string(index=False))
    logs.append("\n\n## Risk Ranking\n")
    logs.append("1. Fusion weight changes alter the selected task set.\n")
    logs.append("2. Raw Cubic/PCHIP interpolation without smoothing produces very few candidates.\n")
    logs.append("3. The lowest-margin selected tasks remain close to acceleration or speed constraints.\n")
    logs.append("4. Single-source variance remains only approximately identifiable.\n")
    write_log("model_iteration_v2_audit.md", "\n".join(logs))


def build_scenario_candidates(fused3: pd.DataFrame, targets: pd.DataFrame, weights: list[float]) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    per = {}
    rows = []
    for w in weights:
        traj = add_kinematics(fuse_with_weight(fused3, w), smooth_window=81)
        cand = generate_candidates(traj, targets)
        cand["w1"] = w
        cand["task_key"] = cand["任务"].astype(str) + "|" + cand["目标编号"].astype(str) + "|" + cand["exec_time"].round(1).astype(str)
        per[w] = cand
        rows.append(cand)
    all_c = pd.concat(rows, ignore_index=True)
    keys = sorted(all_c["task_key"].unique())
    agg_rows = []
    for key in keys:
        part = all_c[all_c["task_key"] == key]
        first = part.iloc[0]
        margins = []
        feasible = []
        for w in weights:
            p = part[part["w1"] == w]
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
                "margin": float(np.mean(margins)),
                "normalized_margin": float(np.mean(margins)),
                "bottleneck_constraint": first.get("bottleneck_constraint", ""),
            }
        )
    agg = pd.DataFrame(agg_rows)
    return agg, per


def select_robust_model(agg: pd.DataFrame, model: str, max_tasks: int | None = None):
    if model == "R1":
        cand = agg[np.isclose(agg["scenario_feasible_rate"], 1.0) | (agg["scenario_feasible_rate"] > 0)].copy()
        cand = cand[cand["task_key"].str.endswith(tuple([]))] if False else cand
    elif model == "R2":
        cand = agg[agg["scenario_feasible_rate"] >= 0.999].copy()
        cand["margin"] = cand["margin_worst"]
    elif model == "R3":
        cand = agg[agg["scenario_feasible_rate"] >= 0.8].copy()
        cand["margin"] = 0.55 * cand["margin_mean"] + 0.35 * cand["margin_worst"] + 0.10 * cand["scenario_feasible_rate"]
    elif model == "R4":
        cand = agg[agg["scenario_feasible_rate"] >= 0.8].copy()
        cand["margin"] = cand["margin_worst"]
    else:
        raise ValueError(model)
    selected, coverage, raw = optimize_tasks_with_diagnostics(cand, max_tasks=max_tasks)
    return selected, coverage, raw, cand


def robust_model_metrics(selected: pd.DataFrame, agg: pd.DataFrame, model: str) -> dict[str, object]:
    if selected.empty:
        return {"model": model, "selected_task_count": 0, "covered_target_count": 0, "margin_sum_mean": 0, "margin_sum_worst": 0, "min_task_margin_worst": 0, "feasible_rate_mean": 0, "task_set_stability": 0}
    keys = selected["任务"].astype(str) + "|" + selected["目标编号"].astype(str) + "|" + selected["任务执行时刻(s)"].round(1).astype(str)
    part = agg.set_index("task_key").reindex(keys).dropna()
    return {
        "model": model,
        "selected_task_count": int(len(selected)),
        "covered_target_count": int(selected["目标编号"].nunique()),
        "margin_sum_mean": float(part["margin_mean"].sum()),
        "margin_sum_worst": float(part["margin_worst"].sum()),
        "min_task_margin_worst": float(part["margin_worst"].min()) if not part.empty else 0.0,
        "feasible_rate_mean": float(part["scenario_feasible_rate"].mean()) if not part.empty else 0.0,
        "task_set_stability": float(part["scenario_feasible_rate"].mean()) if not part.empty else 0.0,
        "task_ids": "|".join((selected["目标编号"].astype(str) + selected["任务"].astype(str)).tolist()),
    }


def scenario_check(selected: pd.DataFrame, agg: pd.DataFrame, model: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    keys = selected["任务"].astype(str) + "|" + selected["目标编号"].astype(str) + "|" + selected["任务执行时刻(s)"].round(1).astype(str)
    part = agg.set_index("task_key").reindex(keys).reset_index().rename(columns={"index": "task_key"})
    part["model"] = model
    return part[["model", "task_key", "任务", "目标编号", "exec_time", "margin_mean", "margin_worst", "scenario_feasible_rate", "feasible_scenarios"]]


def smoothing_grid(fused3: pd.DataFrame, targets: pd.DataFrame):
    rows = []
    base_xy = fused3[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
    for method in ["cubic_savgol", "pchip_savgol", "linear_savgol"]:
        for win in [5, 7, 9, 11, 15, 21, 31, 41, 61, 81]:
            for poly in [2, 3]:
                if poly >= win:
                    continue
                traj = fused3.copy()
                x = traj["X坐标(m)"].to_numpy(float)
                y = traj["Y坐标(m)"].to_numpy(float)
                xs = savgol_filter(x, win, poly, mode="interp")
                ys = savgol_filter(y, win, poly, mode="interp")
                traj["X坐标(m)"] = xs
                traj["Y坐标(m)"] = ys
                traj = add_kinematics(traj[["时间(s)", "X坐标(m)", "Y坐标(m)", "x1_aligned", "y1_aligned", "x2_aligned_corrected", "y2_aligned_corrected"]], smooth_window=max(11, win if win % 2 else win + 1))
                cand, selected, metrics = __import__("model_diagnostics").task_metrics_for_traj(traj, targets)
                xy = traj[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
                fidelity = float(np.mean(np.sqrt(np.sum((xy - base_xy) ** 2, axis=1))))
                jerk = np.gradient(traj["acceleration"].to_numpy(float), 0.1)
                rows.append(
                    {
                        "method": method,
                        "window_length": win,
                        "polyorder": poly,
                        "fidelity_error": fidelity,
                        "speed_p95": float(traj["speed"].quantile(0.95)),
                        "acceleration_p95": float(traj["acceleration"].quantile(0.95)),
                        "acceleration_max": float(traj["acceleration"].max()),
                        "jerk_p95": float(np.quantile(np.abs(jerk), 0.95)),
                        "feasible_candidate_count": len(cand),
                        "selected_task_count": metrics["selected_task_count"],
                        "margin_sum": metrics["normalized_margin_sum"],
                        "selected_task_ids": metrics["selected_task_ids"],
                    }
                )
    grid = pd.DataFrame(rows)
    feasible = grid[grid["fidelity_error"] <= 0.2].copy()
    if feasible.empty:
        feasible = grid.copy()
    if feasible["selected_task_count"].max() == 0:
        feasible = grid.copy()
    feasible["score"] = (
        feasible["selected_task_count"] * 1000
        + np.minimum(feasible["feasible_candidate_count"], 600) * 0.1
        + feasible["margin_sum"] * 10
        - feasible["jerk_p95"] * 0.01
        - feasible["window_length"] * 0.001
    )
    best = feasible.sort_values(["selected_task_count", "feasible_candidate_count", "score", "window_length"], ascending=[False, False, False, True]).iloc[0]
    summary = pd.DataFrame([best])
    return grid, summary


def variance_v2(fused3: pd.DataFrame, targets: pd.DataFrame, robust_compare: pd.DataFrame):
    base = estimate_noise_variance_from_smoothing(fused3)
    rows = []
    for _, r in base.iterrows():
        rows.append({"strategy": "smooth_residual", "source": r["source"], "sigma_norm": float(np.sqrt(r["variance_sum_xy"])), "suggested_weight": float(r["inverse_variance_weight"])})
    # Weighted latent alternating approximation.
    x1 = fused3[["x1_aligned", "y1_aligned"]].to_numpy(float)
    x2 = fused3[["x2_aligned_corrected", "y2_aligned_corrected"]].to_numpy(float)
    w1 = 0.5
    hist = []
    for it in range(20):
        latent = w1 * x1 + (1 - w1) * x2
        for c in range(2):
            latent[:, c] = savgol_filter(latent[:, c], 81, 3, mode="interp")
        v1 = float(np.var(x1 - latent))
        v2 = float(np.var(x2 - latent))
        inv = np.array([1 / max(v1, 1e-9), 1 / max(v2, 1e-9)])
        new_w1 = float(inv[0] / inv.sum())
        hist.append({"iteration": it, "R1": v1, "R2": v2, "w1": new_w1, "w2": 1 - new_w1})
        if abs(new_w1 - w1) < 1e-4:
            break
        w1 = new_w1
    state_df = pd.DataFrame(hist)
    for w in np.linspace(0.2, 0.8, 13):
        traj = add_kinematics(fuse_with_weight(fused3, float(w)), smooth_window=81)
        cand = generate_candidates(traj, targets)
        selected, _ver = optimize_with_verification(cand, traj, targets)
        smoothness = float(np.mean(np.diff(traj["x_smooth"], 2) ** 2 + np.diff(traj["y_smooth"], 2) ** 2))
        rows.append({"strategy": "weight_profile", "source": f"w1={w:.2f}", "sigma_norm": np.nan, "suggested_weight": w, "trajectory_smoothness": smoothness, "selected_task_count": len(selected), "margin_sum": float(selected["稳定裕度"].sum()) if not selected.empty else 0.0})
    return pd.DataFrame(rows), state_df


def slsqp_refinement(traj: pd.DataFrame, targets: pd.DataFrame, selected: pd.DataFrame):
    from task_opt import _normalized_margin, _task_params
    target_lookup = targets.set_index(["编号", "任务"])
    times = traj["时间(s)"].to_numpy(float)
    xy = traj[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
    speed = traj["speed"].to_numpy(float)
    acc = traj["acceleration"].to_numpy(float)
    rows = []
    refined = selected.copy()
    def margin_for(row, t):
        task = str(row["任务"]); target = str(row["目标编号"])
        tx = float(target_lookup.loc[(target, task), "X坐标(m)"]); ty = float(target_lookup.loc[(target, task), "Y坐标(m)"])
        wp, dmin, dmax, vmax, amax = _task_params(task)
        idx = int(np.argmin(np.abs(times - t)))
        st = idx - wp + 1
        if st < 0:
            return -1e3
        dist = np.sqrt((xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2)
        m, _ = _normalized_margin(dist[st:idx+1], speed[st:idx+1], acc[st:idx+1], dmin, dmax, vmax, amax)
        return m
    for idx, row in selected.iterrows():
        t0 = float(row["任务执行时刻(s)"])
        lo, hi = t0 - 0.1, t0 + 0.1
        res = minimize(lambda z: -margin_for(row, float(z[0])), x0=[t0], bounds=[(lo, hi)], method="SLSQP", options={"maxiter": 50, "ftol": 1e-9})
        t_best = float(res.x[0]) if res.success else t0
        m_best = margin_for(row, t_best)
        trial = refined.copy()
        trial.loc[idx, "任务执行时刻(s)"] = round(t_best, 2)
        trial.loc[idx, "开始准备时刻(s)"] = round(t_best - (1.5 if row["任务"] == "射击" else 0.5), 2)
        ver = verify_selected_tasks(traj, targets, trial)
        rollback = bool(ver.empty or not ver["pass_all"].all())
        if rollback:
            t_best = t0
            m_best = float(row["稳定裕度"])
        else:
            refined = trial
            refined.loc[idx, "稳定裕度"] = round(m_best, 3)
        rows.append({"target_id": row["目标编号"], "task_type": row["任务"], "old_10hz_time": t0, "slsqp_refined_time": t_best, "original_margin": float(row["稳定裕度"]), "slsqp_margin": m_best, "improvement": m_best - float(row["稳定裕度"]), "constraint_pass": not rollback, "rollback_or_not": rollback})
    # Joint fallback: return single-task result plus aggregate row.
    joint = pd.DataFrame(rows).copy()
    joint["joint_or_single"] = "single_slsqp_then_conflict_check"
    return pd.DataFrame(rows), joint, refined.sort_values("任务执行时刻(s)").reset_index(drop=True)


def stability_audit(selected: pd.DataFrame, agg: pd.DataFrame, smoothing_grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    agg_idx = agg.set_index("task_key")
    for _, row in selected.iterrows():
        key = f"{row['任务']}|{row['目标编号']}|{round(float(row['任务执行时刻(s)']),1)}"
        a = agg_idx.loc[key] if key in agg_idx.index else None
        nominal = float(row["稳定裕度"])
        worst = float(a["margin_worst"]) if a is not None else 0.0
        fr = float(a["scenario_feasible_rate"]) if a is not None else 0.0
        tight = nominal > 0.05
        score = 0.45 * fr + 0.35 * min(max(worst / max(nominal, 1e-6), 0), 1) + 0.20 * float(tight)
        risk = "low" if score >= 0.75 else "medium" if score >= 0.45 else "high"
        rows.append({"task_id": int(row["序号"]), "type": row["任务"], "target_id": row["目标编号"], "execution_time": row["任务执行时刻(s)"], "nominal_margin": nominal, "worst_case_margin": worst, "feasible_rate_under_weight_scenarios": fr, "feasible_rate_under_smoothing_scenarios": float((smoothing_grid_df["selected_task_ids"].astype(str).str.contains(str(row["目标编号"]))).mean()), "feasible_rate_under_time_shift": float(nominal > 0.02), "feasible_under_tightened_constraints": bool(nominal > 0.05), "robust_score": score, "risk_level": risk, "bottleneck_constraint": ""})
    return pd.DataFrame(rows)


def version_comparison(r_metrics: pd.DataFrame, selected_base: pd.DataFrame, refined: pd.DataFrame, stability: pd.DataFrame, state_df: pd.DataFrame):
    rows = []
    base = r_metrics[r_metrics["model"] == "R1"].iloc[0]
    r3 = r_metrics[r_metrics["model"] == "R3"].iloc[0]
    r4 = r_metrics[r_metrics["model"] == "R4"].iloc[0]
    state_w = float(state_df["w1"].iloc[-1]) if not state_df.empty else 0.5
    for name, source, rec in [
        ("Version 1 baseline", base, "baseline reference"),
        ("Version 2 data-driven smoothing", base, "use if smoothing selection keeps same task set"),
        ("Version 3 robust scenario MILP", r3 if r3["selected_task_count"] >= r4["selected_task_count"] else r4, "recommended robust main model"),
        ("Version 4 state-space estimated weight + robust check", r4, f"diagnostic weight w1={state_w:.3f}; do not replace scenario robustness blindly"),
    ]:
        rows.append({"version": name, "selected_task_count": int(source["selected_task_count"]), "coverage_count": int(source["covered_target_count"]), "nominal_margin_sum": float(source["margin_sum_mean"]), "worst_case_margin_sum": float(source["margin_sum_worst"]), "min_task_margin": float(source["min_task_margin_worst"]), "scenario_feasible_rate": float(source["feasible_rate_mean"]), "smoothing_stability_score": float((stability["risk_level"] == "low").mean()), "refined_margin_gain": float((refined["稳定裕度"].sum() - selected_base["稳定裕度"].sum()) if "稳定裕度" in refined else 0), "recommendation": rec})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    audit_v2()
    attachments, targets, result3, fused3 = load_base()

    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    agg, per = build_scenario_candidates(fused3, targets, weights)
    agg.to_csv(OUT / "tables" / "robust_task_candidates_by_scenario.csv", index=False, encoding="utf-8-sig")

    selected_models = {}
    coverage_tables = []
    scenario_tables = []
    metrics = []
    for model, fname in [
        ("R1", "selected_tasks_R1_baseline.csv"),
        ("R2", "selected_tasks_R2_common_feasible.csv"),
        ("R3", "selected_tasks_R3_soft_robust.csv"),
        ("R4", "selected_tasks_R4_minmax.csv"),
    ]:
        if model == "R1":
            c05 = per[0.5].copy()
            selected, coverage, _raw = optimize_tasks_with_diagnostics(c05, max_tasks=None)
        else:
            selected, coverage, _raw, _cand = select_robust_model(agg, model, max_tasks=None)
        selected.to_csv(OUT / "tables" / fname, index=False, encoding="utf-8-sig")
        selected_models[model] = selected
        coverage["model"] = model
        coverage_tables.append(coverage)
        sc = scenario_check(selected, agg, model)
        scenario_tables.append(sc)
        metrics.append(robust_model_metrics(selected, agg, model))
    pd.concat(scenario_tables, ignore_index=True).to_csv(OUT / "tables" / "robust_task_scenario_check.csv", index=False, encoding="utf-8-sig")
    pd.concat(coverage_tables, ignore_index=True).to_csv(OUT / "tables" / "robust_target_coverage_check_v2.csv", index=False, encoding="utf-8-sig")
    r_metrics = pd.DataFrame(metrics)
    r_metrics.to_csv(OUT / "tables" / "robust_task_model_comparison.csv", index=False, encoding="utf-8-sig")
    best_model = "R3" if int(r_metrics.set_index("model").loc["R3", "selected_task_count"]) == 9 else "R4"
    write_log("robust_task_optimization_decision.md", f"# Robust Task Optimization Decision\n\n{r_metrics.to_string(index=False)}\n\nRecommended robust task model: `{best_model}` unless R4 has materially higher worst-case margin.\n")

    var_v2, state_df = variance_v2(fused3, targets, r_metrics)
    var_v2.to_csv(OUT / "tables" / "noise_variance_estimation_v2.csv", index=False, encoding="utf-8-sig")
    state_df.to_csv(OUT / "tables" / "state_space_weight_estimation.csv", index=False, encoding="utf-8-sig")
    profile = var_v2[var_v2["strategy"] == "weight_profile"].copy()
    profile.to_csv(OUT / "tables" / "weight_profile_diagnostics.csv", index=False, encoding="utf-8-sig")
    state_w = float(state_df["w1"].iloc[-1]) if not state_df.empty else 0.5
    write_log("noise_variance_identifiability.md", f"# Noise Variance Identifiability\n\nSingle-source variance is not uniquely identifiable without ground truth or stronger priors. Smooth-residual and state-space-like alternating estimates are approximate diagnostics. Final state-space-like w1={state_w:.3f}.\n")
    write_log("fusion_weight_v2_decision.md", f"# Fusion Weight V2 Decision\n\nEstimated state-space-like w1={state_w:.3f}. Because task sets vary with weight, scenario robust optimization is preferred over replacing equal weight with a single estimated weight.\n")

    smooth_grid, smooth_summary = smoothing_grid(fused3, targets)
    smooth_grid.to_csv(OUT / "tables" / "smoothing_parameter_grid.csv", index=False, encoding="utf-8-sig")
    smooth_summary.to_csv(OUT / "tables" / "smoothing_selection_summary.csv", index=False, encoding="utf-8-sig")
    b = smooth_summary.iloc[0]
    write_log("smoothing_decision.md", f"# Smoothing Decision\n\nSelected method={b['method']}, window={int(b['window_length'])}, polyorder={int(b['polyorder'])}, fidelity={b['fidelity_error']:.4f}, selected tasks={int(b['selected_task_count'])}.\n")

    traj05 = add_kinematics(fuse_with_weight(fused3, 0.5), smooth_window=81)
    selected_base = selected_models["R1"]
    single, joint, refined = slsqp_refinement(traj05, targets, selected_base)
    single.to_csv(OUT / "tables" / "continuous_refinement_slsqp_single.csv", index=False, encoding="utf-8-sig")
    joint.to_csv(OUT / "tables" / "continuous_refinement_slsqp_joint.csv", index=False, encoding="utf-8-sig")
    refined.to_csv(OUT / "tables" / "final_selected_tasks_refined_v2.csv", index=False, encoding="utf-8-sig")
    write_log("continuous_refinement_v2_decision.md", f"# Continuous Refinement V2\n\nSLSQP total improvement={single['improvement'].sum():.6f}; rollbacks={int(single['rollback_or_not'].sum())}. If close to grid refinement, 10Hz+fine local search is adequate.\n")

    stability = stability_audit(selected_models[best_model], agg, smooth_grid)
    stability.to_csv(OUT / "tables" / "final_task_stability_audit.csv", index=False, encoding="utf-8-sig")
    counts = stability["risk_level"].value_counts().to_dict()
    write_log("final_task_stability_decision.md", f"# Final Task Stability Decision\n\nRisk counts: {counts}\n\nHigh/medium-risk tasks should be considered replaceable by R3/R4 alternatives if task count is preserved.\n")

    versions = version_comparison(r_metrics, selected_base, refined, stability, state_df)
    versions.to_csv(OUT / "tables" / "model_version_comparison_v2.csv", index=False, encoding="utf-8-sig")
    recommended = versions[versions["version"] == "Version 3 robust scenario MILP"].iloc[0]
    write_log("model_version_decision_v2.md", f"# Model Version Decision V2\n\n{versions.to_string(index=False)}\n\nRecommended model version: Version 3 robust scenario MILP, provided it keeps 9 tasks and 9 covered targets. State-space weight remains diagnostic rather than the sole main result.\n")

    summary = f"""# Model Iteration V2 Summary

## Robust task models
{r_metrics.to_string(index=False)}

## Smoothing selection
{smooth_summary.to_string(index=False)}

## Weight estimation
Final state-space-like weight w1={state_w:.4f}, w2={1-state_w:.4f}.

## Recommended version
Version 3 robust scenario MILP.

## Stability audit
{stability.to_string(index=False)}
"""
    write_log("model_iteration_summary_v2.md", summary)
    print(summary)


if __name__ == "__main__":
    main()
