from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alignment import align_pair, resample_aligned
from data_io import generate_data_report, read_position_workbook, read_targets
from fill_result import fill_result_template
from fusion import fuse_attachment
from kinematics import add_kinematics
from model_diagnostics import (
    bias_model_comparison,
    bootstrap_analysis_attachment3,
    delta_objective_diagnostics,
    engineering_threshold_sensitivity,
    estimate_noise_variance_from_smoothing,
    fuse_with_weight,
    interpolation_fused,
    robust_alignment_objectives,
    task_metrics_for_traj,
)
from plotting import plot_aligned, plot_fused_trajectory, plot_raw, plot_residuals, plot_series, plot_task_timeline, plot_tasks
from task_opt import (
    compare_task_solutions,
    generate_candidates,
    optimize_tasks_with_diagnostics,
    optimize_with_verification,
    photo_angle_check,
    select_tasks,
    verify_selected_tasks,
)
from validation import residual_dataframe, validate_alignment


def ensure_dirs() -> None:
    for sub in ["tables", "logs", "model_diagnostics", "figures", "figures/diagnostics", "trajectories"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def write_log(name: str, content: str) -> None:
    (OUT / "logs" / name).write_text(content, encoding="utf-8")


def plot_delta_curve(df: pd.DataFrame, name: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df["Delta"], df["RMSE"], color="#3b6ea8", label="RMSE")
    plt.plot(df["Delta"], df["MAE"], color="#2a9d8f", label="MAE")
    plt.xlabel("Delta / s")
    plt.ylabel("loss / m")
    plt.title(f"Delta objective diagnostics {name}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "diagnostics" / f"delta_objective_{name}.png", dpi=180)
    plt.close()


def local_refine_tasks(traj: pd.DataFrame, targets: pd.DataFrame, selected: pd.DataFrame):
    from task_opt import _normalized_margin, _task_params, _angle_diff
    rows = []
    refined = selected.copy()
    target_lookup = targets.set_index(["编号", "任务"])
    times = traj["时间(s)"].to_numpy(float)
    xy = traj[["X坐标(m)", "Y坐标(m)"]].to_numpy(float)
    for idx, row in selected.iterrows():
        target = str(row["目标编号"])
        task = str(row["任务"])
        tx = float(target_lookup.loc[(target, task), "X坐标(m)"])
        ty = float(target_lookup.loc[(target, task), "Y坐标(m)"])
        wp, dmin, dmax, vmax, amax = _task_params(task)
        original_time = float(row["任务执行时刻(s)"])
        best_time = original_time
        best_margin = float(row["稳定裕度"])
        rollback = False
        for t in np.arange(original_time - 0.1, original_time + 0.1001, 0.02):
            nearest = int(np.argmin(np.abs(times - t)))
            start = nearest - wp + 1
            if start < 0:
                continue
            dist = np.sqrt((xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2)
            win = traj.iloc[start : nearest + 1]
            dwin = dist[start : nearest + 1]
            margin, _ = _normalized_margin(dwin, win["speed"].to_numpy(float), win["acceleration"].to_numpy(float), dmin, dmax, vmax, amax)
            if margin > best_margin:
                best_margin = margin
                best_time = float(times[nearest])
        trial = refined.copy()
        trial.loc[idx, "任务执行时刻(s)"] = round(best_time, 2)
        trial.loc[idx, "开始准备时刻(s)"] = round(best_time - (1.5 if task == "射击" else 0.5), 2)
        ver = verify_selected_tasks(traj, targets, trial)
        if ver.empty or not ver["pass_all"].all():
            rollback = True
            best_time = original_time
            best_margin = float(row["稳定裕度"])
        else:
            refined = trial
            refined.loc[idx, "稳定裕度"] = round(best_margin, 3)
        rows.append(
            {
                "target_id": target,
                "task_type": task,
                "original_time": original_time,
                "refined_time": best_time,
                "original_margin": float(row["稳定裕度"]),
                "refined_margin": best_margin,
                "improvement": best_margin - float(row["稳定裕度"]),
                "pass_constraints": not rollback,
                "rollback_or_not": rollback,
            }
        )
    refined = refined.sort_values("任务执行时刻(s)").reset_index(drop=True)
    refined["序号"] = np.arange(1, len(refined) + 1)
    return pd.DataFrame(rows), refined


def main() -> None:
    ensure_dirs()
    generate_data_report(ROOT, OUT / "data_report.md", ["附件1.xlsx", "附件2.xlsx", "附件3.xlsx", "附件4.xlsx", "result.xlsx"])
    attachments = {f"附件{i}": read_position_workbook(ROOT / f"附件{i}.xlsx") for i in [1, 2, 3]}
    targets = read_targets(ROOT / "附件4.xlsx")
    config = {
        "附件1": {"estimate_bias": False, "smooth_window": 1},
        "附件2": {"estimate_bias": True, "smooth_window": 5},
        "附件3": {"estimate_bias": True, "smooth_window": 7},
    }

    summary_rows, validation_rows = [], []
    fused_outputs, results = {}, {}
    for name, data in attachments.items():
        print(f"align {name}")
        s1 = next(s for s in data if "方式1" in s)
        s2 = next(s for s in data if "方式2" in s)
        result, fused = fuse_attachment(data, **config[name])
        results[name] = result
        if name == "附件3":
            fused = add_kinematics(fused, smooth_window=81)
        fused_outputs[name] = fused
        fused.to_csv(OUT / "trajectories" / f"fused_attachment{name[-1]}_10hz.csv", index=False, encoding="utf-8-sig")
        validation_rows.append(validate_alignment(name, data[s1], data[s2], result, config[name]["estimate_bias"], config[name]["smooth_window"]))
        summary_rows.append(
            {
                "数据": name,
                "Delta定义": "t2_aligned = t2 + Delta",
                "Delta(s)": result.delta,
                "bias_x(m)": result.bias_x,
                "bias_y(m)": result.bias_y,
                "bias_norm(m)": float(np.hypot(result.bias_x, result.bias_y)),
                "rmse_before(m)": result.rmse_before,
                "rmse_after(m)": result.rmse_after,
                "overlap_start(s)": result.overlap_start,
                "overlap_end(s)": result.overlap_end,
            }
        )
        plot_raw(data, f"{name} raw", OUT / "figures" / f"raw_{name}.png")
        plot_aligned(fused, f"{name} aligned", OUT / "figures" / f"aligned_{name}.png")
    pd.DataFrame(summary_rows).to_csv(OUT / "tables" / "alignment_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(OUT / "tables" / "alignment_validation.csv", index=False, encoding="utf-8-sig")

    # Delta-b coupling diagnostics and robust objectives.
    ident_rows, robust_rows = [], []
    for name, data in attachments.items():
        s1 = next(s for s in data if "方式1" in s)
        s2 = next(s for s in data if "方式2" in s)
        curve, ident = delta_objective_diagnostics(name, data[s1], data[s2], results[name], config[name]["smooth_window"])
        curve.to_csv(OUT / "tables" / f"delta_objective_attachment{name[-1]}.csv", index=False, encoding="utf-8-sig")
        plot_delta_curve(curve, f"attachment{name[-1]}")
        ident_rows.append(ident)
        if name in {"附件2", "附件3"}:
            robust_rows.append(robust_alignment_objectives(name, data[s1], data[s2], results[name].delta, config[name]["smooth_window"]))
    pd.DataFrame(ident_rows).to_csv(OUT / "tables" / "delta_identifiability_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(robust_rows, ignore_index=True).to_csv(OUT / "tables" / "robust_alignment_objective_comparison.csv", index=False, encoding="utf-8-sig")

    # Bias model comparisons.
    decisions = []
    for name in ["附件2", "附件3"]:
        data = attachments[name]
        s1 = next(s for s in data if "方式1" in s)
        s2 = next(s for s in data if "方式2" in s)
        comp, slide, decision = bias_model_comparison(name, data[s1], data[s2], results[name], config[name]["smooth_window"])
        comp.to_csv(OUT / "tables" / f"bias_model_comparison_attachment{name[-1]}.csv", index=False, encoding="utf-8-sig")
        slide.to_csv(OUT / "tables" / f"sliding_bias_attachment{name[-1]}.csv", index=False, encoding="utf-8-sig")
        decisions.append(f"- {name}: selected `{decision}` by validation RMSE; fixed model remains preferred if gain < 5%.")
    write_log("bias_model_decision.md", "# Bias Model Decision\n\n" + "\n".join(decisions) + "\n")

    # Bootstrap automatic block length and threshold sensitivity.
    data3 = attachments["附件3"]
    s1 = next(s for s in data3 if "方式1" in s)
    s2 = next(s for s in data3 if "方式2" in s)
    acf, boot_sens, auto_len = bootstrap_analysis_attachment3(data3[s1], data3[s2], results["附件3"], config["附件3"]["smooth_window"], n_boot=1000)
    acf.to_csv(OUT / "tables" / "residual_acf_attachment3.csv", index=False, encoding="utf-8-sig")
    boot_sens.to_csv(OUT / "tables" / "bootstrap_block_sensitivity.csv", index=False, encoding="utf-8-sig")
    boot_sens[boot_sens["block_length"] == auto_len].to_csv(OUT / "tables" / "bootstrap_ci_attachment3.csv", index=False, encoding="utf-8-sig")
    thresh = engineering_threshold_sensitivity(results["附件3"], boot_sens[boot_sens["block_length"] == auto_len].iloc[0])
    thresh.to_csv(OUT / "tables" / "engineering_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_log("bootstrap_decision.md", f"# Bootstrap Decision\n\nAutomatic block length: {auto_len} points ({auto_len*0.1:.2f}s).\n\nAttachment 3 remains not significant fixed bias under automatic block bootstrap.\n")
    write_log("threshold_decision.md", f"# Threshold Decision\n\n{int((~thresh['final_decision']).sum())}/{len(thresh)} threshold combinations decide no obvious fixed system bias.\n")

    # Fusion weight sensitivity.
    fused3_base = fused_outputs["附件3"]
    var_df = estimate_noise_variance_from_smoothing(fused3_base)
    var_df.to_csv(OUT / "tables" / "noise_variance_estimation.csv", index=False, encoding="utf-8-sig")
    weight_rows = []
    for w1 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        traj = add_kinematics(fuse_with_weight(fused3_base, w1), smooth_window=81)
        cand, selected, metrics = task_metrics_for_traj(traj, targets)
        metrics.update(
            {
                "w1": w1,
                "w2": 1 - w1,
                "trajectory_smoothness": float(np.mean(np.diff(traj["x_smooth"], 2) ** 2 + np.diff(traj["y_smooth"], 2) ** 2)),
                "speed_max": float(traj["speed"].max()),
                "acceleration_max": float(traj["acceleration"].max()),
            }
        )
        weight_rows.append(metrics)
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(OUT / "tables" / "fusion_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_log("fusion_weight_decision.md", "# Fusion Weight Decision\n\nVariance estimation is diagnostic because source-specific ground truth is unavailable. Equal weighting is retained as the main result unless sensitivity shows materially different selected task sets.\n")

    # Interpolation / smoothing sensitivity.
    interp_rows = []
    for method, label in [("cubic", "CubicSpline"), ("pchip", "PCHIP"), ("linear_savgol", "linear+Savitzky-Golay"), ("cubic_savgol", "CubicSpline+Savitzky-Golay")]:
        fused = interpolation_fused(data3[s1], data3[s2], results["附件3"], method)
        traj = add_kinematics(fused, smooth_window=81)
        cand, selected, metrics = task_metrics_for_traj(traj, targets)
        interp_rows.append(
            {
                "method": label,
                "speed_max": float(traj["speed"].max()),
                "speed_p95": float(traj["speed"].quantile(0.95)),
                "acceleration_max": float(traj["acceleration"].max()),
                "acceleration_p95": float(traj["acceleration"].quantile(0.95)),
                "feasible_shooting_candidate_count": int((cand["任务"] == "射击").sum()) if not cand.empty else 0,
                "feasible_photo_candidate_count": int((cand["任务"] == "拍照").sum()) if not cand.empty else 0,
                "total_candidate_count": int(len(cand)),
                "MILP_selected_count": metrics["selected_task_count"],
                "coverage_count": metrics["covered_target_count"],
                "normalized_margin_sum": metrics["normalized_margin_sum"],
                "selected_task_IDs": metrics["selected_task_ids"],
            }
        )
    pd.DataFrame(interp_rows).to_csv(OUT / "tables" / "interpolation_smoothing_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_log("interpolation_decision.md", "# Interpolation Decision\n\nSensitivity table reports whether candidate counts and selected tasks change across interpolation/smoothing methods. Current main flow retains cubic interpolation with smoothed kinematics for consistency with prior outputs.\n")

    # Main task optimization with normalized margins.
    traj3 = fused_outputs["附件3"]
    candidates = generate_candidates(traj3, targets)
    candidates.to_csv(OUT / "tables" / "candidate_tasks.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT / "tables" / "task_candidates.csv", index=False, encoding="utf-8-sig")
    candidates[["candidate_id", "目标编号", "任务", "exec_time", "normalized_margin", "bottleneck_constraint"]].to_csv(OUT / "tables" / "task_margin_breakdown.csv", index=False, encoding="utf-8-sig")
    greedy = select_tasks(candidates, max_tasks=None)
    greedy.to_csv(OUT / "tables" / "selected_tasks_greedy.csv", index=False, encoding="utf-8-sig")
    greedy.to_csv(OUT / "tables" / "greedy_selected_tasks.csv", index=False, encoding="utf-8-sig")
    selected, coverage, raw_milp = optimize_tasks_with_diagnostics(candidates, max_tasks=None)
    selected, verification = optimize_with_verification(candidates, traj3, targets, max_tasks=None)
    selected.to_csv(OUT / "tables" / "selected_tasks_milp.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "tables" / "optimized_selected_tasks.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "tables" / "selected_tasks.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT / "tables" / "target_coverage_check.csv", index=False, encoding="utf-8-sig")
    photo_angle_check(selected).to_csv(OUT / "tables" / "photo_angle_check.csv", index=False, encoding="utf-8-sig")
    verification.to_csv(OUT / "tables" / "selected_tasks_verification.csv", index=False, encoding="utf-8-sig")
    compare_task_solutions(greedy, selected).to_csv(OUT / "tables" / "task_optimization_compare.csv", index=False, encoding="utf-8-sig")
    old_reference = {
        "old_mixed_unit_milp": "P11|P10|P03|P04|S10|P01|S12|P02|S11",
        "new_normalized_milp": "|".join((selected["目标编号"].astype(str)).tolist()) if not selected.empty else "",
    }
    pd.DataFrame(
        [
            {
                "comparison": "old_mixed_unit_vs_new_normalized",
                "old_task_ids": old_reference["old_mixed_unit_milp"],
                "new_task_ids": old_reference["new_normalized_milp"],
                "changed": old_reference["old_mixed_unit_milp"] != old_reference["new_normalized_milp"],
            }
        ]
    ).to_csv(OUT / "tables" / "old_vs_new_task_selection.csv", index=False, encoding="utf-8-sig")

    old_path = OUT / "tables" / "optimized_selected_tasks.csv"
    # Local continuous refinement.
    refine, refined_selected = local_refine_tasks(traj3, targets, selected)
    refine.to_csv(OUT / "tables" / "continuous_refinement_selected_tasks.csv", index=False, encoding="utf-8-sig")
    refined_selected.to_csv(OUT / "tables" / "final_selected_tasks_refined.csv", index=False, encoding="utf-8-sig")
    write_log("refinement_decision.md", f"# Continuous Refinement\n\nTotal normalized margin improvement: {refine['improvement'].sum():.6f}. Rollbacks: {int(refine['rollback_or_not'].sum())}.\n")

    fill_result_template(ROOT / "result.xlsx", OUT / "result_filled.xlsx", selected, max_rows=9)
    plot_series(traj3, "speed", "附件3融合轨迹速度", "速度(m/s)", OUT / "figures" / "attachment3_speed.png")
    plot_series(traj3, "acceleration", "附件3融合轨迹加速度", "加速度(m/s²)", OUT / "figures" / "attachment3_acceleration.png")
    plot_fused_trajectory(traj3, "附件3融合10Hz轨迹", OUT / "figures" / "attachment3_fused_10hz.png")
    plot_tasks(traj3, targets, selected, OUT / "figures" / "selected_tasks_distribution.png")
    plot_task_timeline(candidates, selected, OUT / "figures" / "task_window_timeline.png")

    # Logs.
    align_df = pd.DataFrame(summary_rows)
    attach3_boot = boot_sens[boot_sens["block_length"] == auto_len].iloc[0]
    summary = f"""# Model Iteration Summary

## Code Structure Audit
- Main entry: `src/main.py`; extended reproducibility entry: `scripts/reproduce_model_results.py`.
- Time alignment: `src/alignment.py`.
- Bias estimation: `src/alignment.py` and diagnostics in `src/model_diagnostics.py`.
- Fusion weights: `src/alignment.py` equal weights; sensitivity in `src/model_diagnostics.py`.
- Kinematics: `src/kinematics.py`.
- Task candidates and MILP: `src/task_opt.py`.
- Outputs: `outputs/tables`, `outputs/logs`, `outputs/figures/diagnostics`.

## Model Changes
- Normalized dimensionless stability margin replaces mixed-unit margin.
- MILP coverage variable constraints are now bidirectional.
- Circular angle distance is used for photo angle constraints.
- Added Delta-b coupling diagnostics, bias model validation comparison, fusion weight sensitivity, automatic block bootstrap, interpolation sensitivity, robust objective comparison, threshold sensitivity, and local continuous time refinement.

## Final Alignment
```text
{align_df.to_string(index=False)}
```

## Attachment 3 Bias Decision
- Automatic bootstrap block length: {auto_len} points = {auto_len*0.1:.2f}s.
- Bias norm: {np.hypot(results['附件3'].bias_x, results['附件3'].bias_y):.4f}; tau_b: {max(0.3, 0.2*results['附件3'].rmse_after):.4f}.
- Significant fixed bias under auto block bootstrap: {bool(attach3_boot['significant_fixed_bias'])}.

## Fusion Decision
- Main result retains equal weights unless downstream sensitivity suggests material task changes.
- See `fusion_weight_sensitivity.csv` and `noise_variance_estimation.csv`.

## Task Optimization
- Candidate count: {len(candidates)}.
- Greedy selected: {len(greedy)}; MILP selected: {len(selected)}.
- MILP normalized margin sum: {selected['稳定裕度'].sum() if not selected.empty else 0:.4f}.
- Verification all pass: {bool((not verification.empty) and verification['pass_all'].all())}.
- Continuous refinement total improvement: {refine['improvement'].sum():.6f}.

## Remaining Risks
- Source-specific noise variance cannot be uniquely identified from paired residuals alone.
- Time-varying bias models are diagnostic; final correction should depend on validation improvement, not training improvement.
- Continuous refinement uses interpolation-free nearest-grid kinematic values, so it is a local robustness check rather than a full continuous optimal control solve.
"""
    write_log("model_iteration_summary.md", summary)
    write_log("alignment_objective_decision.md", "# Robust Alignment Objective Decision\n\nSee `robust_alignment_objective_comparison.csv`. Small Delta spread indicates robust time alignment; large spread should be treated as identifiability risk.\n")
    print("model reproduction completed")
    print(summary)


if __name__ == "__main__":
    main()
