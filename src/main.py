from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS / ".mplconfig"))

import numpy as np
import pandas as pd

from data_io import generate_data_report, read_position_workbook, read_targets
from fill_result import fill_result_template_v2
from fusion import fuse_attachment
from kalman_bias import run_kalman_bias_attachment2
from kinematics import add_kinematics
from plotting import (
    plot_aligned,
    plot_fused_trajectory,
    plot_raw,
    plot_residuals,
    plot_series,
    plot_task_feasibility_heatmap,
    plot_task_timeline,
    plot_tasks,
)
from task_events import run_event_task_optimization
from task_opt import generate_candidates
from validation import bootstrap_bias_test, compare_bias_models, residual_dataframe, validate_alignment
from bias_structure import run_attachment3_bias_structure


def save_alignment_summary(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(OUTPUTS / "tables" / "alignment_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for sub in ["trajectories", "tables", "figures"]:
        (OUTPUTS / sub).mkdir(parents=True, exist_ok=True)

    excel_files = ["附件1.xlsx", "附件2.xlsx", "附件3.xlsx", "附件4.xlsx", "result.xlsx"]
    generate_data_report(BASE_DIR, OUTPUTS / "data_report.md", excel_files)
    print(f"数据报告已输出: {OUTPUTS / 'data_report.md'}")

    attachments = {
        "附件1": read_position_workbook(BASE_DIR / "附件1.xlsx"),
        "附件2": read_position_workbook(BASE_DIR / "附件2.xlsx"),
        "附件3": read_position_workbook(BASE_DIR / "附件3.xlsx"),
    }
    targets = read_targets(BASE_DIR / "附件4.xlsx")

    summary = []
    validation_rows = []
    bias_model_rows = []
    fused_outputs: dict[str, pd.DataFrame] = {}
    alignment_results = {}
    config = {
        "附件1": {"estimate_bias": False, "smooth_window": 1},
        "附件2": {"estimate_bias": True, "smooth_window": 5},
        "附件3": {"estimate_bias": True, "smooth_window": 7},
    }
    for name, data in attachments.items():
        print(f"处理{name}...")
        plot_raw(data, f"{name}原始轨迹", OUTPUTS / "figures" / f"raw_{name}.png")
        result, fused = fuse_attachment(data, **config[name])
        alignment_results[name] = result
        validation_rows.append(
            validate_alignment(
                name,
                data[next(s for s in data if "方式1" in s)],
                data[next(s for s in data if "方式2" in s)],
                result,
                config[name]["estimate_bias"],
                config[name]["smooth_window"],
            )
        )
        if name in {"附件2", "附件3"}:
            bias_model_rows.extend(
                compare_bias_models(
                    name,
                    data[next(s for s in data if "方式1" in s)],
                    data[next(s for s in data if "方式2" in s)],
                    result,
                    config[name]["smooth_window"],
                )
            )
        if name == "附件2":
            residuals2 = residual_dataframe(
                data[next(s for s in data if "方式1" in s)],
                data[next(s for s in data if "方式2" in s)],
                result,
                config[name]["smooth_window"],
            )
            residuals2.to_csv(OUTPUTS / "tables" / "attachment2_residuals.csv", index=False, encoding="utf-8-sig")
            plot_residuals(residuals2, "附件2校正后残差散点", OUTPUTS / "figures" / "attachment2_residual_scatter.png")
        if name == "附件3":
            fused = add_kinematics(fused, smooth_window=81)
            bias_test = bootstrap_bias_test(
                name,
                data[next(s for s in data if "方式1" in s)],
                data[next(s for s in data if "方式2" in s)],
                result,
                config[name]["smooth_window"],
            )
            has_bias = bias_test["has_system_bias"]
        else:
            has_bias = "" if name != "附件2" else True
        out_path = OUTPUTS / "trajectories" / f"fused_attachment{name[-1]}_10hz.csv"
        fused.to_csv(out_path, index=False, encoding="utf-8-sig")
        plot_aligned(fused, f"{name}对齐后轨迹", OUTPUTS / "figures" / f"aligned_{name}.png")
        fused_outputs[name] = fused
        summary.append(
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
                "附件3是否判断存在系统偏差": has_bias,
            }
        )
        print(
            f"{name}: Delta={result.delta:.4f}s, bias=({result.bias_x:.3f},{result.bias_y:.3f})m, "
            f"RMSE {result.rmse_before:.3f}->{result.rmse_after:.3f}m"
        )

    save_alignment_summary(summary)
    pd.DataFrame(validation_rows).to_csv(OUTPUTS / "tables" / "alignment_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bias_model_rows).to_csv(OUTPUTS / "tables" / "bias_model_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([bias_test]).to_csv(OUTPUTS / "tables" / "system_bias_test.csv", index=False, encoding="utf-8-sig")
    run_kalman_bias_attachment2(
        attachments["附件2"][next(s for s in attachments["附件2"] if "方式1" in s)],
        attachments["附件2"][next(s for s in attachments["附件2"] if "方式2" in s)],
        OUTPUTS,
    )
    run_attachment3_bias_structure(
        attachments["附件3"][next(s for s in attachments["附件3"] if "方式1" in s)],
        attachments["附件3"][next(s for s in attachments["附件3"] if "方式2" in s)],
        OUTPUTS,
    )

    traj3 = fused_outputs["附件3"]
    if "speed" not in traj3.columns:
        traj3 = add_kinematics(traj3, smooth_window=81)
    plot_series(traj3, "speed", "附件3融合轨迹速度", "速度(m/s)", OUTPUTS / "figures" / "attachment3_speed.png")
    plot_series(traj3, "acceleration", "附件3融合轨迹加速度", "加速度(m/s²)", OUTPUTS / "figures" / "attachment3_acceleration.png")
    plot_fused_trajectory(traj3, "附件3融合10Hz轨迹", OUTPUTS / "figures" / "attachment3_fused_10hz.png")

    candidates = generate_candidates(traj3, targets)
    candidates.to_csv(OUTPUTS / "tables" / "legacy_task_candidates_single_target.csv", index=False, encoding="utf-8-sig")
    task_outputs = run_event_task_optimization(traj3, targets, OUTPUTS, fov_main=45.0)
    selected = task_outputs["joint_selected"]
    fill_result_template_v2(BASE_DIR / "result.xlsx", OUTPUTS / "result_filled_v2.xlsx", selected)
    plot_task_timeline(candidates, pd.DataFrame(), OUTPUTS / "figures" / "legacy_candidate_timeline.png")
    plot_task_feasibility_heatmap(candidates, OUTPUTS / "figures" / "task_feasibility_heatmap.png")

    print(f"候选任务数: {len(candidates)}")
    print(f"联合事件任务数: {len(selected)}")
    print(f"任务方案对比已输出: {OUTPUTS / 'tables' / 'task_plan_comparison.csv'}")
    print(f"结果模板已输出: {OUTPUTS / 'result_filled_v2.xlsx'}")


if __name__ == "__main__":
    main()
