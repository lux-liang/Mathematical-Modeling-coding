from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import pandas as pd

from bias_structure import run_attachment3_bias_structure
from data_io import read_position_workbook, read_targets
from fill_result import fill_result_template_v2
from fusion import fuse_attachment
from kalman_bias import run_kalman_bias_attachment2
from kinematics import add_kinematics
from task_events import run_event_task_optimization


def ensure_dirs() -> None:
    for sub in ["tables", "figures", "trajectories", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Full v4 reproduction for teacher-feedback model iteration."""

    ensure_dirs()
    att2 = read_position_workbook(ROOT / "附件2.xlsx")
    att3 = read_position_workbook(ROOT / "附件3.xlsx")
    sheet1_2 = next(s for s in att2 if "方式1" in s)
    sheet2_2 = next(s for s in att2 if "方式2" in s)
    sheet1_3 = next(s for s in att3 if "方式1" in s)
    sheet2_3 = next(s for s in att3 if "方式2" in s)

    print("Running problem 2 Kalman-RTS bias model...")
    kalman_summary, kalman_sens, _kalman_traj = run_kalman_bias_attachment2(att2[sheet1_2], att2[sheet2_2], OUT)
    print(kalman_summary[["method", "Delta", "bx", "by", "rmse_after"]].to_string(index=False))

    print("Running problem 3 bias-structure identification...")
    bias_cmp, _coeffs, _residual_frame = run_attachment3_bias_structure(att3[sheet1_3], att3[sheet2_3], OUT)
    print(bias_cmp[["model_name", "cv_rmse", "improvement_vs_M1", "conclusion"]].to_string(index=False))

    print("Building attachment 3 trajectory for event-level task optimization...")
    _result3, fused3 = fuse_attachment(att3, estimate_bias=True, smooth_window=7)
    traj3 = add_kinematics(fused3, smooth_window=81)
    traj3.to_csv(OUT / "trajectories" / "fused_attachment3_v4_10hz.csv", index=False, encoding="utf-8-sig")
    targets = read_targets(ROOT / "附件4.xlsx")

    print("Running problem 4 event-level shooting/photo/joint optimization...")
    task_outputs = run_event_task_optimization(traj3, targets, OUT, fov_main=45.0)
    joint = task_outputs["joint_selected"]
    fill_result_template_v2(ROOT / "result.xlsx", OUT / "result_filled_v2.xlsx", joint)
    print(task_outputs["comparison"].to_string(index=False))

    notes = [
        "# v4 model iteration outputs",
        "",
        "- Problem 2 adds Kalman Filter / RTS smoother for random-noise separation and fixed source-2 bias estimation.",
        "- Problem 3 compares M0-M4 residual bias structures with blocked cross-validation.",
        "- Problem 4 replaces the old 9-row task cap with event-level pure shooting, pure photo, and joint plans.",
        "- Joint task solver_status is a deterministic greedy interval fallback; no 9-task capacity constraint is used.",
    ]
    (OUT / "logs" / "v4_iteration_summary.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
