# Model Iteration Summary

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
 数据                 Delta定义    Delta(s)  bias_x(m)  bias_y(m)  bias_norm(m)  rmse_before(m)  rmse_after(m)  overlap_start(s)  overlap_end(s)
附件1 t2_aligned = t2 + Delta -198.431701   0.000000   0.000000      0.000000    9.055191e-08   9.055191e-08        270.399999      970.750000
附件2 t2_aligned = t2 + Delta  -50.494699   3.479972  -1.822119      3.928145    3.997029e+00   7.312951e-01        161.826301      851.750000
附件3 t2_aligned = t2 + Delta  367.884334  -0.178205  -0.279553      0.331522    2.749707e+00   2.745182e+00        469.050000      769.014334
```

## Attachment 3 Bias Decision
- Automatic bootstrap block length: 10 points = 1.00s.
- Bias norm: 0.3315; tau_b: 0.5490.
- Significant fixed bias under auto block bootstrap: False.

## Fusion Decision
- Main result retains equal weights unless downstream sensitivity suggests material task changes.
- See `fusion_weight_sensitivity.csv` and `noise_variance_estimation.csv`.

## Task Optimization
- Candidate count: 582.
- Greedy selected: 9; MILP selected: 9.
- MILP normalized margin sum: 1.8100.
- Verification all pass: True.
- Continuous refinement total improvement: 0.000704.

## Remaining Risks
- Source-specific noise variance cannot be uniquely identified from paired residuals alone.
- Time-varying bias models are diagnostic; final correction should depend on validation improvement, not training improvement.
- Continuous refinement uses interpolation-free nearest-grid kinematic values, so it is a local robustness check rather than a full continuous optimal control solve.
