# Model Iteration V2 Summary

## Robust task models
model  selected_task_count  covered_target_count  margin_sum_mean  margin_sum_worst  min_task_margin_worst  feasible_rate_mean  task_set_stability                                              task_ids
   R1                    9                     9         1.346058          0.329273               0.000000            0.844444            0.844444 P10拍照|P03拍照|P11拍照|P04拍照|P02拍照|P01拍照|S10射击|S11射击|S12射击
   R2                    7                     7         0.962821          0.633827               0.034449            1.000000            1.000000             S03射击|P10拍照|P11拍照|P03拍照|P01拍照|P02拍照|S12射击
   R3                    9                     9         1.255405          0.408148               0.000000            0.933333            0.933333 S04射击|P04拍照|P03拍照|P10拍照|P11拍照|P01拍照|P02拍照|S11射击|S12射击
   R4                    9                     9         1.004643          0.633827               0.000000            0.955556            0.955556 S03射击|P11拍照|P04拍照|P03拍照|P10拍照|P01拍照|P02拍照|S10射击|S12射击

## Smoothing selection
      method  window_length  polyorder  fidelity_error  speed_p95  acceleration_p95  acceleration_max  jerk_p95  feasible_candidate_count  selected_task_count  margin_sum                                     selected_task_ids       score
cubic_savgol             81          3        0.914456   6.468326          1.034194          1.814094  0.528712                       607                    9       2.278 S05射击|P11拍照|P03拍照|P04拍照|P01拍照|P02拍照|S11射击|S10射击|S12射击 9082.693713

## Weight estimation
Final state-space-like weight w1=0.2915, w2=0.7085.

## Recommended version
Version 3 robust scenario MILP.

## Stability audit
 task_id type target_id  execution_time  nominal_margin  worst_case_margin  feasible_rate_under_weight_scenarios  feasible_rate_under_smoothing_scenarios  feasible_rate_under_time_shift  feasible_under_tightened_constraints  robust_score risk_level bottleneck_constraint
       1   射击       S04          482.95           0.187           0.000000                                   0.0                                     0.00                             1.0                                  True      0.200000       high                      
       2   拍照       P04          493.75           0.143           0.000000                                   0.8                                     0.45                             1.0                                  True      0.560000     medium                      
       3   拍照       P03          494.35           0.229           0.032564                                   1.0                                     0.35                             1.0                                  True      0.699771     medium                      
       4   拍照       P10          494.95           0.249           0.000000                                   0.0                                     0.25                             1.0                                  True      0.200000       high                      
       5   拍照       P11          495.55           0.149           0.034449                                   1.0                                     0.30                             1.0                                  True      0.730921     medium                      
       6   拍照       P01          508.75           0.141           0.000000                                   0.8                                     0.40                             1.0                                  True      0.560000     medium                      
       7   拍照       P02          509.65           0.242           0.038926                                   1.0                                     0.50                             1.0                                  True      0.706298     medium                      
       8   射击       S11          523.75           0.133           0.000000                                   0.8                                     0.40                             1.0                                  True      0.560000     medium                      
       9   射击       S12          763.75           0.200           0.096676                                   1.0                                     0.20                             1.0                                  True      0.819183        low                      
