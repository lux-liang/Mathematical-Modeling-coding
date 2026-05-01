# Robust Task Optimization V3

Smoothing scenarios: [61, 71, 81]; weights: [0.3, 0.4, 0.5, 0.6, 0.7].

               model  selected_task_count  coverage_count  nominal_margin_sum  mean_scenario_margin_sum  worst_case_margin_sum  min_task_worst_margin  scenario_feasible_rate  high_risk_task_count  smoothing_stability_score  whether_contains_S04  whether_contains_P10                                              task_ids
                  R3                    9               9               1.673                  1.195443               0.372890                    0.0                0.629630                    10                   0.629630                  True                  True S04射击|P04拍照|P03拍照|P10拍照|P11拍照|P01拍照|P02拍照|S11射击|S12射击
                  R4                    9               9               0.634                  0.935662               0.376758                    0.0                0.625000                    10                   0.625000                 False                  True S03射击|P11拍照|P04拍照|P03拍照|P10拍照|P01拍照|P02拍照|S10射击|S12射击
                  R5                    9               9               1.730                  1.567661               0.507204                    0.0                0.970370                     4                   0.970370                 False                  True S06射击|S03射击|P11拍照|P03拍照|P10拍照|P01拍照|P02拍照|S16射击|S15射击
R6_buffered_eps_0.00                    9               9               1.087                  1.255542               0.539195                    0.0                0.940741                     5                   0.940741                  True                  True S04射击|S05射击|P11拍照|P10拍照|P04拍照|P01拍照|P02拍照|S15射击|S16射击
                  R7                    9               9               1.203                  1.567661               0.507204                    0.0                0.970370                     4                   0.970370                  True                  True S03射击|S04射击|P11拍照|P03拍照|P10拍照|P01拍照|P02拍照|S15射击|S14射击

R6 epsilon selected: 0.0.
