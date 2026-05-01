# Model Version Decision V2

                                              version  selected_task_count  coverage_count  nominal_margin_sum  worst_case_margin_sum  min_task_margin  scenario_feasible_rate  smoothing_stability_score  refined_margin_gain                                                         recommendation
                                   Version 1 baseline                    9               9            1.346058               0.329273              0.0                0.844444                   0.111111                  0.0                                                     baseline reference
                      Version 2 data-driven smoothing                    9               9            1.346058               0.329273              0.0                0.844444                   0.111111                  0.0                         use if smoothing selection keeps same task set
                       Version 3 robust scenario MILP                    9               9            1.255405               0.408148              0.0                0.933333                   0.111111                  0.0                                          recommended robust main model
Version 4 state-space estimated weight + robust check                    9               9            1.004643               0.633827              0.0                0.955556                   0.111111                  0.0 diagnostic weight w1=0.291; do not replace scenario robustness blindly

Recommended model version: Version 3 robust scenario MILP, provided it keeps 9 tasks and 9 covered targets. State-space weight remains diagnostic rather than the sole main result.
