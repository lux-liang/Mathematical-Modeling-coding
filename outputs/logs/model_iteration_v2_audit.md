# Model Iteration V2 Audit

## Fusion Weight Sensitivity

Weights with task-set changes vs w1=0.5:

 w1  selected_task_count  covered_target_count  normalized_margin_sum                                     selected_task_ids
0.2                    9                     9                  1.537 P04拍照|P05拍照|P11拍照|P03拍照|P10拍照|P01拍照|P02拍照|S11射击|S10射击
0.3                    9                     9                  0.872 P07拍照|P06拍照|P04拍照|P11拍照|P10拍照|P03拍照|P01拍照|S11射击|P02拍照
0.4                    9                     9                  1.743 P03拍照|P04拍照|P11拍照|P10拍照|P02拍照|P01拍照|S11射击|S12射击|S10射击
0.6                    9                     9                  1.752 P12拍照|P04拍照|P10拍照|P11拍照|P03拍照|P01拍照|P02拍照|S12射击|S11射击
0.7                    9                     9                  1.892 P10拍照|P03拍照|P11拍照|P04拍照|P02拍照|P01拍照|S11射击|S10射击|S12射击
0.8                    9                     9                  1.586 P10拍照|P03拍照|P11拍照|P04拍照|P02拍照|P01拍照|S11射击|S10射击|S12射击


Weight metrics:

 w1  selected_task_count  covered_target_count  normalized_margin_sum  min_margin_proxy
0.2                    9                     9                  1.537          0.170778
0.3                    9                     9                  0.872          0.096889
0.4                    9                     9                  1.743          0.193667
0.5                    9                     9                  1.810          0.201111
0.6                    9                     9                  1.752          0.194667
0.7                    9                     9                  1.892          0.210222
0.8                    9                     9                  1.586          0.176222


## Interpolation / Smoothing Sensitivity

                    method  total_candidate_count  candidate_count_ratio_to_median  MILP_selected_count  normalized_margin_sum                                     selected_task_IDs
               CubicSpline                      9                         0.028391                    1                  0.043                                                 S11射击
                     PCHIP                     12                         0.037855                    1                  0.073                                                 S12射击
     linear+Savitzky-Golay                    622                         1.962145                    9                  1.565 P10拍照|P03拍照|P04拍照|P11拍照|P01拍照|P02拍照|S10射击|S11射击|S12射击
CubicSpline+Savitzky-Golay                    626                         1.974763                    9                  1.294 P11拍照|P10拍照|P03拍照|P04拍照|P01拍照|P02拍照|S10射击|S12射击|S11射击


## Boundary Tasks

 序号 目标编号 任务  开始准备时刻(s)  任务执行时刻(s)  距离(m)  方向角(deg)  速度(m/s)  加速度(m/s2)  稳定裕度
  4  P04 拍照     495.15     495.65 32.802      9.44    1.408      0.245 0.061
  9  S12 射击     762.35     763.85 15.597    -21.83    1.642      0.742 0.136
  1  P10 拍照     493.35     493.85 29.348    -68.37    1.044      0.725 0.160
  7  S10 射击     509.85     511.35 10.378     13.30    1.217      0.094 0.184
  6  P01 拍照     509.15     509.65 34.116    -79.00    0.944      0.369 0.196


## Refinement Summary

target_id task_type  original_time  refined_time  original_margin  refined_margin  improvement  pass_constraints  rollback_or_not
      P04        拍照         495.65        495.65            0.061        0.061000     0.000000             False             True
      S12        射击         763.85        763.75            0.136        0.136329     0.000329              True            False
      P10        拍照         493.85        493.85            0.160        0.160000     0.000000              True            False
      S10        射击         511.35        511.25            0.184        0.184220     0.000220              True            False
      P01        拍照         509.65        509.65            0.196        0.196000     0.000000             False             True


## Risk Ranking

1. Fusion weight changes alter the selected task set.

2. Raw Cubic/PCHIP interpolation without smoothing produces very few candidates.

3. The lowest-margin selected tasks remain close to acceleration or speed constraints.

4. Single-source variance remains only approximately identifiable.
