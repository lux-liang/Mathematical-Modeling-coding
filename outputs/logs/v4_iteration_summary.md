# v4 model iteration outputs

- Problem 2 adds Kalman Filter / RTS smoother for random-noise separation and fixed source-2 bias estimation.
- Problem 3 compares M0-M4 residual bias structures with blocked cross-validation.
- Problem 4 replaces the old 9-row task cap with event-level pure shooting, pure photo, and joint plans.
- Joint task solver_status is a deterministic greedy interval fallback; no 9-task capacity constraint is used.