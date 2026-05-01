from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data_io import TIME_COL, X_COL, Y_COL


def add_kinematics(df: pd.DataFrame, smooth_window: int = 11) -> pd.DataFrame:
    out = df.copy()
    t = out[TIME_COL].to_numpy(dtype=float)
    x = out[X_COL].to_numpy(dtype=float)
    y = out[Y_COL].to_numpy(dtype=float)
    dt = float(np.nanmedian(np.diff(t)))
    window = min(smooth_window, len(out) - 1 if len(out) % 2 == 0 else len(out))
    if window >= 5:
        if window % 2 == 0:
            window -= 1
        xs = savgol_filter(x, window_length=window, polyorder=3, mode="interp")
        ys = savgol_filter(y, window_length=window, polyorder=3, mode="interp")
    else:
        xs, ys = x, y
    vx = np.gradient(xs, dt)
    vy = np.gradient(ys, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    out["x_smooth"] = xs
    out["y_smooth"] = ys
    out["vx"] = vx
    out["vy"] = vy
    out["speed"] = np.sqrt(vx * vx + vy * vy)
    out["ax"] = ax
    out["ay"] = ay
    out["acceleration"] = np.sqrt(ax * ax + ay * ay)
    return out
