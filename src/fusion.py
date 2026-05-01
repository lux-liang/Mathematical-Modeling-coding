from __future__ import annotations

import pandas as pd

from alignment import align_pair, resample_aligned


def fuse_attachment(data: dict[str, pd.DataFrame], estimate_bias: bool, smooth_window: int):
    sheet1 = next(s for s in data if "方式1" in s)
    sheet2 = next(s for s in data if "方式2" in s)
    result = align_pair(data[sheet1], data[sheet2], estimate_bias=estimate_bias, smooth_window=smooth_window)
    fused = resample_aligned(data[sheet1], data[sheet2], result, smooth_window=smooth_window)
    return result, fused
