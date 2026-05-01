from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


def fill_result_template(template_path: Path, output_path: Path, tasks: pd.DataFrame, max_rows: int = 9) -> None:
    wb = load_workbook(template_path)
    ws = wb[wb.sheetnames[0]]
    fill_rows = []
    for r in range(2, ws.max_row + 1):
        if isinstance(ws.cell(row=r, column=1).value, int):
            fill_rows.append(r)
    fill_rows = fill_rows[:max_rows]
    for row in fill_rows:
        for col in range(2, 6):
            ws.cell(row=row, column=col).value = None
    for i, (_, task) in enumerate(tasks.head(len(fill_rows)).iterrows()):
        row = fill_rows[i]
        ws.cell(row=row, column=1).value = int(i + 1)
        ws.cell(row=row, column=2).value = str(task["目标编号"])
        ws.cell(row=row, column=3).value = str(task["任务"])
        ws.cell(row=row, column=4).value = float(task["开始准备时刻(s)"])
        ws.cell(row=row, column=5).value = float(task["任务执行时刻(s)"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
