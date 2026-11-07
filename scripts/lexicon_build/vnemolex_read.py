"""Read VnEmoLex rows and collect canonical emotion candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import openpyxl
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .config import EMOTION_ORDER
from .keys import canonicalize_key


def read_rows(source_path: Path, segmenter: VnCoreNLPSegmenter):
    workbook, rows_iter, vi_idx, emotion_indices = _open_sheet(source_path)
    rows_by_key = defaultdict(list)
    dropped_multitoken: list[dict[str, Any]] = []
    data_rows = 0
    blank_cells = 0

    for row_no, row in enumerate(rows_iter, 2):
        parsed = _parse_row(row, row_no, vi_idx, emotion_indices, segmenter)
        if parsed is None:
            continue
        data_rows += 1
        key, item, dropped_entry, row_blanks = parsed
        blank_cells += row_blanks
        if not key:
            continue
        if dropped_entry is not None:
            dropped_multitoken.append(dropped_entry)
            continue
        rows_by_key[key].append(item)

    workbook.close()
    return rows_by_key, dropped_multitoken, data_rows, blank_cells


def _open_sheet(source_path: Path):
    workbook = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
    rows_iter = workbook.active.iter_rows(values_only=True)
    header = [str(cell).strip() if cell is not None else "" for cell in next(rows_iter)]
    vi_idx = next(i for i, name in enumerate(header) if "vietnamese" in name.lower())
    emotion_indices = {
        emotion: next(
            i for i, name in enumerate(header) if name.lower().startswith(emotion)
        )
        for emotion in EMOTION_ORDER
    }
    return workbook, rows_iter, vi_idx, emotion_indices


def _parse_row(row, row_no, vi_idx, emotion_indices, segmenter):
    if not row or row[vi_idx] is None:
        return None
    raw_vi = str(row[vi_idx]).strip()
    if not raw_vi:
        return None
    en_word = str(row[0]).strip() if row[0] is not None else ""
    values, masks, raw_cells, blanks = _emotion_values(row, emotion_indices)
    key = canonicalize_key(raw_vi, segmenter)
    if not key:
        return key, None, None, blanks
    if " " in key:
        return (
            key,
            None,
            {
                "source": "vnemolex",
                "line": row_no,
                "raw_term": raw_vi,
                "canonical_key": key,
                "detail": f"en={en_word} emotions={','.join(raw_cells)}",
            },
            blanks,
        )
    item = {
        "row": row_no,
        "en_word": en_word,
        "raw_vi": raw_vi,
        "vals": values,
        "masks": masks,
    }
    return key, item, None, blanks


def _emotion_values(row, emotion_indices):
    values: list[float] = []
    masks: list[int] = []
    raw_cells: list[str] = []
    blanks = 0
    for emotion in EMOTION_ORDER:
        cell = row[emotion_indices[emotion]]
        if cell is None or str(cell).strip() == "":
            values.append(0.0)
            masks.append(0)
            raw_cells.append("BLANK")
            blanks += 1
        else:
            value = int(float(cell))
            values.append(float(value))
            masks.append(1)
            raw_cells.append(str(value))
    return values, masks, raw_cells, blanks