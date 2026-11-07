"""VnEmoLex parsing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .vnemolex_read import read_rows
from .vnemolex_resolve import resolve_duplicates


def parse_vnemolex(
    segmenter: VnCoreNLPSegmenter,
    source_path: Path,
) -> tuple[
    dict[str, tuple[list[float], list[int]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows_by_key, dropped, data_rows, blank_cells = read_rows(source_path, segmenter)
    resolved, conflict_logs, stats = resolve_duplicates(
        rows_by_key,
        data_rows=data_rows,
        blank_cells=blank_cells,
        dropped_count=len(dropped),
    )
    return resolved, conflict_logs, dropped, stats