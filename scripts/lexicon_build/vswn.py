"""VietSentiWordNet parsing orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .vswn_read import read_candidates
from .vswn_resolve import resolve_conflicts


def parse_vswn(
    segmenter: VnCoreNLPSegmenter,
    train_word_counts,
    dev_word_counts,
    source_path: Path,
    curated_path: Path,
) -> tuple[
    dict[str, tuple[float, int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, tuple[float, int]],
]:
    candidates, dropped, data_rows, raw_terms = read_candidates(source_path, segmenter)
    resolved, baseline, conflict_logs, stats = resolve_conflicts(
        candidates,
        _load_curated(curated_path),
        train_word_counts,
        dev_word_counts,
        data_rows=data_rows,
        raw_terms_count=raw_terms,
        dropped_count=len(dropped),
    )
    return resolved, conflict_logs, dropped, stats, baseline


def _load_curated(path: Path) -> dict[str, tuple[str, float, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: (item["pos_sense_id"], float(item["score"]), item["rationale"])
        for key, item in raw.items()
    }