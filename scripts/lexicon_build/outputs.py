"""Write the composite JSON, conflict tables, dropped log, and README."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .readme import write_readme
from .tsv import write_tsv


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    vswn_conflicts: list[dict[str, Any]],
    vad_conflicts: list[dict[str, Any]],
    vnemo_conflicts: list[dict[str, Any]],
    all_dropped: list[dict[str, Any]],
) -> None:
    _write_composite(output_dir, payload)
    _write_vswn_conflicts(output_dir, vswn_conflicts)
    _write_vad_conflicts(output_dir, vad_conflicts)
    _write_vnemo_conflicts(output_dir, vnemo_conflicts)
    _write_dropped(output_dir, all_dropped)
    write_readme(output_dir / "README.source.md", payload)


def _write_composite(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "composite_lexicon.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_vswn_conflicts(
    output_dir: Path, rows: list[dict[str, Any]]
) -> None:
    write_tsv(
        output_dir / "conflicts_vswn.tsv",
        [
            "key",
            "train_freq",
            "dev_freq",
            "num_candidates",
            "unique_scores",
            "decision",
            "selected_pos_sense",
            "selected_score",
            "rationale",
            "candidates_detail",
        ],
        rows,
    )


def _write_vad_conflicts(
    output_dir: Path, rows: list[dict[str, Any]]
) -> None:
    write_tsv(
        output_dir / "conflicts_vad.tsv",
        [
            "key",
            "num_rows",
            "decision",
            "valence_mask",
            "valence_value",
            "valence_candidates",
            "arousal_mask",
            "arousal_value",
            "arousal_candidates",
            "dominance_mask",
            "dominance_value",
            "dominance_candidates",
            "en_words",
            "raw_vi_terms",
        ],
        rows,
    )


def _write_vnemo_conflicts(
    output_dir: Path, rows: list[dict[str, Any]]
) -> None:
    write_tsv(
        output_dir / "conflicts_vnemolex.tsv",
        [
            "key",
            "num_rows",
            "rows",
            "en_words",
            "raw_vi_terms",
            "decision",
            "resolved_vector",
            "resolved_mask",
            "candidate_vectors",
            "rationale",
        ],
        rows,
    )


def _write_dropped(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_tsv(
        output_dir / "dropped_multitoken.tsv",
        ["source", "line", "raw_term", "canonical_key", "detail"],
        rows,
    )