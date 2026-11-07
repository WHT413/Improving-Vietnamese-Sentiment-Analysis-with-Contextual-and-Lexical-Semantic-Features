"""Orchestrate the offline composite lexicon build."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .config import (
    CURATED_PATH,
    DEFAULT_OUTPUT_DIR,
    DEV_PATH,
    TRAIN_PATH,
    VAD_PATH,
    VERSION,
    VNEMO_PATH,
    VSWN_PATH,
)
from .corpus import load_corpus_token_counts
from .coverage import compute_split_coverage
from .join import build_entries
from .outputs import write_outputs
from .payload import build_payload
from .vad import parse_vad
from .validation import validate_composite
from .vnemolex import parse_vnemolex
from .vswn import parse_vswn


@dataclass(frozen=True)
class _Artifacts:
    entries: dict[str, dict[str, list[Any]]]
    baseline_entries: dict[str, dict[str, list[Any]]]
    cov_train: dict[str, Any]
    cov_dev: dict[str, Any]
    base_cov_train: dict[str, Any]
    base_cov_dev: dict[str, Any]


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    output_dir = Path(_parse_args(argv).output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _print_summary(output_dir, _build_artifacts(output_dir))


def _parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def _build_artifacts(output_dir: Path) -> _Artifacts:
    segmenter = VnCoreNLPSegmenter()
    train_raw, train_word = load_corpus_token_counts(TRAIN_PATH)
    dev_raw, dev_word = load_corpus_token_counts(DEV_PATH)
    vswn, vad, vnemo = _load_sources(segmenter, train_word, dev_word)

    vswn_data, vswn_conflicts, vswn_dropped, vswn_stats, base_vswn = vswn
    vad_data, vad_conflicts, vad_dropped, vad_stats, base_vad = vad
    vnemo_data, vnemo_conflicts, vnemo_dropped, vnemo_stats = vnemo

    entries = build_entries(vswn_data, vad_data, vnemo_data)
    baseline_entries = build_entries(base_vswn, base_vad, vnemo_data)
    cov_train = compute_split_coverage(train_raw, train_word, entries)
    cov_dev = compute_split_coverage(dev_raw, dev_word, entries)
    base_cov_train = compute_split_coverage(train_raw, train_word, baseline_entries)
    base_cov_dev = compute_split_coverage(dev_raw, dev_word, baseline_entries)
    all_dropped = vswn_dropped + vad_dropped + vnemo_dropped

    payload = build_payload(
        vswn_stats=vswn_stats,
        vad_stats=vad_stats,
        vnemo_stats=vnemo_stats,
        entries=entries,
        baseline_entries=baseline_entries,
        cov_train=cov_train,
        cov_dev=cov_dev,
        base_cov_train=base_cov_train,
        base_cov_dev=base_cov_dev,
        all_dropped=all_dropped,
    )
    validate_composite(payload)
    write_outputs(
        output_dir,
        payload,
        vswn_conflicts=vswn_conflicts,
        vad_conflicts=vad_conflicts,
        vnemo_conflicts=vnemo_conflicts,
        all_dropped=all_dropped,
    )
    return _Artifacts(
        entries=entries,
        baseline_entries=baseline_entries,
        cov_train=cov_train,
        cov_dev=cov_dev,
        base_cov_train=base_cov_train,
        base_cov_dev=base_cov_dev,
    )


def _load_sources(
    segmenter: VnCoreNLPSegmenter,
    train_word_counts,
    dev_word_counts,
):
    vswn = parse_vswn(
        segmenter, train_word_counts, dev_word_counts, VSWN_PATH, CURATED_PATH
    )
    vad = parse_vad(segmenter, VAD_PATH)
    vnemo = parse_vnemolex(segmenter, VNEMO_PATH)
    return vswn, vad, vnemo


def _print_summary(output_dir: Path, artifacts: _Artifacts) -> None:
    print(f"Generated composite lexicon v{VERSION} at: {output_dir}")
    print(
        f"Total entries: {len(artifacts.entries)} "
        f"(baseline: {len(artifacts.baseline_entries)})"
    )
    print(_coverage_line("Train", artifacts.cov_train, artifacts.base_cov_train))
    print(
        _coverage_line(
            "Dev", artifacts.cov_dev, artifacts.base_cov_dev, colon_padding=2
        )
    )


def _coverage_line(
    label: str,
    coverage: dict[str, Any],
    baseline: dict[str, Any],
    *,
    colon_padding: int = 0,
) -> str:
    current = coverage["word_token_coverage"]
    previous = baseline["word_token_coverage"]
    padding = " " * colon_padding
    return (
        f"{label} word-token coverage:{padding} {current['token_coverage_pct']:.2f}% token / "
        f"{current['type_coverage_pct']:.2f}% type "
        f"(baseline: {previous['token_coverage_pct']:.2f}% / "
        f"{previous['type_coverage_pct']:.2f}%)"
    )