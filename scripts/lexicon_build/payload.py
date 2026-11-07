"""Build the composite JSON payload."""

from __future__ import annotations

from typing import Any

from .config import FEATURES, VERSION


def build_payload(
    *,
    vswn_stats: dict[str, Any],
    vad_stats: dict[str, Any],
    vnemo_stats: dict[str, Any],
    entries: dict[str, dict[str, list[Any]]],
    baseline_entries: dict[str, dict[str, list[Any]]],
    cov_train: dict[str, Any],
    cov_dev: dict[str, Any],
    base_cov_train: dict[str, Any],
    base_cov_dev: dict[str, Any],
    all_dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source": _source_metadata(),
        "features": list(FEATURES),
        "stats": _stats_metadata(
            vswn_stats,
            vad_stats,
            vnemo_stats,
            entries,
            baseline_entries,
            cov_train,
            cov_dev,
            base_cov_train,
            base_cov_dev,
            all_dropped,
        ),
        "entries": entries,
    }


def _source_metadata() -> dict[str, Any]:
    return {
        "name": "vnsentiment_composite_12d",
        "version": VERSION,
        "scale_contract": {
            "polarity": "[-1.0, 1.0] raw PosScore - NegScore (B1 loader rescales via (p + 1) / 2)",
            "vad": "[0.0, 1.0] native Vietnamese NRC-VAD scale (no (x + 1) / 2)",
            "emotion": "{0.0, 1.0} binary VnEmoLex (Positive/Negative dropped)",
            "mask_semantics": "mask=1 & value=0.0 means true zero annotation; mask=0 & value=0.0 means missing or unresolved conflict",
        },
        "inputs": {
            "vietsentiwordnet": "data/lexicons/vietsentiwordnet/VietSentiWordnet_Ver1.3.5.txt",
            "nrc_vad_vi": "data/lexicons/nrc-vad/Vietnamese-NRC-VAD-Lexicon.txt",
            "vnemolex": "data/lexicons/vnemolex/VnEmoLex.xlsx",
        },
        "canonicalization": "NFC -> lowercase -> whitespace collapse -> VnCoreNLPSegmenter (RDRSegmenter) -> strip leading/trailing underscore -> single-token filter",
        "conflict_policy": {
            "vietsentiwordnet": "curated standalone sense selection for verified sentiment-bearing keys; unresolved/function fragments masked (mask=0)",
            "nrc_vad_vi": "per-dimension comparison; keep consistent dimension(s) (mask=1), mask only conflicting dimension(s) (mask=0)",
            "vnemolex": "blank cells -> mask=0; audited 19 duplicate keys resolved via verified primary sense / OR-merge union; English-source rows are not necessarily supersets",
        },
    }


def _stats_metadata(
    vswn_stats: dict[str, Any],
    vad_stats: dict[str, Any],
    vnemo_stats: dict[str, Any],
    entries: dict[str, dict[str, list[Any]]],
    baseline_entries: dict[str, dict[str, list[Any]]],
    cov_train: dict[str, Any],
    cov_dev: dict[str, Any],
    base_cov_train: dict[str, Any],
    base_cov_dev: dict[str, Any],
    all_dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total_entries": len(entries),
        "baseline_total_entries": len(baseline_entries),
        "active_entries_per_dimension": _active_per_dimension(entries),
        "baseline_active_entries_per_dimension": _active_per_dimension(baseline_entries),
        "dropped_multitoken_total_rows": len(all_dropped),
        "sources": {
            "vietsentiwordnet": vswn_stats,
            "nrc_vad_vi": vad_stats,
            "vnemolex": vnemo_stats,
        },
        "coverage": {"train": cov_train, "dev": cov_dev},
        "baseline_exclude_all_conflicts_coverage": {
            "train": base_cov_train,
            "dev": base_cov_dev,
        },
    }


def _active_per_dimension(
    entries: dict[str, dict[str, list[Any]]],
) -> dict[str, int]:
    return {
        feature: sum(1 for item in entries.values() if item["mask"][index] == 1)
        for index, feature in enumerate(FEATURES)
    }