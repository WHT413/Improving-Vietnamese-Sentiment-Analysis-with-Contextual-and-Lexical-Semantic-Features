"""Vietnamese NRC-VAD parsing and per-dimension conflict resolution."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .keys import canonicalize_key


def parse_vad(
    segmenter: VnCoreNLPSegmenter,
    source_path: Path,
) -> tuple[
    dict[str, tuple[list[float], list[int]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, tuple[list[float], list[int]]],
]:
    rows_by_key, dropped_multitoken, data_rows = _read_rows(source_path, segmenter)
    resolved, baseline, conflict_logs, stats = _resolve_rows(
        rows_by_key, data_rows=data_rows, dropped_count=len(dropped_multitoken)
    )
    return resolved, conflict_logs, dropped_multitoken, stats, baseline


def _read_rows(source_path: Path, segmenter: VnCoreNLPSegmenter):
    rows_by_key = defaultdict(list)
    dropped_multitoken: list[dict[str, Any]] = []
    data_rows = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line or line.lower().startswith("english"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            en_word, v_s, a_s, d_s, vi_word = parts[:5]
            try:
                v, a, d = float(v_s), float(a_s), float(d_s)
            except ValueError:
                continue
            data_rows += 1
            key = canonicalize_key(vi_word, segmenter)
            if not key:
                continue
            if " " in key:
                dropped_multitoken.append(
                    {
                        "source": "nrc-vad-vi",
                        "line": line_no,
                        "raw_term": vi_word,
                        "canonical_key": key,
                        "detail": f"en={en_word} V={v} A={a} D={d}",
                    }
                )
                continue
            rows_by_key[key].append(
                {
                    "en_word": en_word,
                    "raw_vi": vi_word,
                    "v": v,
                    "a": a,
                    "d": d,
                }
            )
    return rows_by_key, dropped_multitoken, data_rows


def _resolve_rows(rows_by_key, *, data_rows: int, dropped_count: int):
    resolved: dict[str, tuple[list[float], list[int]]] = {}
    baseline: dict[str, tuple[list[float], list[int]]] = {}
    conflict_logs: list[dict[str, Any]] = []
    counters = {"unambiguous": 0, "conflict": 0, "partial": 0, "all_masked": 0}
    kept_by_dim = {"valence": 0, "arousal": 0, "dominance": 0}

    for key in sorted(rows_by_key):
        resolved_value, baseline_value, log = _resolve_key(
            key, rows_by_key[key], counters, kept_by_dim
        )
        resolved[key] = resolved_value
        baseline[key] = baseline_value
        if log is not None:
            conflict_logs.append(log)

    stats = {
        "rows": data_rows,
        "usable_single_token_keys": len(rows_by_key),
        "unambiguous_keys": counters["unambiguous"],
        "conflict_keys": counters["conflict"],
        "conflict_partial_kept_keys": counters["partial"],
        "conflict_all_masked_keys": counters["all_masked"],
        "active_keys_by_dim": kept_by_dim,
        "dropped_multitoken_rows": dropped_count,
    }
    return resolved, baseline, conflict_logs, stats


def _resolve_key(key, items, counters, kept_by_dim):
    v_vals = sorted({item["v"] for item in items})
    a_vals = sorted({item["a"] for item in items})
    d_vals = sorted({item["d"] for item in items})

    if len(v_vals) == 1 and len(a_vals) == 1 and len(d_vals) == 1:
        counters["unambiguous"] += 1
        vals = [v_vals[0], a_vals[0], d_vals[0]]
        masks = [1, 1, 1]
        for dimension in kept_by_dim:
            kept_by_dim[dimension] += 1
        return (vals, masks), (list(vals), list(masks)), None

    counters["conflict"] += 1
    vals = [
        v_vals[0] if len(v_vals) == 1 else 0.0,
        a_vals[0] if len(a_vals) == 1 else 0.0,
        d_vals[0] if len(d_vals) == 1 else 0.0,
    ]
    masks = [
        1 if len(v_vals) == 1 else 0,
        1 if len(a_vals) == 1 else 0,
        1 if len(d_vals) == 1 else 0,
    ]
    if any(masks):
        counters["partial"] += 1
        _increment_kept(masks, kept_by_dim)
        decision = "partial_keep_consistent_dims"
    else:
        counters["all_masked"] += 1
        decision = "masked_all_3_dims_conflict"

    log = _build_log(
        key, items, decision, masks, vals, v_vals, a_vals, d_vals
    )
    return (vals, masks), ([0.0, 0.0, 0.0], [0, 0, 0]), log


def _increment_kept(masks, kept_by_dim) -> None:
    if masks[0]:
        kept_by_dim["valence"] += 1
    if masks[1]:
        kept_by_dim["arousal"] += 1
    if masks[2]:
        kept_by_dim["dominance"] += 1


def _build_log(key, items, decision, masks, vals, v_vals, a_vals, d_vals):
    return {
        "key": key,
        "num_rows": len(items),
        "decision": decision,
        "valence_mask": masks[0],
        "valence_value": f"{vals[0]:.6g}",
        "valence_candidates": ",".join(f"{x:.6g}" for x in v_vals),
        "arousal_mask": masks[1],
        "arousal_value": f"{vals[1]:.6g}",
        "arousal_candidates": ",".join(f"{x:.6g}" for x in a_vals),
        "dominance_mask": masks[2],
        "dominance_value": f"{vals[2]:.6g}",
        "dominance_candidates": ",".join(f"{x:.6g}" for x in d_vals),
        "en_words": ",".join(item["en_word"] for item in items),
        "raw_vi_terms": ",".join(sorted({item["raw_vi"] for item in items})),
    }