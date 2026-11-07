"""Resolve VSWN candidates using curated senses or masking."""

from __future__ import annotations

from typing import Any


def resolve_conflicts(
    candidates_by_key,
    curated: dict[str, tuple[str, float, str]],
    train_counts,
    dev_counts,
    *,
    data_rows: int,
    raw_terms_count: int,
    dropped_count: int,
):
    resolved: dict[str, tuple[float, int]] = {}
    baseline: dict[str, tuple[float, int]] = {}
    conflict_logs: list[dict[str, Any]] = []
    counters = {
        "unambiguous": 0,
        "conflict": 0,
        "selected": 0,
        "selected_in_train": 0,
        "masked": 0,
        "in_train": 0,
    }

    for key in sorted(candidates_by_key):
        resolved_value, baseline_value, log = _resolve_key(
            key,
            candidates_by_key[key],
            curated,
            train_counts,
            dev_counts,
            counters,
        )
        resolved[key] = resolved_value
        baseline[key] = baseline_value
        if log is not None:
            conflict_logs.append(log)

    stats = {
        "rows": data_rows,
        "raw_terms": raw_terms_count,
        "usable_single_token_keys": len(candidates_by_key),
        "unambiguous_keys": counters["unambiguous"],
        "conflict_keys": counters["conflict"],
        "conflict_keys_in_train": counters["in_train"],
        "conflict_selected_keys": counters["selected"],
        "conflict_selected_in_train": counters["selected_in_train"],
        "conflict_masked_keys": counters["masked"],
        "active_polarity_keys": counters["unambiguous"] + counters["selected"],
        "dropped_multitoken_rows": dropped_count,
    }
    return resolved, baseline, conflict_logs, stats


def _resolve_key(key, cands, curated, train_counts, dev_counts, counters):
    unique_scores = sorted({candidate["score"] for candidate in cands})
    if len(unique_scores) == 1:
        counters["unambiguous"] += 1
        value = (unique_scores[0], 1)
        return value, value, None

    counters["conflict"] += 1
    train_freq = train_counts.get(key, 0)
    dev_freq = dev_counts.get(key, 0)
    if train_freq > 0:
        counters["in_train"] += 1

    selected = _select_curated(key, cands, curated)
    if selected is not None:
        chosen, selected_pos_sense, rationale = selected
        counters["selected"] += 1
        if train_freq > 0:
            counters["selected_in_train"] += 1
        decision = "selected_verified_sense"
        selected_score = f"{chosen['score']:.6g}"
        resolved_value = (chosen["score"], 1)
    else:
        counters["masked"] += 1
        decision = "masked_unresolved"
        selected_pos_sense = ""
        selected_score = "0.0"
        rationale = (
            "Function word, degree modifier, or multi-word phrase fragment without a "
            "single unambiguous sentiment-bearing standalone sense; masked (mask=0, value=0.0)."
        )
        resolved_value = (0.0, 0)

    log = _build_log(
        key,
        cands,
        train_freq,
        dev_freq,
        unique_scores,
        decision,
        selected_pos_sense,
        selected_score,
        rationale,
    )
    return resolved_value, (0.0, 0), log


def _select_curated(key: str, cands, curated):
    if key not in curated:
        return None
    selected_id, expected_score, rationale = curated[key]
    matching = [candidate for candidate in cands if candidate["pos_sense_id"] == selected_id]
    if not matching:
        raise ValueError(
            f"Curated VSWN selection {selected_id} not found for key '{key}'"
        )
    chosen = matching[0]
    if abs(chosen["score"] - expected_score) > 1e-6:
        raise ValueError(
            f"Score mismatch for '{key}' ({selected_id}): "
            f"got {chosen['score']}, expected {expected_score}"
        )
    return chosen, selected_id, rationale


def _build_log(
    key,
    cands,
    train_freq,
    dev_freq,
    unique_scores,
    decision,
    selected_pos_sense,
    selected_score,
    rationale,
):
    cand_summary = "; ".join(
        f"{candidate['pos_sense_id']}:{candidate['term_tag']}={candidate['score']:+.3f}"
        f"({'standalone' if candidate['is_standalone'] else 'fragment'} "
        f"in '{candidate['full_terms']}')"
        for candidate in cands
    )
    return {
        "key": key,
        "train_freq": train_freq,
        "dev_freq": dev_freq,
        "num_candidates": len(cands),
        "unique_scores": ",".join(f"{score:.6g}" for score in unique_scores),
        "decision": decision,
        "selected_pos_sense": selected_pos_sense,
        "selected_score": selected_score,
        "rationale": rationale,
        "candidates_detail": cand_summary,
    }