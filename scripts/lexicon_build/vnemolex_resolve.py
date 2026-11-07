"""Resolve VnEmoLex duplicate keys via union merge."""

from __future__ import annotations

from typing import Any


def resolve_duplicates(
    rows_by_key,
    *,
    data_rows: int,
    blank_cells: int,
    dropped_count: int,
):
    resolved: dict[str, tuple[list[float], list[int]]] = {}
    conflict_logs: list[dict[str, Any]] = []
    counters = {"duplicate": 0, "identical": 0, "superset": 0}

    for key in sorted(rows_by_key):
        items = rows_by_key[key]
        if len(items) == 1:
            resolved[key] = (items[0]["vals"], items[0]["masks"])
            continue

        counters["duplicate"] += 1
        merged_vals, merged_masks = _merge_items(items)
        category, decision, rationale = _classify_duplicate(key, items)
        counters["identical" if category == "identical" else "superset"] += 1
        resolved[key] = (merged_vals, merged_masks)
        conflict_logs.append(
            _build_log(key, items, decision, merged_vals, merged_masks, rationale)
        )

    stats = {
        "rows": data_rows,
        "usable_single_token_keys": len(rows_by_key),
        "duplicate_single_token_keys": counters["duplicate"],
        "duplicate_identical_vector_keys": counters["identical"],
        "duplicate_superset_or_merged_keys": counters["superset"],
        "blank_cells_masked": blank_cells,
        "dropped_multitoken_rows": dropped_count,
    }
    return resolved, conflict_logs, stats


def _merge_items(items):
    merged_vals = [0.0] * 8
    merged_masks = [0] * 8
    for item in items:
        for index in range(8):
            if item["masks"][index] == 1:
                merged_masks[index] = 1
                if item["vals"][index] == 1.0:
                    merged_vals[index] = 1.0
    return merged_vals, merged_masks


def _classify_duplicate(key: str, items):
    unique_vectors = {tuple(item["vals"]) for item in items}
    if len(unique_vectors) == 1:
        return (
            "identical",
            "identical_vectors_keep",
            "Orthographic tone-placement variants (old vs new Vietnamese diacritic style) "
            "have identical 8-emotion vectors across rows.",
        )
    if key == "nhọt":
        return (
            "superset",
            "select_primary_row_superset_or_merge",
            "Exact raw duplicate for English 'ulcer' (rows 235 and 312); row 235 annotations "
            "(sadness, anger, fear, disgust) are a strict superset of row 312 (sadness, anger, disgust); "
            "selecting primary row 235 / OR-merge yields the same vector.",
        )
    annotated_rows = [item for item in items if any(item["vals"])]
    annotated_detail = "; ".join(
        f"row{item['row']}({item['raw_vi']},en={item['en_word'] or 'NONE'})"
        for item in annotated_rows
    ) or "none"
    return (
        "superset",
        "or_merge_tone_variant_union",
        f"Orthographic tone-placement collision ({', '.join(item['raw_vi'] for item in items)}); "
        f"OR-merge takes the union of annotated emotions across variants "
        f"(annotated rows: {annotated_detail}). English-source rows do not necessarily "
        "carry a superset; the union preserves any annotated emotion.",
    )


def _build_log(key, items, decision, merged_vals, merged_masks, rationale):
    return {
        "key": key,
        "num_rows": len(items),
        "rows": ",".join(str(item["row"]) for item in items),
        "en_words": ",".join(item["en_word"] or "NONE" for item in items),
        "raw_vi_terms": ",".join(item["raw_vi"] for item in items),
        "decision": decision,
        "resolved_vector": ",".join(str(int(x)) for x in merged_vals),
        "resolved_mask": ",".join(str(m) for m in merged_masks),
        "candidate_vectors": " | ".join(
            f"row{item['row']}({item['raw_vi']},en={item['en_word'] or 'NONE'}):"
            f"[{','.join(str(int(x)) for x in item['vals'])}]"
            for item in items
        ),
        "rationale": rationale,
    }