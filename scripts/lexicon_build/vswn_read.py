"""Read VSWN rows and collect canonical single-token candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter

from .keys import canonicalize_key


def read_candidates(source_path: Path, segmenter: VnCoreNLPSegmenter):
    candidates_by_key = defaultdict(list)
    dropped_multitoken: list[dict[str, Any]] = []
    data_rows = 0
    raw_terms_count = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            parsed = _parse_line(raw_line)
            if parsed is None:
                continue
            pos, synset_id, pos_score, neg_score, terms = parsed
            data_rows += 1
            pol = round(pos_score - neg_score, 6)
            pos_sense_id = f"{pos}#{synset_id}"
            term_tags = terms.split()
            raw_terms_count += len(term_tags)
            _collect_terms(
                candidates_by_key,
                dropped_multitoken,
                line_no,
                pos_sense_id,
                pol,
                terms,
                term_tags,
                segmenter,
            )
    return candidates_by_key, dropped_multitoken, data_rows, raw_terms_count


def _parse_line(raw_line: str):
    line = raw_line.rstrip("\r\n")
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) < 5:
        return None
    pos, synset_id, pos_s, neg_s, terms = parts[:5]
    try:
        pos_score = float(pos_s)
        neg_score = float(neg_s)
    except ValueError:
        return None
    return pos, synset_id, pos_score, neg_score, terms


def _collect_terms(
    candidates_by_key,
    dropped_multitoken: list[dict[str, Any]],
    line_no: int,
    pos_sense_id: str,
    pol: float,
    terms: str,
    term_tags: list[str],
    segmenter: VnCoreNLPSegmenter,
) -> None:
    for term_tag in term_tags:
        raw_term = term_tag.split("#")[0]
        key = canonicalize_key(raw_term, segmenter)
        if not key:
            continue
        if " " in key:
            dropped_multitoken.append(
                {
                    "source": "vietsentiwordnet",
                    "line": line_no,
                    "raw_term": raw_term,
                    "canonical_key": key,
                    "detail": f"{pos_sense_id} pol={pol}",
                }
            )
            continue
        is_standalone = _is_standalone(term_tag, terms, key, segmenter)
        candidates_by_key[key].append(
            {
                "pos_sense_id": pos_sense_id,
                "term_tag": term_tag,
                "score": pol,
                "is_standalone": is_standalone,
                "full_terms": terms,
            }
        )


def _is_standalone(
    term_tag: str,
    terms: str,
    key: str,
    segmenter: VnCoreNLPSegmenter,
) -> bool:
    if "#" not in term_tag:
        return False
    if canonicalize_key(term_tag.rsplit("#", 1)[0], segmenter) != key:
        return False
    return (
        f"không {term_tag}" not in terms
        and f"không_{term_tag}" not in terms
        and f"chưa {term_tag}" not in terms
        and f"kém {term_tag}" not in terms
        and f"thiếu {term_tag}" not in terms
        and f"mất {term_tag}" not in terms
        and f"phi {term_tag}" not in terms
        and f"đừng {term_tag}" not in terms
    )