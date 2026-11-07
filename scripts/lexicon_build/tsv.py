"""Deterministic TSV writer for conflict and dropped-key logs."""

from pathlib import Path
from typing import Any


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(fieldnames) + "\n")
        for row in rows:
            values = [
                str(row.get(column, "")).replace("\t", " ").replace("\r", " ").replace("\n", " ")
                for column in fieldnames
            ]
            handle.write("\t".join(values) + "\n")