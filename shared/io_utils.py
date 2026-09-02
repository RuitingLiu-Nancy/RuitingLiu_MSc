#!/usr/bin/env python3
"""IO helpers: read the existing rebuild outputs, write redesigned outputs.

The graph-construction pipeline writes parquet
plus a *.preview.csv next to each parquet. We prefer parquet (clean, typed); if
no parquet engine is available we fall back to the preview CSV so the code still
runs in a minimal environment.

All multi-value label columns in the source use a '|' separator.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

SEP = "|"


def read_table(compiled_dir: Path, stem: str) -> pd.DataFrame:
    """Read <stem>.parquet, else <stem>.preview.csv."""
    pq = compiled_dir / f"{stem}.parquet"
    full_csv = compiled_dir / f"{stem}.csv"          # full CSV (our fallback writes)
    preview_csv = compiled_dir / f"{stem}.preview.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as exc:  # pragma: no cover - engine missing
            if not (full_csv.exists() or preview_csv.exists()):
                raise RuntimeError(
                    f"Cannot read {pq} (no parquet engine) and no CSV fallback. "
                    f"Install pyarrow. Underlying error: {exc}"
                ) from exc
            print(f"[io] parquet engine unavailable, falling back to CSV for {stem}")
    # prefer the full CSV (complete rows) over a truncated preview
    if full_csv.exists():
        return pd.read_csv(full_csv)
    if preview_csv.exists():
        return pd.read_csv(preview_csv)
    raise FileNotFoundError(f"None of {pq} / {full_csv} / {preview_csv} exists.")


def split_labels(value: object) -> list[str]:
    """Split a '|'-joined label cell into a clean list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [x.strip() for x in s.split(SEP) if x.strip()]


def join_labels(labels) -> str:
    return SEP.join(dict.fromkeys(x for x in labels if x))  # dedupe, keep order


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, preview_rows: int = 50) -> None:
    """Write parquet if possible (always write CSV preview for inspection)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote_parquet = False
    try:
        df.to_parquet(out_dir / f"{stem}.parquet", index=False)
        wrote_parquet = True
    except Exception as exc:  # pragma: no cover
        print(f"[io] parquet write unavailable for {stem} ({exc}); writing full CSV instead")
    # preview (or full CSV fallback)
    if wrote_parquet:
        df.head(preview_rows).to_csv(out_dir / f"{stem}.preview.csv", index=False)
    else:
        df.to_csv(out_dir / f"{stem}.csv", index=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write dictionaries as stable UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file using its header as the row schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    """Write dictionaries as CSV, preserving an explicit schema when supplied."""
    materialised = [dict(row) for row in rows]
    names = list(fieldnames or (materialised[0].keys() if materialised else []))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialised)


def write_json(path: Path, value: Any) -> None:
    """Write a stable, human-readable UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
