"""CSV / JSONL export of events, tracks and metric samples.

Writers are generators of text chunks so the API can stream large exports
without materialising them, and the CLI can write them to files. CSV cells
that start with formula characters are prefixed with a quote to prevent CSV
injection when exports are opened in spreadsheet software.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from vap.errors import InvalidInputError

ExportFormat = Literal["csv", "jsonl"]
ExportKind = Literal["events", "tracks", "metrics"]

EVENT_COLUMNS = (
    "event_id",
    "type",
    "severity",
    "pipeline_id",
    "camera_id",
    "rule_id",
    "frame_index",
    "stream_time",
    "created_at",
    "message",
    "track_ids",
    "zone_id",
    "line_id",
    "data",
    "snapshot",
)
TRACK_COLUMNS = (
    "run_id",
    "pipeline_id",
    "track_id",
    "label",
    "first_frame",
    "last_frame",
    "first_ts",
    "last_ts",
    "hits",
    "path_length",
    "mean_speed",
    "direction",
    "last_bbox",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict | list | tuple):
        value = json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        # Numbers like "-1.5" are safe and must stay numeric.
        try:
            float(value)
        except ValueError:
            return "'" + value
    return value


def flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with dotted keys (lists are kept as JSON cells)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def iter_csv(rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> Iterator[str]:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    yield buf.getvalue()
    for row in rows:
        buf.seek(0)
        buf.truncate()
        writer.writerow([_cell(row.get(c)) for c in columns])
        yield buf.getvalue()


def iter_jsonl(rows: Iterable[dict[str, Any]]) -> Iterator[str]:
    for row in rows:
        yield json.dumps(row, separators=(",", ":"), default=str) + "\n"


def metric_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Union of flattened keys across metric samples, ``ts`` first."""
    keys: dict[str, None] = {"ts": None}
    for row in rows:
        for k in flatten(row):
            keys.setdefault(k, None)
    return list(keys)


def render(rows: Iterable[dict[str, Any]], kind: ExportKind, fmt: ExportFormat) -> Iterator[str]:
    if fmt == "jsonl":
        return iter_jsonl(rows)
    if fmt != "csv":
        raise InvalidInputError(f"unsupported export format {fmt!r}")
    if kind == "events":
        return iter_csv(rows, EVENT_COLUMNS)
    if kind == "tracks":
        return iter_csv(rows, TRACK_COLUMNS)
    if kind == "metrics":
        materialised = [flatten(r) for r in rows]
        return iter_csv(materialised, metric_columns(materialised))
    raise InvalidInputError(f"unknown export kind {kind!r}")


def write_export(
    path: str | Path, rows: Iterable[dict[str, Any]], kind: ExportKind, fmt: ExportFormat
) -> int:
    """Write an export file atomically; returns the number of data rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        for i, chunk in enumerate(render(rows, kind, fmt)):
            fh.write(chunk)
            if fmt == "jsonl" or i > 0:
                count += 1
    tmp.replace(target)
    return count
