"""Repositories: thin, typed data-access helpers over :class:`Database`."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from vap.errors import ConflictError, InvalidInputError, NotFoundError
from vap.events.model import Event
from vap.storage.database import Database

MAX_PAGE = 1000


def _limit(limit: int) -> int:
    if limit < 1 or limit > MAX_PAGE:
        raise InvalidInputError(f"limit must be within [1, {MAX_PAGE}]")
    return limit


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else None


@dataclass(frozen=True, slots=True)
class CameraRecord:
    camera_id: str
    name: str
    spec: dict[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class PipelineRecord:
    pipeline_id: str
    camera_id: str
    config: dict[str, Any]
    desired_state: str
    created_at: float
    updated_at: float


class CameraRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _row(row: sqlite3.Row) -> CameraRecord:
        return CameraRecord(
            row["camera_id"],
            row["name"],
            json.loads(row["spec_json"]),
            row["created_at"],
            row["updated_at"],
        )

    def create(self, camera_id: str, name: str, spec: dict[str, Any]) -> CameraRecord:
        now = time.time()
        try:
            self.db.execute(
                "INSERT INTO cameras (camera_id, name, spec_json, created_at, updated_at) VALUES (?,?,?,?,?)",
                (camera_id, name, json.dumps(spec), now, now),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"camera {camera_id!r} already exists") from exc
        return CameraRecord(camera_id, name, spec, now, now)

    def update(self, camera_id: str, name: str, spec: dict[str, Any]) -> CameraRecord:
        now = time.time()
        n = self.db.execute(
            "UPDATE cameras SET name = ?, spec_json = ?, updated_at = ? WHERE camera_id = ?",
            (name, json.dumps(spec), now, camera_id),
        )
        if n == 0:
            raise NotFoundError(f"camera {camera_id!r} not found")
        return self.get(camera_id)

    def get(self, camera_id: str) -> CameraRecord:
        row = self.db.query_one("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        if row is None:
            raise NotFoundError(f"camera {camera_id!r} not found")
        return self._row(row)

    def list(self) -> list[CameraRecord]:
        return [self._row(r) for r in self.db.query("SELECT * FROM cameras ORDER BY camera_id")]

    def delete(self, camera_id: str) -> None:
        try:
            n = self.db.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"camera {camera_id!r} is used by a pipeline") from exc
        if n == 0:
            raise NotFoundError(f"camera {camera_id!r} not found")


class PipelineRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _row(row: sqlite3.Row) -> PipelineRecord:
        return PipelineRecord(
            row["pipeline_id"],
            row["camera_id"],
            json.loads(row["config_json"]),
            row["desired_state"],
            row["created_at"],
            row["updated_at"],
        )

    def create(self, pipeline_id: str, camera_id: str, config: dict[str, Any]) -> PipelineRecord:
        now = time.time()
        try:
            self.db.execute(
                "INSERT INTO pipelines (pipeline_id, camera_id, config_json, desired_state, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?)",
                (pipeline_id, camera_id, json.dumps(config), "stopped", now, now),
            )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc):
                raise NotFoundError(f"camera {camera_id!r} not found") from exc
            raise ConflictError(f"pipeline {pipeline_id!r} already exists") from exc
        return self.get(pipeline_id)

    def get(self, pipeline_id: str) -> PipelineRecord:
        row = self.db.query_one("SELECT * FROM pipelines WHERE pipeline_id = ?", (pipeline_id,))
        if row is None:
            raise NotFoundError(f"pipeline {pipeline_id!r} not found")
        return self._row(row)

    def list(self) -> list[PipelineRecord]:
        return [self._row(r) for r in self.db.query("SELECT * FROM pipelines ORDER BY pipeline_id")]

    def update_config(self, pipeline_id: str, config: dict[str, Any]) -> PipelineRecord:
        n = self.db.execute(
            "UPDATE pipelines SET config_json = ?, updated_at = ? WHERE pipeline_id = ?",
            (json.dumps(config), time.time(), pipeline_id),
        )
        if n == 0:
            raise NotFoundError(f"pipeline {pipeline_id!r} not found")
        return self.get(pipeline_id)

    def set_desired_state(self, pipeline_id: str, state: str) -> None:
        n = self.db.execute(
            "UPDATE pipelines SET desired_state = ?, updated_at = ? WHERE pipeline_id = ?",
            (state, time.time(), pipeline_id),
        )
        if n == 0:
            raise NotFoundError(f"pipeline {pipeline_id!r} not found")

    def delete(self, pipeline_id: str) -> None:
        if self.db.execute("DELETE FROM pipelines WHERE pipeline_id = ?", (pipeline_id,)) == 0:
            raise NotFoundError(f"pipeline {pipeline_id!r} not found")


class RunRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def start(self, run_id: str, pipeline_id: str) -> None:
        self.db.execute(
            "INSERT INTO pipeline_runs (run_id, pipeline_id, started_at, status) VALUES (?,?,?,?)",
            (run_id, pipeline_id, time.time(), "running"),
        )

    def finish(self, run_id: str, frames: int, status: str, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE pipeline_runs SET stopped_at = ?, frames = ?, status = ?, error = ? WHERE run_id = ?",
            (time.time(), frames, status, error, run_id),
        )

    def list(self, pipeline_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM pipeline_runs WHERE pipeline_id = ? ORDER BY started_at DESC LIMIT ?",
            (pipeline_id, _limit(limit)),
        )
        return [dict(r) for r in rows]


class EventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, event: Event) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO events (event_id, type, severity, pipeline_id, camera_id, rule_id, "
            "frame_index, stream_time, created_at, message, track_ids, zone_id, line_id, data_json, "
            "snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.event_id,
                event.type.value,
                event.severity.value,
                event.pipeline_id,
                event.camera_id,
                event.rule_id,
                event.frame_index,
                event.stream_time,
                event.created_at,
                event.message,
                json.dumps(event.track_ids),
                event.zone_id,
                event.line_id,
                json.dumps(event.data),
                event.snapshot,
            ),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["track_ids"] = _loads(out.pop("track_ids")) or []
        out["data"] = _loads(out.pop("data_json")) or {}
        return out

    def get(self, event_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM events WHERE event_id = ?", (event_id,))
        if row is None:
            raise NotFoundError(f"event {event_id!r} not found")
        return self._row(row)

    def list(
        self,
        *,
        pipeline_id: str | None = None,
        camera_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        before: float | None = None,
        limit: int = 100,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("pipeline_id", pipeline_id),
            ("camera_id", camera_id),
            ("type", event_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("created_at <= ?")
            params.append(until)
        if before is not None:
            clauses.append("created_at < ?")
            params.append(before)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "ASC" if ascending else "DESC"
        params.append(_limit(limit))
        rows = self.db.query(
            f"SELECT * FROM events {where} ORDER BY created_at {order}, event_id {order} LIMIT ?",
            params,
        )
        return [self._row(r) for r in rows]

    def counts_by_type(self, pipeline_id: str | None = None) -> dict[str, int]:
        if pipeline_id is None:
            rows = self.db.query("SELECT type, COUNT(*) AS n FROM events GROUP BY type")
        else:
            rows = self.db.query(
                "SELECT type, COUNT(*) AS n FROM events WHERE pipeline_id = ? GROUP BY type",
                (pipeline_id,),
            )
        return {r["type"]: int(r["n"]) for r in rows}

    def purge_before(self, cutoff: float) -> int:
        return self.db.execute("DELETE FROM events WHERE created_at < ?", (cutoff,))


class TrackRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = time.time()
        self.db.executemany(
            "INSERT OR REPLACE INTO tracks (run_id, track_id, pipeline_id, label, first_frame, last_frame, "
            "first_ts, last_ts, hits, path_length, mean_speed, direction, last_bbox, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["run_id"],
                    r["track_id"],
                    r["pipeline_id"],
                    r["label"],
                    r["first_frame"],
                    r["last_frame"],
                    r["first_ts"],
                    r["last_ts"],
                    r["hits"],
                    r["path_length"],
                    r["mean_speed"],
                    r.get("direction"),
                    json.dumps(r["last_bbox"]),
                    now,
                )
                for r in rows
            ],
        )

    def list(
        self, pipeline_id: str | None = None, run_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if pipeline_id is not None:
            clauses.append("pipeline_id = ?")
            params.append(pipeline_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(_limit(limit))
        rows = self.db.query(
            f"SELECT * FROM tracks {where} ORDER BY created_at DESC, track_id LIMIT ?", params
        )
        out = []
        for r in rows:
            d = dict(r)
            d["last_bbox"] = _loads(d["last_bbox"])
            out.append(d)
        return out


class MetricRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, pipeline_id: str, ts: float, data: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO metric_samples (pipeline_id, ts, data_json) VALUES (?,?,?)",
            (pipeline_id, ts, json.dumps(data)),
        )

    def list(
        self, pipeline_id: str, since: float | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        params: list[Any] = [pipeline_id]
        extra = ""
        if since is not None:
            extra = "AND ts >= ?"
            params.append(since)
        params.append(_limit(limit))
        rows = self.db.query(
            f"SELECT ts, data_json FROM metric_samples WHERE pipeline_id = ? {extra} ORDER BY ts DESC LIMIT ?",
            params,
        )
        return [{"ts": r["ts"], **json.loads(r["data_json"])} for r in rows]

    def purge_before(self, cutoff: float) -> int:
        return self.db.execute("DELETE FROM metric_samples WHERE ts < ?", (cutoff,))


class EvaluationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, evaluation_id: str, request: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO evaluations (evaluation_id, created_at, status, request_json) VALUES (?,?,?,?)",
            (evaluation_id, time.time(), "running", json.dumps(request)),
        )

    def complete(self, evaluation_id: str, report: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE evaluations SET status = 'completed', report_json = ? WHERE evaluation_id = ?",
            (json.dumps(report), evaluation_id),
        )

    def fail(self, evaluation_id: str, error: str) -> None:
        self.db.execute(
            "UPDATE evaluations SET status = 'failed', error = ? WHERE evaluation_id = ?",
            (error, evaluation_id),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "evaluation_id": row["evaluation_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "request": _loads(row["request_json"]),
            "report": _loads(row["report_json"]),
            "error": row["error"],
        }

    def get(self, evaluation_id: str) -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT * FROM evaluations WHERE evaluation_id = ?", (evaluation_id,)
        )
        if row is None:
            raise NotFoundError(f"evaluation {evaluation_id!r} not found")
        return self._row(row)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (_limit(limit),)
        )
        return [self._row(r) for r in rows]


class Repositories:
    """Bundle of all repositories sharing one database."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.cameras = CameraRepository(db)
        self.pipelines = PipelineRepository(db)
        self.runs = RunRepository(db)
        self.events = EventRepository(db)
        self.tracks = TrackRepository(db)
        self.metrics = MetricRepository(db)
        self.evaluations = EvaluationRepository(db)
