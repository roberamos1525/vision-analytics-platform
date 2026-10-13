"""SQLite (stdlib) database wrapper with versioned migrations.

A single connection is shared between threads and serialised with a
re-entrant lock; SQLite runs in WAL mode so readers from other processes
(e.g. ``sqlite3`` CLI) do not block the service.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vap.errors import StorageError

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE cameras (
            camera_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            spec_json   TEXT NOT NULL,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );
        CREATE TABLE pipelines (
            pipeline_id   TEXT PRIMARY KEY,
            camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE RESTRICT,
            config_json   TEXT NOT NULL,
            desired_state TEXT NOT NULL DEFAULT 'stopped',
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL
        );
        CREATE TABLE pipeline_runs (
            run_id       TEXT PRIMARY KEY,
            pipeline_id  TEXT NOT NULL,
            started_at   REAL NOT NULL,
            stopped_at   REAL,
            frames       INTEGER NOT NULL DEFAULT 0,
            status       TEXT NOT NULL,
            error        TEXT
        );
        CREATE INDEX idx_runs_pipeline ON pipeline_runs(pipeline_id, started_at);
        CREATE TABLE events (
            event_id     TEXT PRIMARY KEY,
            type         TEXT NOT NULL,
            severity     TEXT NOT NULL,
            pipeline_id  TEXT NOT NULL,
            camera_id    TEXT NOT NULL,
            rule_id      TEXT NOT NULL,
            frame_index  INTEGER NOT NULL,
            stream_time  REAL NOT NULL,
            created_at   REAL NOT NULL,
            message      TEXT NOT NULL,
            track_ids    TEXT NOT NULL,
            zone_id      TEXT,
            line_id      TEXT,
            data_json    TEXT NOT NULL,
            snapshot     TEXT
        );
        CREATE INDEX idx_events_pipeline ON events(pipeline_id, created_at);
        CREATE INDEX idx_events_type ON events(type, created_at);
        CREATE TABLE tracks (
            run_id       TEXT NOT NULL,
            track_id     INTEGER NOT NULL,
            pipeline_id  TEXT NOT NULL,
            label        TEXT NOT NULL,
            first_frame  INTEGER NOT NULL,
            last_frame   INTEGER NOT NULL,
            first_ts     REAL NOT NULL,
            last_ts      REAL NOT NULL,
            hits         INTEGER NOT NULL,
            path_length  REAL NOT NULL,
            mean_speed   REAL NOT NULL,
            direction    TEXT,
            last_bbox    TEXT NOT NULL,
            created_at   REAL NOT NULL,
            PRIMARY KEY (run_id, track_id)
        );
        CREATE INDEX idx_tracks_pipeline ON tracks(pipeline_id, created_at);
        CREATE TABLE metric_samples (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id  TEXT NOT NULL,
            ts           REAL NOT NULL,
            data_json    TEXT NOT NULL
        );
        CREATE INDEX idx_metrics_pipeline ON metric_samples(pipeline_id, ts);
        CREATE TABLE models (
            name           TEXT NOT NULL,
            version        TEXT NOT NULL,
            format         TEXT NOT NULL,
            sha256         TEXT NOT NULL,
            size_bytes     INTEGER NOT NULL,
            path           TEXT NOT NULL,
            metadata_json  TEXT NOT NULL,
            active         INTEGER NOT NULL DEFAULT 0,
            created_at     REAL NOT NULL,
            PRIMARY KEY (name, version)
        );
        CREATE UNIQUE INDEX idx_models_active ON models(name) WHERE active = 1;
        CREATE TABLE evaluations (
            evaluation_id  TEXT PRIMARY KEY,
            created_at     REAL NOT NULL,
            status         TEXT NOT NULL,
            request_json   TEXT NOT NULL,
            report_json    TEXT,
            error          TEXT
        );
        """,
    ),
]


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open database: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
        self.migrate()

    def migrate(self) -> int:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = int(row["v"] or 0)
            for version, sql in MIGRATIONS:
                if version <= current:
                    continue
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    for statement in filter(None, (s.strip() for s in sql.split(";"))):
                        self._conn.execute(statement)
                    self._conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)", (version,)
                    )
                    self._conn.execute("COMMIT")
                except sqlite3.Error as exc:
                    self._conn.execute("ROLLBACK")
                    raise StorageError(f"migration {version} failed: {exc}") from exc
                current = version
            return current

    @property
    def schema_version(self) -> int:
        row = self.query_one("SELECT MAX(version) AS v FROM schema_version")
        return int(row["v"]) if row and row["v"] is not None else 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._check_open()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock:
            self._check_open()
            try:
                cur = self._conn.execute(sql, params)
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc
            return cur.rowcount

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        with self.transaction() as conn:
            try:
                conn.executemany(sql, rows)
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            self._check_open()
            try:
                return list(self._conn.execute(sql, params).fetchall())
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def ping(self) -> bool:
        try:
            self.query_one("SELECT 1")
        except StorageError:
            return False
        return True

    def _check_open(self) -> None:
        if self._closed:
            raise StorageError("database is closed")

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True
