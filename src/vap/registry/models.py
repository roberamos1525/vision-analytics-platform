"""Model registry: versioned model artefacts with checksums and activation.

Artefacts are streamed to ``<models_dir>/<name>/<version>/model.<ext>`` via a
temporary file, hashed with SHA-256 while writing, and recorded in SQLite.
At most one version per model name is *active* (enforced by a partial unique
index); detectors referencing a model without a version use the active one.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vap.errors import (
    ChecksumMismatchError,
    ConflictError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
)
from vap.storage.database import Database

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$")
FORMATS = {"onnx": "onnx", "npz": "npz", "json": "json", "bin": "bin"}
MAX_METADATA_BYTES = 16_384


def semver_key(version: str) -> tuple[int, int, int, int, str]:
    m = SEMVER_RE.match(version)
    if not m:
        raise InvalidInputError(f"invalid semantic version {version!r}")
    major, minor, patch, pre = m.groups()
    # Releases sort after pre-releases of the same version.
    return (int(major), int(minor), int(patch), 0 if pre else 1, pre or "")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelVersion:
    name: str
    version: str
    format: str
    sha256: str
    size_bytes: int
    path: Path
    metadata: dict[str, Any]
    active: bool
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "active": self.active,
            "created_at": round(self.created_at, 3),
        }


class ModelRegistry:
    def __init__(
        self, db: Database, models_dir: str | Path, *, max_bytes: int = 500 * 1024 * 1024
    ) -> None:
        self.db = db
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    # -- validation -----------------------------------------------------------------

    @staticmethod
    def validate_name(name: str) -> str:
        if not NAME_RE.match(name):
            raise InvalidInputError(
                "model name must match ^[a-z0-9][a-z0-9_.-]{0,63}$", details={"name": name}
            )
        return name

    @staticmethod
    def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        meta = dict(metadata or {})
        if len(json.dumps(meta)) > MAX_METADATA_BYTES:
            raise InvalidInputError("model metadata is too large")
        names = meta.get("class_names")
        if names is not None and (
            not isinstance(names, list) or not all(isinstance(n, str) and n for n in names)
        ):
            raise InvalidInputError("metadata.class_names must be a list of non-empty strings")
        size = meta.get("input_size")
        if size is not None and (
            not isinstance(size, list)
            or len(size) != 2
            or not all(isinstance(v, int) and 1 <= v <= 8192 for v in size)
        ):
            raise InvalidInputError("metadata.input_size must be [width, height]")
        return meta

    # -- persistence helpers ------------------------------------------------------------

    def _row(self, row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            name=row["name"],
            version=row["version"],
            format=row["format"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            path=Path(row["path"]),
            metadata=json.loads(row["metadata_json"]),
            active=bool(row["active"]),
            created_at=float(row["created_at"]),
        )

    def _artifact_path(self, name: str, version: str, fmt: str) -> Path:
        return self.models_dir / name / version / f"model.{FORMATS[fmt]}"

    # -- operations -----------------------------------------------------------------------

    def register(
        self,
        name: str,
        version: str,
        chunks: Iterable[bytes],
        *,
        fmt: str = "onnx",
        metadata: dict[str, Any] | None = None,
        expected_sha256: str | None = None,
        activate: bool = False,
    ) -> ModelVersion:
        """Store a new model version from a stream of byte chunks."""
        self.validate_name(name)
        semver_key(version)
        if fmt not in FORMATS:
            raise InvalidInputError(
                f"unsupported model format {fmt!r}", details={"allowed": sorted(FORMATS)}
            )
        meta = self.validate_metadata(metadata)
        if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise InvalidInputError("expected sha256 must be 64 lowercase hex characters")
        if self.db.query_one(
            "SELECT 1 FROM models WHERE name = ? AND version = ?", (name, version)
        ):
            raise ConflictError(f"model {name}@{version} already exists")

        final = self._artifact_path(name, version, fmt)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as fh:
                for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise PayloadTooLargeError(
                            "model artefact exceeds the size limit",
                            details={"max_bytes": self.max_bytes},
                        )
                    digest.update(chunk)
                    fh.write(chunk)
            if size == 0:
                raise InvalidInputError("model artefact is empty")
            checksum = digest.hexdigest()
            if expected_sha256 is not None and checksum != expected_sha256:
                raise ChecksumMismatchError(
                    "uploaded artefact does not match the expected checksum",
                    details={"expected": expected_sha256, "actual": checksum},
                )
            tmp.replace(final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            self._cleanup_dirs(final.parent)
            raise

        now = time.time()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO models (name, version, format, sha256, size_bytes, path, metadata_json, "
                    "active, created_at) VALUES (?,?,?,?,?,?,?,0,?)",
                    (name, version, fmt, checksum, size, str(final), json.dumps(meta), now),
                )
        except sqlite3.IntegrityError as exc:
            final.unlink(missing_ok=True)
            raise ConflictError(f"model {name}@{version} already exists") from exc
        if activate:
            return self.activate(name, version)
        return self.get(name, version)

    def register_file(
        self, name: str, version: str, path: str | Path, **kwargs: Any
    ) -> ModelVersion:
        src = Path(path)
        if not src.is_file():
            raise InvalidInputError(f"model file not found: {src}")

        def reader() -> Iterable[bytes]:
            with src.open("rb") as fh:
                while block := fh.read(1 << 20):
                    yield block

        return self.register(name, version, reader(), **kwargs)

    def get(self, name: str, version: str) -> ModelVersion:
        row = self.db.query_one(
            "SELECT * FROM models WHERE name = ? AND version = ?", (name, version)
        )
        if row is None:
            raise NotFoundError(f"model {name}@{version} not found")
        return self._row(row)

    def list(self, name: str | None = None) -> list[ModelVersion]:
        if name is None:
            rows = self.db.query("SELECT * FROM models")
        else:
            rows = self.db.query("SELECT * FROM models WHERE name = ?", (name,))
        versions = [self._row(r) for r in rows]
        return sorted(versions, key=lambda m: (m.name, semver_key(m.version)))

    def names(self) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for m in self.list():
            entry = out.setdefault(
                m.name, {"name": m.name, "versions": 0, "active": None, "latest": None}
            )
            entry["versions"] += 1
            entry["latest"] = m.version
            if m.active:
                entry["active"] = m.version
        return list(out.values())

    def active(self, name: str) -> ModelVersion | None:
        row = self.db.query_one("SELECT * FROM models WHERE name = ? AND active = 1", (name,))
        return self._row(row) if row else None

    def activate(self, name: str, version: str) -> ModelVersion:
        self.get(name, version)
        with self.db.transaction() as conn:
            conn.execute("UPDATE models SET active = 0 WHERE name = ? AND active = 1", (name,))
            conn.execute(
                "UPDATE models SET active = 1 WHERE name = ? AND version = ?", (name, version)
            )
        return self.get(name, version)

    def deactivate(self, name: str) -> None:
        self.db.execute("UPDATE models SET active = 0 WHERE name = ?", (name,))

    def verify(self, name: str, version: str) -> ModelVersion:
        model = self.get(name, version)
        if not model.path.is_file():
            raise ChecksumMismatchError(f"artefact for {name}@{version} is missing on disk")
        actual = sha256_file(model.path)
        if actual != model.sha256:
            raise ChecksumMismatchError(
                f"artefact for {name}@{version} is corrupted",
                details={"expected": model.sha256, "actual": actual},
            )
        return model

    def resolve(self, name: str, version: str | None = None) -> tuple[Path, dict[str, Any]]:
        """Return the verified artefact path and metadata (active version by default)."""
        if version is None:
            model = self.active(name)
            if model is None:
                raise NotFoundError(f"model {name!r} has no active version")
            version = model.version
        model = self.verify(name, version)
        return model.path, model.metadata

    def delete(self, name: str, version: str, *, force: bool = False) -> None:
        model = self.get(name, version)
        if model.active and not force:
            raise ConflictError(
                f"model {name}@{version} is active; deactivate it first or use force"
            )
        self.db.execute("DELETE FROM models WHERE name = ? AND version = ?", (name, version))
        model.path.unlink(missing_ok=True)
        self._cleanup_dirs(model.path.parent)

    def _cleanup_dirs(self, directory: Path) -> None:
        root = self.models_dir.resolve()
        current = directory.resolve()
        while current != root and root in current.parents:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent
