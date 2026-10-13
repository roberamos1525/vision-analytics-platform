from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from vap.errors import (
    ChecksumMismatchError,
    ConflictError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    StorageError,
)
from vap.events.model import Event, EventType
from vap.registry.models import ModelRegistry, semver_key
from vap.storage.database import Database
from vap.storage.repositories import Repositories


@pytest.fixture
def repos(tmp_path: Path) -> Repositories:
    db = Database(tmp_path / "db" / "vap.db")
    yield Repositories(db)
    db.close()


class TestDatabase:
    def test_migrations_are_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "x.db"
        db = Database(path)
        assert db.schema_version == 1
        assert db.migrate() == 1
        db.close()
        db2 = Database(path)
        assert db2.schema_version == 1 and db2.ping()
        db2.close()
        assert not db2.ping()
        with pytest.raises(StorageError):
            db2.query("SELECT 1")

    def test_transaction_rolls_back(self, repos: Repositories) -> None:
        def insert_then_fail() -> None:
            with repos.db.transaction() as conn:
                conn.execute("INSERT INTO cameras VALUES ('c','c','{}',0,0)")
                raise RuntimeError("abort")

        with pytest.raises(RuntimeError):
            insert_then_fail()
        assert repos.cameras.list() == []
        with pytest.raises(StorageError):
            repos.db.query("SELECT * FROM nowhere")


class TestRepositories:
    def test_camera_and_pipeline_crud(self, repos: Repositories) -> None:
        repos.cameras.create("c1", "Front", {"kind": "synthetic"})
        with pytest.raises(ConflictError):
            repos.cameras.create("c1", "dup", {})
        assert repos.cameras.update("c1", "Gate", {"kind": "synthetic", "seed": 2}).name == "Gate"
        with pytest.raises(NotFoundError):
            repos.cameras.update("nope", "x", {})
        p = repos.pipelines.create("p1", "c1", {"a": 1})
        assert p.desired_state == "stopped"
        with pytest.raises(NotFoundError):
            repos.pipelines.create("p2", "missing", {})
        with pytest.raises(ConflictError):
            repos.pipelines.create("p1", "c1", {})
        with pytest.raises(ConflictError):
            repos.cameras.delete("c1")  # still referenced
        repos.pipelines.set_desired_state("p1", "running")
        assert repos.pipelines.update_config("p1", {"b": 2}).config == {"b": 2}
        assert repos.pipelines.get("p1").desired_state == "running"
        repos.pipelines.delete("p1")
        repos.cameras.delete("c1")
        with pytest.raises(NotFoundError):
            repos.cameras.get("c1")
        with pytest.raises(NotFoundError):
            repos.pipelines.delete("p1")

    def test_events_filters_and_pagination(self, repos: Repositories) -> None:
        for i in range(5):
            ev = Event(
                EventType.LOITERING if i % 2 else EventType.ZONE_INTRUSION,
                "p1" if i < 4 else "p2",
                "c",
                i,
                i,
                f"e{i}",
            )
            ev.created_at = 1000.0 + i
            repos.events.insert(ev)
            repos.events.insert(ev)  # idempotent
        assert len(repos.events.list(limit=100)) == 5
        page = repos.events.list(limit=2)
        assert [e["message"] for e in page] == ["e4", "e3"]
        nxt = repos.events.list(limit=2, before=page[-1]["created_at"])
        assert [e["message"] for e in nxt] == ["e2", "e1"]
        assert len(repos.events.list(pipeline_id="p1", event_type="loitering")) == 2
        assert len(repos.events.list(since=1002, until=1003)) == 2
        assert repos.events.counts_by_type("p1") == {"loitering": 2, "zone_intrusion": 2}
        one = repos.events.get(page[0]["event_id"])
        assert one["track_ids"] == [] and one["data"] == {}
        assert repos.events.purge_before(1002) == 2
        with pytest.raises(NotFoundError):
            repos.events.get("missing")
        with pytest.raises(InvalidInputError):
            repos.events.list(limit=0)

    def test_runs_tracks_metrics_evaluations(self, repos: Repositories) -> None:
        repos.runs.start("r1", "p1")
        repos.runs.finish("r1", 42, "completed")
        assert repos.runs.list("p1")[0]["frames"] == 42
        row = {
            "run_id": "r1",
            "track_id": 1,
            "pipeline_id": "p1",
            "label": "car",
            "first_frame": 0,
            "last_frame": 9,
            "first_ts": 0.0,
            "last_ts": 0.9,
            "hits": 10,
            "path_length": 30.0,
            "mean_speed": 33.3,
            "direction": "E",
            "last_bbox": [1, 2, 3, 4],
        }
        repos.tracks.insert_many([row])
        repos.tracks.insert_many([])
        assert repos.tracks.list("p1")[0]["last_bbox"] == [1, 2, 3, 4]
        assert repos.tracks.list(run_id="nope") == []
        now = time.time()
        repos.metrics.insert("p1", now - 100, {"fps": 1})
        repos.metrics.insert("p1", now, {"fps": 2})
        assert [m["fps"] for m in repos.metrics.list("p1")] == [2, 1]
        assert len(repos.metrics.list("p1", since=now - 1)) == 1
        assert repos.metrics.purge_before(now - 50) == 1
        repos.evaluations.create("e1", {"x": 1})
        repos.evaluations.complete("e1", {"mota": 0.9})
        repos.evaluations.create("e2", {})
        repos.evaluations.fail("e2", "boom")
        assert repos.evaluations.get("e1")["report"] == {"mota": 0.9}
        assert repos.evaluations.get("e2")["status"] == "failed"
        assert len(repos.evaluations.list()) == 2


@pytest.fixture
def registry(repos: Repositories, tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(repos.db, tmp_path / "models", max_bytes=1024)


class TestRegistry:
    def test_register_checksum_and_listing(self, registry: ModelRegistry) -> None:
        data = b"onnx-bytes" * 10
        m = registry.register(
            "yolo", "1.0.0", [data[:50], data[50:]], metadata={"class_names": ["car"]}
        )
        assert m.sha256 == hashlib.sha256(data).hexdigest() and m.size_bytes == 100
        assert m.path.read_bytes() == data and not m.active
        registry.register("yolo", "1.10.0", [b"b"])
        registry.register("yolo", "1.2.0-rc.1", [b"c"])
        assert [v.version for v in registry.list("yolo")] == ["1.0.0", "1.2.0-rc.1", "1.10.0"]
        assert registry.names() == [
            {"name": "yolo", "versions": 3, "active": None, "latest": "1.10.0"}
        ]

    def test_activation_is_exclusive_and_resolve(self, registry: ModelRegistry) -> None:
        registry.register("det", "1.0.0", [b"a"], activate=True)
        registry.register("det", "2.0.0", [b"b"])
        with pytest.raises(NotFoundError):
            registry.resolve("other")
        path, _ = registry.resolve("det")
        assert path.read_bytes() == b"a"
        registry.activate("det", "2.0.0")
        assert [m.version for m in registry.list("det") if m.active] == ["2.0.0"]
        assert registry.resolve("det", "1.0.0")[0].read_bytes() == b"a"
        registry.deactivate("det")
        assert registry.active("det") is None

    def test_verify_detects_corruption(self, registry: ModelRegistry) -> None:
        m = registry.register("m", "0.1.0", [b"original"])
        m.path.write_bytes(b"tampered")
        with pytest.raises(ChecksumMismatchError):
            registry.verify("m", "0.1.0")
        m.path.unlink()
        with pytest.raises(ChecksumMismatchError, match="missing"):
            registry.verify("m", "0.1.0")

    def test_upload_failures_leave_nothing_behind(
        self, registry: ModelRegistry, tmp_path: Path
    ) -> None:
        with pytest.raises(ChecksumMismatchError):
            registry.register("m", "1.0.0", [b"x"], expected_sha256="0" * 64)
        with pytest.raises(PayloadTooLargeError):
            registry.register("m", "1.0.0", [b"x" * 2000])
        with pytest.raises(InvalidInputError):
            registry.register("m", "1.0.0", [])
        assert registry.list() == []
        assert not list((tmp_path / "models").rglob("*.part"))
        registry.register("m", "1.0.0", [b"ok"])
        with pytest.raises(ConflictError):
            registry.register("m", "1.0.0", [b"again"])

    @pytest.mark.parametrize(
        ("name", "version", "kwargs"),
        [
            ("Bad Name", "1.0.0", {}),
            ("ok", "1.0", {}),
            ("ok", "1.0.0", {"fmt": "exe"}),
            ("ok", "1.0.0", {"metadata": {"class_names": [1]}}),
            ("ok", "1.0.0", {"metadata": {"input_size": [0, 5]}}),
            ("ok", "1.0.0", {"expected_sha256": "abc"}),
        ],
    )
    def test_validation(
        self, registry: ModelRegistry, name: str, version: str, kwargs: dict[str, object]
    ) -> None:
        with pytest.raises(InvalidInputError):
            registry.register(name, version, [b"x"], **kwargs)  # type: ignore[arg-type]

    def test_delete_and_register_file(self, registry: ModelRegistry, tmp_path: Path) -> None:
        src = tmp_path / "w.onnx"
        src.write_bytes(b"weights")
        m = registry.register_file("w", "1.0.0", src, activate=True)
        with pytest.raises(ConflictError):
            registry.delete("w", "1.0.0")
        registry.delete("w", "1.0.0", force=True)
        assert not m.path.exists() and not (tmp_path / "models" / "w").exists()
        with pytest.raises(InvalidInputError):
            registry.register_file("w", "1.0.0", tmp_path / "missing")
        assert semver_key("1.0.0") > semver_key("1.0.0-alpha")
