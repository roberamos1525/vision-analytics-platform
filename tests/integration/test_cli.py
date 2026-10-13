"""CLI commands exercised through ``vap.cli.main``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import requires_opencv
from vap.cli import main, parse_synthetic, source_from_arg
from vap.errors import InvalidInputError
from vap.events.model import Event, EventType
from vap.storage.database import Database
from vap.storage.repositories import Repositories


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, object]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def test_generate_analyze_evaluate_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gen = tmp_path / "gen"
    code, out = run(capsys, "generate", "--out", str(gen), "--params", "seed=2,objects=2,frames=80")
    assert code == 0 and out["frames"] == 80  # type: ignore[index]
    assert len(list(gen.glob("*.png"))) == 80

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "detector": {"kind": "background", "warmup_frames": 10},
                "tracker": {"min_hits": 2},
                "zones": [
                    {
                        "zone_id": "all",
                        "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "normalized": True,
                    }
                ],
                "rules": [{"kind": "zone_intrusion", "rule_id": "r", "zone_id": "all"}],
            }
        )
    )
    preds = tmp_path / "preds.jsonl"
    events = tmp_path / "events.jsonl"
    code, summary = run(
        capsys,
        "analyze",
        str(gen),
        "--config",
        str(cfg),
        "--out",
        str(preds),
        "--events",
        str(events),
        "--backend",
        "numpy",
        "--fps",
        "15",
    )
    assert code == 0 and summary["frames"] == 80 and summary["events"] >= 1  # type: ignore[index]
    assert events.read_text().count("zone_intrusion") == summary["events"]  # type: ignore[index]

    code, report = run(
        capsys,
        "evaluate",
        "--gt",
        str(gen / "ground_truth.jsonl"),
        "--pred",
        str(preds),
        "--iou",
        "0.3,0.5",
    )
    assert code == 0
    assert report["tracking"]["mota"] > 0.3  # type: ignore[index]
    assert len(report["detection"]["thresholds"]) == 2  # type: ignore[index]


def test_evaluate_synthetic_request_and_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"scene": {"num_frames": 60, "num_objects": 2}, "backend": "numpy"}))
    code, report = run(capsys, "evaluate", "--request", str(req))
    assert code == 0 and "tracking" in report  # type: ignore[operator]
    code = main(["evaluate", "--request", str(req), "--min-mota", "1.01"])
    capsys.readouterr()
    assert code == 2
    assert main(["evaluate", "--gt", str(req)]) == 1


def test_extract_and_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "frames"
    code, res = run(
        capsys,
        "extract",
        "synthetic:frames=30,objects=1",
        "--out",
        str(out),
        "--every-n",
        "10",
        "--image-format",
        "png",
    )
    assert code == 0 and res["count"] == 3  # type: ignore[index]
    image = sorted(out.glob("*.png"))[0]
    edges = tmp_path / "edges.png"
    code, res = run(
        capsys,
        "process",
        str(image),
        "--ops",
        '[{"op":"gray"},{"op":"canny"}]',
        "--out",
        str(edges),
    )
    assert code == 0 and edges.is_file() and res["steps"][1]["op"] == "canny"  # type: ignore[index]


@requires_opencv
@pytest.mark.opencv
def test_generate_avi_and_extract_scene_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run(
        capsys, "generate", "--out", str(tmp_path / "v"), "--format", "avi", "--params", "frames=20"
    )
    assert code == 0 and (tmp_path / "v" / "video.avi").is_file()
    code, res = run(
        capsys,
        "extract",
        str(tmp_path / "v" / "video.avi"),
        "--out",
        str(tmp_path / "x"),
        "--mode",
        "scene_change",
    )
    assert code == 0 and res["count"] >= 1  # type: ignore[index]


def test_models_export_and_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    cfg = tmp_path / "vap.toml"
    cfg.write_text(f'[auth]\nenabled = false\n[storage]\nstate_dir = "{state.as_posix()}"\n')
    monkeypatch.setenv("VAP_WEBHOOK_SECRET", "hidden")
    weights = tmp_path / "w.onnx"
    weights.write_bytes(b"weights")
    code, model = run(
        capsys,
        "models",
        "register",
        "--name",
        "w",
        "--version",
        "1.0.0",
        "--file",
        str(weights),
        "--metadata",
        '{"class_names": ["a"]}',
        "--activate",
        "--config",
        str(cfg),
    )
    assert code == 0 and model["active"]  # type: ignore[index]
    code, listing = run(capsys, "models", "list", "--config", str(cfg))
    assert [m["version"] for m in listing] == ["1.0.0"]  # type: ignore[union-attr]
    code, _ = run(
        capsys, "models", "verify", "--name", "w", "--version", "1.0.0", "--config", str(cfg)
    )
    assert code == 0

    db = Database(state / "vap.db")
    Repositories(db).events.insert(Event(EventType.CROWDING, "p", "c", 1, 0.1, "crowd"))
    db.close()
    out = tmp_path / "events.csv"
    code, res = run(capsys, "export", "events", "--out", str(out), "--config", str(cfg))
    assert code == 0 and res["rows"] == 1 and "crowding" in out.read_text()  # type: ignore[index]
    assert main(["export", "metrics", "--out", str(out), "--config", str(cfg)]) == 1

    code, effective = run(capsys, "config", "--config", str(cfg))
    assert effective["webhooks"]["secret"] == "***"  # type: ignore[index]


def test_argument_and_config_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        main(["models", "activate"])
    with pytest.raises(SystemExit):
        main(["models", "register", "--name", "a", "--version", "1.0.0"])
    monkeypatch.delenv("VAP_API_KEYS", raising=False)
    monkeypatch.delenv("VAP_AUTH__ENABLED", raising=False)
    assert main(["config", "--config", str(tmp_path / "missing.toml")]) == 3
    assert main(["process", str(tmp_path / "nope.png"), "--ops", "[]", "--out", "x.png"]) != 0
    capsys.readouterr()


def test_source_parsing(tmp_path: Path) -> None:
    spec = parse_synthetic("synthetic:seed=4,objects=2,frames=10,noise=1.5")
    assert spec.seed == 4 and spec.num_objects == 2 and spec.noise_std == 1.5
    with pytest.raises(InvalidInputError):
        parse_synthetic("synthetic:seed")
    with pytest.raises(InvalidInputError):
        parse_synthetic("synthetic:width=2")
    with pytest.raises(InvalidInputError):
        source_from_arg(str(tmp_path / "missing"))
    assert source_from_arg(str(tmp_path)).kind == "images"
