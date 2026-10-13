from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from vap.errors import InvalidInputError
from vap.exporting import flatten, iter_csv, metric_columns, render, write_export
from vap.extraction import ExtractionSpec, FrameSelector, extract_frames, histogram_signature
from vap.sources.synthetic import SceneSpec, SyntheticObject, SyntheticScene


class TestExport:
    def test_csv_injection_guard_and_json_cells(self) -> None:
        rows = [{"a": "=SUM(A1)", "b": [1, 2], "c": "-1.5", "d": None, "e": "@cmd"}]
        text = "".join(iter_csv(rows, ["a", "b", "c", "d", "e"]))
        parsed = list(csv.reader(io.StringIO(text)))
        assert parsed[1] == ["'=SUM(A1)", "[1,2]", "-1.5", "", "'@cmd"]

    def test_metrics_flattening(self) -> None:
        rows = [{"ts": 1, "fps": 2, "zones": {"z1": 3}}, {"ts": 2, "lines": {"l": 1}}]
        flat = [flatten(r) for r in rows]
        assert flat[0] == {"ts": 1, "fps": 2, "zones.z1": 3}
        assert metric_columns(flat) == ["ts", "fps", "zones.z1", "lines.l"]
        text = "".join(render(rows, "metrics", "csv"))
        assert text.splitlines()[0] == "ts,fps,zones.z1,lines.l"

    def test_write_export_counts_rows(self, tmp_path: Path) -> None:
        rows = [{"event_id": str(i), "type": "loitering"} for i in range(3)]
        assert write_export(tmp_path / "e.csv", rows, "events", "csv") == 3
        assert write_export(tmp_path / "e.jsonl", rows, "events", "jsonl") == 3
        assert json.loads((tmp_path / "e.jsonl").read_text().splitlines()[2])["event_id"] == "2"
        assert (tmp_path / "e.csv").read_text().startswith("event_id,type")
        tracks = "".join(render([{"track_id": 1, "last_bbox": [1, 2]}], "tracks", "csv"))
        assert '"[1,2]"' in tracks
        with pytest.raises(InvalidInputError):
            list(render([], "events", "xml"))  # type: ignore[arg-type]
        with pytest.raises(InvalidInputError):
            list(render([], "people", "csv"))  # type: ignore[arg-type]


def cut_scene() -> SyntheticScene:
    """Static red square, then a hard cut to a different background at frame 10."""
    spec = SceneSpec(
        width=64,
        height=64,
        fps=10,
        num_frames=20,
        background=(20, 20, 20),
        objects=[
            SyntheticObject(1, 10, 10, 20, 20, color=(220, 30, 30), end_frame=9),
            SyntheticObject(2, 0, 0, 64, 64, color=(30, 30, 220), start_frame=10),
        ],
    )
    return SyntheticScene(spec)


class TestExtraction:
    def test_every_n_and_interval(self) -> None:
        frames = cut_scene().frames()
        every = [s.frame.index for s in FrameSelector(ExtractionSpec(every_n=4)).select(frames)]
        assert every == [0, 4, 8, 12, 16]
        interval = [
            s.frame.index
            for s in FrameSelector(ExtractionSpec(mode="interval", interval_s=0.7)).select(frames)
        ]
        assert interval == [0, 7, 14]
        limited = list(FrameSelector(ExtractionSpec(every_n=1, max_frames=3)).select(frames))
        assert len(limited) == 3

    def test_scene_change_finds_cut(self) -> None:
        frames = cut_scene().frames()
        spec = ExtractionSpec(mode="scene_change", scene_threshold=0.3, min_gap_frames=2)
        picks = [(s.frame.index, s.reason) for s in FrameSelector(spec).select(frames)]
        assert picks == [(0, "first"), (10, "scene_change")]

    def test_histogram_signature_normalised(self) -> None:
        sig = histogram_signature(cut_scene().render(0))
        assert sig.shape == (48,) and sig[:16].sum() == pytest.approx(1.0)

    def test_extract_writes_files_and_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        spec = ExtractionSpec(every_n=5, image_format="png", resize_width=32)
        result = extract_frames(cut_scene().frames(), spec, out, backend="numpy")
        assert result["count"] == 4
        names = sorted(p.name for p in out.glob("*.png"))
        assert names[0] == "frame_000000.png"
        manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
        assert [m["frame_index"] for m in manifest] == [0, 5, 10, 15]
        with pytest.raises(InvalidInputError):
            extract_frames(cut_scene().frames(), spec, out)
