from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeCapture, ScriptedFactory
from tests.helpers import requires_opencv, two_object_scene
from vap.errors import InvalidInputError, PermissionDeniedError, SourceError
from vap.imaging.codec import save_image
from vap.sources.base import SourceState
from vap.sources.image_sequence import ImageSequenceSource, natural_key
from vap.sources.rtsp import RtspSource
from vap.sources.specs import (
    FileSourceSpec,
    ImageSequenceSourceSpec,
    RtspSourceSpec,
    SyntheticSourceSpec,
    build_source,
    public_spec,
    resolve_media_path,
)
from vap.sources.synthetic import (
    SceneSpec,
    SyntheticObject,
    SyntheticScene,
    SyntheticSource,
    random_scene,
)
from vap.sources.video_file import VideoFileSource, write_video


class TestSyntheticScene:
    def test_ground_truth_matches_rendered_pixels(self) -> None:
        scene = SyntheticScene(two_object_scene())
        img = scene.render(10)
        for gt in scene.ground_truth(10):
            x1, y1, x2, y2 = (round(v) for v in gt.bbox.to_list())
            patch = img[y1:y2, x1:x2]
            obj = next(o for o in scene.spec.objects if o.object_id == gt.object_id)
            assert np.all(patch == np.array(obj.color, np.uint8))

    def test_deterministic_and_random_access(self) -> None:
        spec = random_scene(seed=5, num_objects=3, noise_std=4.0)
        a, b = SyntheticScene(spec), SyntheticScene(spec)
        assert np.array_equal(a.render(17), b.render(17))
        assert not np.array_equal(a.render(17), a.render(18))

    def test_bounce_keeps_objects_inside(self) -> None:
        obj = SyntheticObject(1, x=0, y=0, width=20, height=20, vx=7, vy=5)
        scene = SyntheticScene(SceneSpec(width=100, height=80, objects=[obj]))
        for i in range(0, 200, 7):
            box = scene.object_box(obj, i)
            assert box.x1 >= 0 and box.x2 <= 100 and box.y1 >= 0 and box.y2 <= 80

    def test_lifetime_and_exit_without_bounce(self) -> None:
        objs = [
            SyntheticObject(1, x=10, y=10, width=10, height=10, start_frame=5, end_frame=8),
            SyntheticObject(2, x=90, y=10, width=10, height=10, vx=1, bounce=False, shape="circle"),
        ]
        scene = SyntheticScene(SceneSpec(width=100, height=50, objects=objs))
        assert {g.object_id for g in scene.ground_truth(0)} == {2}
        assert {g.object_id for g in scene.ground_truth(6)} == {1, 2}
        assert {g.object_id for g in scene.ground_truth(9)} == {2}  # object 1 ended
        assert scene.ground_truth(11) == []  # object 2 left the frame
        assert scene.ground_truth(1)[0].bbox.x2 == 100  # clipped to the frame

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 4},
            {"fps": 0},
            {"num_frames": 0},
            {"noise_std": -1},
            {"objects": [SyntheticObject(1, 0, 0, 5, 5), SyntheticObject(1, 0, 0, 5, 5)]},
        ],
    )
    def test_spec_validation(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(InvalidInputError):
            SceneSpec(**kwargs)  # type: ignore[arg-type]

    def test_object_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            SyntheticObject(1, 0, 0, 0, 5)
        with pytest.raises(InvalidInputError):
            SyntheticObject(1, 0, 0, 5, 5, start_frame=4, end_frame=2)
        with pytest.raises(InvalidInputError):
            random_scene(num_objects=100)
        with pytest.raises(InvalidInputError):
            SyntheticScene(SceneSpec(num_frames=None)).frames()


class TestSyntheticSource:
    def test_reads_until_exhausted(self) -> None:
        src = SyntheticSource("s", two_object_scene(num_frames=5))
        with src:
            frames = list(src)
            assert [f.index for f in frames] == [0, 1, 2, 3, 4]
            assert frames[1].timestamp == pytest.approx(0.1)
            assert src.read() is None
            assert src.health().state == SourceState.EXHAUSTED
            assert src.health().frames_read == 5
        assert src.health().state == SourceState.CLOSED

    def test_loop_keeps_time_monotonic(self) -> None:
        src = SyntheticSource("s", two_object_scene(num_frames=3), loop=True)
        src.open()
        frames = [src.read() for _ in range(7)]
        assert [f.index for f in frames if f] == list(range(7))
        assert [f.metadata["scene_index"] for f in frames if f] == [0, 1, 2, 0, 1, 2, 0]
        assert src.info().frame_count is None

    def test_read_requires_open(self) -> None:
        with pytest.raises(SourceError):
            SyntheticSource("s", two_object_scene()).read()


class TestImageSequence:
    def test_natural_order_and_info(self, tmp_path: Path) -> None:
        for i in (10, 2, 1):
            save_image(tmp_path / f"f_{i}.png", np.full((8, 10, 3), i, np.uint8))
        (tmp_path / "notes.txt").write_text("ignored")
        src = ImageSequenceSource("seq", tmp_path, fps=5)
        with src:
            frames = list(src)
        assert [int(f.image[0, 0, 0]) for f in frames] == [1, 2, 10]
        assert frames[2].timestamp == pytest.approx(0.4)
        assert src.info().width == 10
        assert natural_key("a10") > natural_key("a9")

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError):
            ImageSequenceSource("s", tmp_path / "missing").open()
        with pytest.raises(SourceError):
            ImageSequenceSource("s", tmp_path).open()
        with pytest.raises(InvalidInputError):
            ImageSequenceSource("s", tmp_path, pattern="../*")
        (tmp_path / "bad.png").write_bytes(b"garbage")
        src = ImageSequenceSource("s", tmp_path)
        src.open()
        with pytest.raises(SourceError):
            src.read()


class TestVideoFileSource:
    def test_fake_capture_converts_bgr_and_loops(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.avi"
        video.write_bytes(b"x")
        factory = ScriptedFactory([], default=lambda: FakeCapture(frames=3, fps=20))
        src = VideoFileSource("v", video, loop=True, capture_factory=factory)
        src.open()
        frames = [src.read() for _ in range(5)]
        assert all(f is not None for f in frames)
        first = frames[0]
        assert first is not None
        assert first.image[0, 0].tolist() == [7, 0, 0]  # BGR (0, 0, 7) -> RGB (7, 0, 0)
        assert frames[4] is not None and frames[4].timestamp == pytest.approx(4 / 20)
        assert len(factory.created) == 2
        src.close()

    def test_missing_and_unopenable(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError):
            VideoFileSource("v", tmp_path / "nope.avi").open()
        video = tmp_path / "clip.avi"
        video.write_bytes(b"x")
        src = VideoFileSource("v", video, capture_factory=lambda _: FakeCapture(opened=False))
        with pytest.raises(SourceError):
            src.open()
        assert src.health().state == SourceState.FAILED

    @requires_opencv
    @pytest.mark.opencv
    def test_real_opencv_roundtrip(self, tmp_path: Path) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=12))
        path = tmp_path / "scene.avi"
        assert write_video(path, (f.image for f in scene.frames()), fps=10) == 12
        src = VideoFileSource("v", path)
        with src:
            frames = list(src)
        assert len(frames) == 12
        assert src.info().fps == pytest.approx(10)
        # MJPEG is lossy; the red object should still be clearly red.
        gt = scene.ground_truth(3)[0]
        cx, cy = (round(v) for v in gt.bbox.center)
        r, g, b = frames[3].image[cy, cx].astype(int)
        assert r > 150 and g < 120 and b < 120


class TestSpecs:
    def test_resolve_media_path_blocks_traversal(self, tmp_path: Path) -> None:
        root = tmp_path / "media"
        (root / "a").mkdir(parents=True)
        assert resolve_media_path("a", [root]) == (root / "a").resolve()
        with pytest.raises(PermissionDeniedError):
            resolve_media_path("../secret", [root])
        with pytest.raises(PermissionDeniedError):
            resolve_media_path(str(tmp_path), [root])
        with pytest.raises(PermissionDeniedError):
            resolve_media_path("a", [])

    def test_build_each_kind(self, tmp_path: Path) -> None:
        assert isinstance(build_source("s", SyntheticSourceSpec(num_objects=2)), SyntheticSource)
        (tmp_path / "v.avi").write_bytes(b"x")
        src = build_source("f", FileSourceSpec(path="v.avi"), media_roots=[tmp_path])
        assert isinstance(src, VideoFileSource)
        src2 = build_source("i", ImageSequenceSourceSpec(directory="."), media_roots=[tmp_path])
        assert isinstance(src2, ImageSequenceSource)
        rtsp = build_source("r", RtspSourceSpec(url="rtsp://cam.local/stream"))
        assert isinstance(rtsp, RtspSource)

    def test_explicit_objects_and_validation(self) -> None:
        spec = SyntheticSourceSpec.model_validate(
            {"objects": [{"object_id": 1, "x": 0, "y": 0, "width": 10, "height": 10, "vx": 1}]}
        )
        scene = spec.scene_spec()
        assert len(scene.objects) == 1 and scene.objects[0].vx == 1
        with pytest.raises(ValueError, match="extra"):
            SyntheticSourceSpec.model_validate({"bogus": 1})
        with pytest.raises(ValueError, match="scheme"):
            RtspSourceSpec(url="ftp://cam/stream")
        with pytest.raises(ValueError, match="reconnect_max_delay"):
            RtspSourceSpec(url="rtsp://cam/s", reconnect_initial_delay=10, reconnect_max_delay=1)

    def test_public_spec_redacts_credentials(self) -> None:
        spec = RtspSourceSpec(url="rtsp://admin:hunter2@10.0.0.5:554/live")
        assert public_spec(spec)["url"] == "rtsp://admin:***@10.0.0.5:554/live"
        assert public_spec(SyntheticSourceSpec())["kind"] == "synthetic"
