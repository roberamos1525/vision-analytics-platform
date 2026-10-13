"""Validated source specifications and the factory that turns them into sources."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vap.errors import InvalidInputError, PermissionDeniedError
from vap.sources.base import FrameSource
from vap.sources.image_sequence import ImageSequenceSource
from vap.sources.rtsp import ReconnectPolicy, RtspSource, redact_url, validate_stream_url
from vap.sources.synthetic import SceneSpec, SyntheticObject, SyntheticSource, random_scene
from vap.sources.video_file import CaptureFactory, VideoFileSource

RGB = tuple[
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
    Annotated[int, Field(ge=0, le=255)],
]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticObjectSpec(_Spec):
    object_id: int = Field(ge=1)
    x: float
    y: float
    width: float = Field(gt=0, le=4096)
    height: float = Field(gt=0, le=4096)
    vx: float = Field(0.0, ge=-200, le=200)
    vy: float = Field(0.0, ge=-200, le=200)
    shape: Literal["rect", "circle"] = "rect"
    color: RGB = (255, 255, 255)
    label: str = Field("object", min_length=1, max_length=64)
    bounce: bool = True
    start_frame: int = Field(0, ge=0)
    end_frame: int | None = Field(None, ge=0)


class SyntheticSourceSpec(_Spec):
    kind: Literal["synthetic"] = "synthetic"
    width: int = Field(320, ge=8, le=4096)
    height: int = Field(240, ge=8, le=4096)
    fps: float = Field(15.0, gt=0, le=240)
    num_frames: int | None = Field(300, ge=1, le=10_000_000)
    loop: bool = False
    seed: int = Field(0, ge=0)
    num_objects: int = Field(3, ge=0, le=64)
    noise_std: float = Field(2.0, ge=0, le=64)
    max_speed: float = Field(4.0, gt=0, le=100)
    background: RGB = (32, 32, 32)
    objects: list[SyntheticObjectSpec] | None = Field(None, max_length=64)

    def scene_spec(self) -> SceneSpec:
        if self.objects is None:
            spec = random_scene(
                seed=self.seed,
                num_objects=self.num_objects,
                width=self.width,
                height=self.height,
                fps=self.fps,
                num_frames=self.num_frames,
                noise_std=self.noise_std,
                max_speed=self.max_speed,
                max_size=min(36, min(self.width, self.height) - 1),
                min_size=min(14, min(self.width, self.height) - 2),
            )
            spec.background = self.background
            return spec
        return SceneSpec(
            width=self.width,
            height=self.height,
            fps=self.fps,
            num_frames=self.num_frames,
            background=self.background,
            noise_std=self.noise_std,
            seed=self.seed,
            objects=[SyntheticObject(**o.model_dump()) for o in self.objects],
        )


class FileSourceSpec(_Spec):
    kind: Literal["file"] = "file"
    path: str = Field(min_length=1, max_length=1024)
    loop: bool = False


class ImageSequenceSourceSpec(_Spec):
    kind: Literal["images"] = "images"
    directory: str = Field(min_length=1, max_length=1024)
    pattern: str = Field("*", min_length=1, max_length=64)
    fps: float = Field(10.0, gt=0, le=240)
    loop: bool = False


class RtspSourceSpec(_Spec):
    kind: Literal["rtsp"] = "rtsp"
    url: str = Field(min_length=1, max_length=2048)
    frame_timeout: float = Field(5.0, gt=0, le=120)
    open_timeout: float = Field(10.0, gt=0, le=120)
    reconnect_initial_delay: float = Field(0.5, ge=0, le=60)
    reconnect_max_delay: float = Field(30.0, ge=0, le=3600)
    reconnect_multiplier: float = Field(2.0, ge=1, le=10)
    reconnect_jitter: float = Field(0.1, ge=0, le=1)
    max_reconnect_attempts: int | None = Field(None, ge=1)

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        try:
            return validate_stream_url(value)
        except InvalidInputError as exc:
            raise ValueError(exc.message) from exc

    @model_validator(mode="after")
    def _check_delays(self) -> RtspSourceSpec:
        if self.reconnect_max_delay < self.reconnect_initial_delay:
            raise ValueError("reconnect_max_delay must be >= reconnect_initial_delay")
        return self

    def policy(self) -> ReconnectPolicy:
        return ReconnectPolicy(
            initial_delay=self.reconnect_initial_delay,
            max_delay=self.reconnect_max_delay,
            multiplier=self.reconnect_multiplier,
            jitter=self.reconnect_jitter,
            max_attempts=self.max_reconnect_attempts,
        )


SourceSpec = Annotated[
    SyntheticSourceSpec | FileSourceSpec | ImageSequenceSourceSpec | RtspSourceSpec,
    Field(discriminator="kind"),
]


def public_spec(spec: BaseModel) -> dict[str, object]:
    """Serialise a spec for API responses with credentials redacted."""
    data = spec.model_dump(mode="json")
    if isinstance(spec, RtspSourceSpec):
        data["url"] = redact_url(spec.url)
    return data


def resolve_media_path(raw: str, media_roots: list[Path]) -> Path:
    """Resolve a user-supplied path, which must live under one of ``media_roots``."""
    if not media_roots:
        raise PermissionDeniedError("file-based sources are disabled (no media roots configured)")
    candidate = Path(raw)
    for root in media_roots:
        root = root.resolve()
        full = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if full == root or root in full.parents:
            return full
    raise PermissionDeniedError("path is outside the configured media roots", details={"path": raw})


def build_source(
    source_id: str,
    spec: SyntheticSourceSpec | FileSourceSpec | ImageSequenceSourceSpec | RtspSourceSpec,
    *,
    media_roots: list[Path] | None = None,
    capture_factory: CaptureFactory | None = None,
) -> FrameSource:
    """Instantiate a (not yet opened) source for ``spec``."""
    roots = media_roots or []
    if isinstance(spec, SyntheticSourceSpec):
        return SyntheticSource(source_id, spec.scene_spec(), loop=spec.loop)
    if isinstance(spec, FileSourceSpec):
        path = resolve_media_path(spec.path, roots)
        return VideoFileSource(source_id, path, loop=spec.loop, capture_factory=capture_factory)
    if isinstance(spec, ImageSequenceSourceSpec):
        directory = resolve_media_path(spec.directory, roots)
        return ImageSequenceSource(
            source_id, directory, pattern=spec.pattern, fps=spec.fps, loop=spec.loop
        )
    if isinstance(spec, RtspSourceSpec):
        return RtspSource(
            source_id,
            spec.url,
            capture_factory=capture_factory,
            policy=spec.policy(),
            frame_timeout=spec.frame_timeout,
            open_timeout=spec.open_timeout,
        )
    raise InvalidInputError(f"unsupported source spec {type(spec).__name__}")
