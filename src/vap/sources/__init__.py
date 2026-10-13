"""Video sources: synthetic generator, image sequences, video files and RTSP cameras."""

from vap.sources.base import FrameSource, SourceHealth, SourceInfo, SourceState
from vap.sources.image_sequence import ImageSequenceSource
from vap.sources.rtsp import ReconnectPolicy, RtspSource
from vap.sources.specs import SourceSpec, build_source
from vap.sources.synthetic import (
    GroundTruth,
    SceneSpec,
    SyntheticObject,
    SyntheticScene,
    SyntheticSource,
)
from vap.sources.video_file import VideoFileSource

__all__ = [
    "FrameSource",
    "GroundTruth",
    "ImageSequenceSource",
    "ReconnectPolicy",
    "RtspSource",
    "SceneSpec",
    "SourceHealth",
    "SourceInfo",
    "SourceSpec",
    "SourceState",
    "SyntheticObject",
    "SyntheticScene",
    "SyntheticSource",
    "VideoFileSource",
    "build_source",
]
