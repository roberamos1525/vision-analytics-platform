"""Per-camera streaming pipelines and their manager."""

from vap.pipeline.config import PipelineConfig
from vap.pipeline.manager import PipelineManager, parse_pipeline_config, parse_source_spec
from vap.pipeline.pipeline import CameraPipeline, PipelineOptions, PipelineState

__all__ = [
    "CameraPipeline",
    "PipelineConfig",
    "PipelineManager",
    "PipelineOptions",
    "PipelineState",
    "parse_pipeline_config",
    "parse_source_spec",
]
