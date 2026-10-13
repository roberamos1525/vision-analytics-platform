"""Motion analysis: differencing, heatmaps, block-matching flow and kinematics."""

from vap.motion.differencing import MotionAnalyzer, frame_difference, three_frame_difference
from vap.motion.flow import (
    FlowField,
    FlowSummary,
    block_matching_flow,
    farneback_flow,
    heading_degrees,
)
from vap.motion.heatmap import MotionHeatmap, apply_colormap
from vap.motion.kinematics import Kinematics, compass_direction, estimate_kinematics, path_length

__all__ = [
    "FlowField",
    "FlowSummary",
    "Kinematics",
    "MotionAnalyzer",
    "MotionHeatmap",
    "apply_colormap",
    "block_matching_flow",
    "compass_direction",
    "estimate_kinematics",
    "farneback_flow",
    "frame_difference",
    "heading_degrees",
    "path_length",
    "three_frame_difference",
]
