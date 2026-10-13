"""Object detection: interface, classical detectors, ONNX adapter and NMS."""

from vap.detection.background import BackgroundSubtractionDetector, RunningGaussianModel
from vap.detection.base import Detection, Detector
from vap.detection.color import ColorDetector, ColorRange
from vap.detection.motion import FrameDifferenceDetector
from vap.detection.nms import batched_nms, nms, suppress
from vap.detection.onnx import YoloOnnxDetector, letterbox
from vap.detection.specs import DetectorSpec, build_detector
from vap.detection.template import TemplateDetector, match_template_ncc

__all__ = [
    "BackgroundSubtractionDetector",
    "ColorDetector",
    "ColorRange",
    "Detection",
    "Detector",
    "DetectorSpec",
    "FrameDifferenceDetector",
    "RunningGaussianModel",
    "TemplateDetector",
    "YoloOnnxDetector",
    "batched_nms",
    "build_detector",
    "letterbox",
    "match_template_ncc",
    "nms",
    "suppress",
]
