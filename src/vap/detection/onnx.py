"""ONNX adapter for YOLO-style detectors.

Pre- and post-processing are implemented in numpy: letterbox resize with
padding, NCHW float normalisation, decoding of both common output layouts and
class-aware NMS. ``onnxruntime`` is imported lazily and only needed when no
session object is injected.

Supported output layouts (batch size 1):

* ``yolov5``: ``(1, N, 5 + C)`` rows of ``cx, cy, w, h, objectness, class scores``
* ``yolov8``: ``(1, 4 + C, N)`` columns of ``cx, cy, w, h, class scores``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from vap.detection.base import Detection, Detector
from vap.detection.nms import batched_nms
from vap.errors import BackendUnavailableError, InvalidInputError, ModelError
from vap.geometry import BBox
from vap.imaging.backend import get_backend
from vap.imaging.frame import Frame, ensure_image

OutputFormat = Literal["auto", "yolov5", "yolov8"]


class InferenceSession(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[Any]: ...


def letterbox(
    image: np.ndarray, size: tuple[int, int], pad_value: int = 114, backend: str = "auto"
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to ``size=(width, height)``.

    Returns the padded image, the scale ratio and the ``(pad_x, pad_y)`` offset
    of the resized content inside the padded canvas.
    """
    image = ensure_image(image, allow_gray=False)
    tw, th = size
    if tw < 1 or th < 1:
        raise InvalidInputError("letterbox size must be positive")
    h, w = image.shape[:2]
    ratio = min(tw / w, th / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = (
        get_backend(backend).resize(image, (nw, nh), "bilinear") if (nw, nh) != (w, h) else image
    )
    pad_x, pad_y = (tw - nw) / 2.0, (th - nh) / 2.0
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, ratio, (float(left), float(top))


def to_nchw(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def decode_predictions(
    output: np.ndarray, conf_threshold: float, layout: OutputFormat = "auto"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLO output into ``(boxes_xyxy, scores, class_ids)`` above threshold."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 3:
        if out.shape[0] != 1:
            raise ModelError("only batch size 1 is supported")
        out = out[0]
    if out.ndim != 2:
        raise ModelError("unexpected model output rank", details={"shape": list(output.shape)})
    if layout == "auto":
        layout = "yolov8" if out.shape[0] < out.shape[1] else "yolov5"
    if layout == "yolov8":
        rows = out.T
        if rows.shape[1] < 5:
            raise ModelError("yolov8 output needs at least 5 channels")
        cls_scores = rows[:, 4:]
    else:
        rows = out
        if rows.shape[1] < 6:
            raise ModelError("yolov5 output needs at least 6 channels")
        cls_scores = rows[:, 5:] * rows[:, 4:5]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= conf_threshold
    cx, cy, bw, bh = rows[keep, 0], rows[keep, 1], rows[keep, 2], rows[keep, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return (
        boxes.astype(np.float64),
        scores[keep].astype(np.float64),
        class_ids[keep].astype(np.int64),
    )


def unletterbox(
    boxes: np.ndarray, ratio: float, pad: tuple[float, float], orig_size: tuple[int, int]
) -> np.ndarray:
    """Map boxes from letterboxed coordinates back to the original image."""
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / ratio
    w, h = orig_size
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def load_onnx_session(path: Path) -> InferenceSession:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BackendUnavailableError(
            "onnxruntime is not installed; install the 'onnx' extra", details={"backend": "onnx"}
        ) from exc
    if not path.is_file():
        raise ModelError(f"model file not found: {path.name}")
    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own exception hierarchy
        raise ModelError(f"failed to load ONNX model: {exc}") from exc


class YoloOnnxDetector(Detector):
    name = "onnx"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        session: InferenceSession | None = None,
        input_size: tuple[int, int] = (640, 640),
        class_names: list[str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        output_format: OutputFormat = "auto",
        backend: str = "auto",
    ) -> None:
        if session is None and model_path is None:
            raise InvalidInputError("either model_path or session is required")
        if not 0 <= conf_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise InvalidInputError("thresholds must be within [0, 1]")
        if output_format not in ("auto", "yolov5", "yolov8"):
            raise InvalidInputError(f"unknown output format {output_format!r}")
        if max_detections < 1:
            raise InvalidInputError("max_detections must be >= 1")
        self.session = session if session is not None else load_onnx_session(Path(str(model_path)))
        inputs = self.session.get_inputs()
        if not inputs:
            raise ModelError("model declares no inputs")
        self.input_name = str(inputs[0].name)
        self.input_size = input_size
        self.class_names = list(class_names or [])
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.output_format: OutputFormat = output_format
        self.backend = backend

    def _label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def detect(self, frame: Frame) -> list[Detection]:
        canvas, ratio, pad = letterbox(frame.image, self.input_size, backend=self.backend)
        try:
            outputs = self.session.run(None, {self.input_name: to_nchw(canvas)})
        except Exception as exc:  # runtime errors from the inference engine
            raise ModelError(f"inference failed: {exc}") from exc
        if not outputs:
            raise ModelError("model returned no outputs")
        boxes, scores, class_ids = decode_predictions(
            np.asarray(outputs[0]), self.conf_threshold, self.output_format
        )
        if boxes.shape[0] == 0:
            return []
        keep = batched_nms(boxes, scores, class_ids, self.iou_threshold)[: self.max_detections]
        mapped = unletterbox(boxes[keep], ratio, pad, (frame.width, frame.height))
        dets: list[Detection] = []
        for box, idx in zip(mapped, keep, strict=True):
            if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                continue
            cid = int(class_ids[idx])
            dets.append(Detection(BBox.from_array(box), float(scores[idx]), self._label(cid), cid))
        return dets

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_size": list(self.input_size),
            "classes": len(self.class_names),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "output_format": self.output_format,
        }
