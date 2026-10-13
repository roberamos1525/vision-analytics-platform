"""Evaluation runs and one-shot image analysis / processing."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from vap.api.deps import Operator, RuntimeDep, Viewer, decode_b64
from vap.api.schemas import AnalyzeImageRequest, ProcessImageRequest
from vap.detection.specs import BackgroundDetectorSpec, MotionDetectorSpec, build_detector
from vap.errors import InvalidInputError
from vap.evaluation.runner import EvaluationRequest, run_evaluation
from vap.imaging.backend import get_backend
from vap.imaging.chain import OpChain, apply_chain
from vap.imaging.codec import decode_image, encode_image
from vap.imaging.frame import Frame

router = APIRouter(tags=["analysis"])


@router.post("/v1/evaluations", status_code=201)
async def create_evaluation(
    body: EvaluationRequest, runtime: RuntimeDep, _: Operator
) -> dict[str, Any]:
    evaluation_id = uuid.uuid4().hex
    repo = runtime.repos.evaluations
    repo.create(evaluation_id, body.model_dump(mode="json"))
    try:
        report = await run_in_threadpool(
            run_evaluation, body, media_roots=list(runtime.settings.media.roots)
        )
    except Exception as exc:
        repo.fail(evaluation_id, f"{type(exc).__name__}: {exc}")
        raise
    repo.complete(evaluation_id, report)
    return repo.get(evaluation_id)


@router.get("/v1/evaluations")
def list_evaluations(
    runtime: RuntimeDep, _: Viewer, limit: int = Query(50, ge=1, le=500)
) -> dict[str, Any]:
    items = runtime.repos.evaluations.list(limit)
    return {"items": items, "count": len(items)}


@router.get("/v1/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.repos.evaluations.get(evaluation_id)


@router.post("/v1/analyze/image")
def analyze_image(body: AnalyzeImageRequest, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    """Run a stateless detector (colour, template or ONNX) on one image."""
    if isinstance(body.detector, BackgroundDetectorSpec | MotionDetectorSpec):
        raise InvalidInputError(
            "background and motion detectors need a video stream, not a single image"
        )
    image = decode_image(
        decode_b64(body.image_b64), max_pixels=runtime.settings.server.max_image_pixels
    )
    detector = build_detector(
        body.detector,
        backend=runtime.settings.pipelines.image_backend,
        media_roots=list(runtime.settings.media.roots),
        model_resolver=runtime.registry.resolve,
    )
    t0 = time.perf_counter()
    detections = detector.detect(Frame(image, 0, 0.0, "api"))
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "detector": detector.describe(),
        "detections": [d.to_dict() for d in detections],
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


@router.post("/v1/imaging/process")
def process_image(body: ProcessImageRequest, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    """Apply a chain of image operations and return the encoded result."""
    image = decode_image(
        decode_b64(body.image_b64), max_pixels=runtime.settings.server.max_image_pixels
    )
    backend = get_backend(runtime.settings.pipelines.image_backend)
    result, steps = apply_chain(image, OpChain(ops=body.ops), backend)
    fmt = "png" if body.output_format == "png" else "jpeg"
    encoded = encode_image(result, fmt)
    return {
        "backend": backend.name,
        "steps": steps,
        "width": int(result.shape[1]),
        "height": int(result.shape[0]),
        "format": fmt,
        "image_b64": base64.b64encode(encoded).decode("ascii"),
    }
