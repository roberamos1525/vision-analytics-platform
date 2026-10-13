"""Model registry, evaluation and one-shot analysis endpoints."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vap.config import Settings
from vap.imaging.codec import decode_image, encode_image, save_image


@pytest.fixture
def client(client_factory: Callable[..., TestClient], settings: Settings) -> TestClient:
    return client_factory(settings)


def test_model_registry_lifecycle(client: TestClient) -> None:
    blob = b"fake-onnx-weights" * 100
    digest = hashlib.sha256(blob).hexdigest()
    meta = json.dumps({"class_names": ["person", "car"], "input_size": [640, 640]})
    r = client.put(
        "/v1/models/yolo-lite/versions/1.0.0",
        content=blob,
        params={"sha256": digest, "activate": True},
        headers={"X-Model-Metadata": meta},
    )
    assert r.status_code == 201, r.text
    assert r.json()["active"] and r.json()["metadata"]["class_names"] == ["person", "car"]
    client.put("/v1/models/yolo-lite/versions/1.1.0", content=b"v2")
    listing = client.get("/v1/models").json()["items"]
    assert listing == [{"name": "yolo-lite", "versions": 2, "active": "1.0.0", "latest": "1.1.0"}]
    assert client.post("/v1/models/yolo-lite/activate", json={"version": "1.1.0"}).json()["active"]
    assert client.get("/v1/models/yolo-lite").json()["active"] == "1.1.0"
    assert client.post("/v1/models/yolo-lite/versions/1.0.0/verify").json()["valid"]
    assert client.delete("/v1/models/yolo-lite/versions/1.1.0").status_code == 409
    assert (
        client.delete("/v1/models/yolo-lite/versions/1.1.0", params={"force": True}).status_code
        == 204
    )
    assert client.post("/v1/models/yolo-lite/deactivate").status_code == 204
    assert client.get("/v1/models/yolo-lite/versions/1.1.0").status_code == 404
    assert client.get("/v1/models/unknown").status_code == 404


def test_model_upload_errors(client: TestClient) -> None:
    bad_sum = client.put("/v1/models/m/versions/1.0.0", content=b"abc", params={"sha256": "0" * 64})
    assert bad_sum.status_code == 422 and bad_sum.json()["error"]["code"] == "checksum_mismatch"
    assert client.put("/v1/models/m/versions/1.0", content=b"abc").status_code == 422
    assert (
        client.put(
            "/v1/models/m/versions/1.0.0", content=b"abc", headers={"X-Model-Metadata": "[1]"}
        ).status_code
        == 422
    )
    client.put("/v1/models/m/versions/1.0.0", content=b"abc")
    assert client.put("/v1/models/m/versions/1.0.0", content=b"abc").status_code == 409


def test_onnx_pipeline_reports_missing_runtime(client: TestClient) -> None:
    client.put("/v1/models/det/versions/1.0.0", content=b"weights", params={"activate": True})
    client.post(
        "/v1/cameras", json={"camera_id": "c", "source": {"kind": "synthetic", "num_frames": 5}}
    )
    client.post(
        "/v1/pipelines",
        json={
            "pipeline_id": "p",
            "camera_id": "c",
            "config": {"detector": {"kind": "onnx", "model": "det"}},
        },
    )
    r = client.post("/v1/pipelines/p/start")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        assert r.status_code == 501 and r.json()["error"]["code"] == "backend_unavailable"
    else:  # pragma: no cover - the fake weights are not a valid ONNX graph
        assert r.status_code == 422


def test_evaluation_endpoints(client: TestClient) -> None:
    body = {"scene": {"num_frames": 70, "seed": 3, "num_objects": 2}, "iou_thresholds": [0.5]}
    r = client.post("/v1/evaluations", json=body)
    assert r.status_code == 201
    result = r.json()
    assert result["status"] == "completed"
    assert 0 <= result["report"]["tracking"]["mota"] <= 1
    assert (
        client.get(f"/v1/evaluations/{result['evaluation_id']}").json()["report"]
        == result["report"]
    )
    assert client.get("/v1/evaluations").json()["count"] == 1
    bad = client.post("/v1/evaluations", json={"scene": {"loop": True}})
    assert bad.status_code == 422
    assert client.get("/v1/evaluations/nope").status_code == 404


def b64_png(img: np.ndarray) -> str:
    return base64.b64encode(encode_image(img, "png")).decode()


def test_analyze_image_with_color_and_template(client: TestClient, settings: Settings) -> None:
    img = np.full((60, 80, 3), 20, np.uint8)
    img[10:30, 15:35] = (230, 30, 30)
    detector = {"kind": "color", "ranges": [{"label": "red", "hue_min": 170, "hue_max": 10}]}
    r = client.post("/v1/analyze/image", json={"image_b64": b64_png(img), "detector": detector})
    assert r.status_code == 200
    dets = r.json()["detections"]
    assert len(dets) == 1 and dets[0]["label"] == "red" and dets[0]["bbox"] == [15, 10, 35, 30]
    media = settings.media.roots[0]
    save_image(media / "tpl.png", img[8:32, 13:37])
    tpl = {"kind": "template", "template_path": "tpl.png", "threshold": 0.9}
    r = client.post("/v1/analyze/image", json={"image_b64": b64_png(img), "detector": tpl})
    assert r.json()["detections"][0]["bbox"][:2] == [13, 8]
    bg = client.post(
        "/v1/analyze/image", json={"image_b64": b64_png(img), "detector": {"kind": "background"}}
    )
    assert bg.status_code == 422
    garbage = client.post("/v1/analyze/image", json={"image_b64": "@@@@", "detector": detector})
    assert garbage.status_code == 422


def test_imaging_process_endpoint(client: TestClient) -> None:
    img = np.zeros((40, 40, 3), np.uint8)
    img[10:30, 10:30] = 255
    ops = [
        {"op": "gray"},
        {"op": "canny", "low": 50, "high": 150},
        {"op": "morphology", "operation": "dilate"},
    ]
    r = client.post("/v1/imaging/process", json={"image_b64": b64_png(img), "ops": ops})
    assert r.status_code == 200
    data = r.json()
    out = decode_image(base64.b64decode(data["image_b64"]))
    assert data["width"] == 40 and len(data["steps"]) == 3
    assert out[10, 20].max() == 255 and out[20, 20].max() == 0
    assert (
        client.post(
            "/v1/imaging/process", json={"image_b64": b64_png(img), "ops": [{"op": "warp"}]}
        ).status_code
        == 422
    )


def test_frame_extraction_endpoint(client: TestClient, settings: Settings) -> None:
    client.post(
        "/v1/cameras", json={"camera_id": "c", "source": {"kind": "synthetic", "num_frames": 30}}
    )
    r = client.post(
        "/v1/cameras/c/extract", json={"job": "job1", "spec": {"mode": "every_n", "every_n": 10}}
    )
    assert r.status_code == 201
    data = r.json()
    assert data["count"] == 3 and data["directory"] == "extractions/c/job1"
    out = Path(settings.storage.exports_dir) / "extractions" / "c" / "job1"
    assert len(list(out.glob("*.jpg"))) == 3
    again = client.post("/v1/cameras/c/extract", json={"job": "job1"})
    assert again.status_code == 422
