# Vision Analytics Platform

Local computer-vision analytics in Python: video sources (synthetic, image
sequences, video files, RTSP cameras), classical object detection, multi-object
tracking, motion analysis, zone/line analytics with event rules, a model
registry, detection/tracking evaluation, and a FastAPI service with an MJPEG
preview — all running on one machine with **numpy + Pillow as the only required
imaging dependencies**. OpenCV (`opencv-python-headless`), SciPy and
`onnxruntime` are optional backends.

> Status: portfolio project, version 0.1.0. Everything listed under
> *Implemented functionality* is covered by the automated test-suite; see
> [Limitations](#limitations) for what is not.

## Implemented functionality

| Area | What exists |
| --- | --- |
| Image processing | `ImageBackend` interface with a pure-numpy implementation and an OpenCV implementation: resize (nearest / bilinear / exact area), RGB/BGR/GRAY/HSV conversion, Gaussian and box blur, Sobel, Canny (NMS + hysteresis), morphology (erode/dilate/open/close/gradient, rect/ellipse/cross), Otsu threshold, run-based connected components. Parity tests compare both backends. Pillow codecs with decompression-bomb guards; drawing primitives with a tiny bitmap font; declarative op chains. |
| Video sources | Synthetic generator with exact ground truth; image-sequence source (natural sort); OpenCV video-file source and writer; RTSP/HTTP adapter (FFmpeg-backed `cv2.VideoCapture`, reader thread, latest-frame mailbox, frame timeouts, exponential backoff with jitter, attempt cap, health status, credential redaction). Clear `backend_unavailable` errors when OpenCV is missing. |
| Frame extraction | Every-N, time-interval and scene-change keyframes (histogram TV distance + mean abs diff), exported with a JSONL manifest. |
| Detection | Pluggable `Detector` interface: background subtraction (running Gaussian with median warm-up and static-pixel absorption, or OpenCV MOG2), HSV colour ranges, ZNCC template matching (FFT + integral images, or `cv2.matchTemplate`), frame differencing, and a YOLO-style ONNX adapter (letterbox, YOLOv5/YOLOv8 output decoding, class-aware NMS in numpy). |
| Tracking | IoU tracker and SORT (constant-velocity Kalman + Hungarian assignment) with a tentative/confirmed/lost/deleted lifecycle. Hungarian solver implemented in numpy (SciPy optional). |
| Motion | Two- and three-frame differencing, decayed motion heatmaps, block-matching optical flow (OpenCV Farneback optional), least-squares speed/heading with optional pixel-to-metre calibration. |
| Streaming analytics | Per-camera pipelines on threads; normalised or pixel zones and directed tripwires; occupancy, dwell statistics, rolling counters, directional line counts. |
| Events | Zone intrusion, line crossing, loitering and crowding rules (+ `source_down` system events) with annotated JPEG snapshots; bounded in-process event bus; SQLite, in-memory, JSONL and HMAC-signed webhook sinks (retries, SSRF guard). |
| Model management | Versioned registry (semver), SHA-256 checksums computed while streaming, expected-checksum verification, single active version per model, integrity re-verification. |
| Evaluation | Precision/recall/F1/AP (all-point or COCO-101) at arbitrary IoU thresholds and mAP@[.5:.95]; CLEAR-MOT (MOTA, MOTP, ID switches, fragmentations, MT/PT/ML) and IDF1/IDP/IDR on synthetic ground truth or JSONL files. |
| Export | CSV (formula-injection safe) and JSONL for events, tracks and metric samples (API streaming + CLI). |
| REST API | Cameras, pipelines (start/stop/status/analytics), events, metrics (JSON + Prometheus), models, evaluations, one-shot image analysis/processing, MJPEG preview, snapshots, heatmaps, health/readiness. API-key auth with viewer/operator/admin roles, consistent error envelope, body-size limits, request ids. |
| Persistence | SQLite via the standard library (`sqlite3`, WAL mode, versioned migrations). |
| CLI | `vap serve | generate | analyze | evaluate | extract | process | models | export | config`. |

## Technology stack

Python 3.11 · numpy · Pillow · FastAPI / Starlette · uvicorn · pydantic v2 ·
httpx · SQLite (stdlib) · pytest · ruff. Optional: opencv-python-headless,
SciPy, onnxruntime.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock                       # pinned, includes optional extras
pip install --no-deps -e .
export VAP_API_KEYS="dev-admin-key-0123456789:admin"   # auth is on by default
vap serve                                              # http://127.0.0.1:8080/docs
API_KEY=dev-admin-key-0123456789 ./examples/api_walkthrough.sh
```

Minimal install without OpenCV/SciPy: `pip install -e .` (numpy, Pillow,
FastAPI, uvicorn, pydantic, httpx). See [GETTING_STARTED.md](GETTING_STARTED.md).

## Configuration

Defaults < TOML file (`--config` / `VAP_CONFIG`) < environment variables
(`VAP_<SECTION>__<FIELD>`). Secrets: `VAP_API_KEYS="key:role,..."`,
`VAP_WEBHOOK_SECRET`. All values are validated at start-up; invalid
configuration exits with code 3 and a list of problems. Reference:
[examples/configs/vap.toml](examples/configs/vap.toml) and
[.env.example](.env.example).

## API usage

```bash
H="X-API-Key: dev-admin-key-0123456789"
curl -X POST localhost:8080/v1/cameras -H "$H" -H 'Content-Type: application/json' \
  -d '{"camera_id":"demo","source":{"kind":"synthetic","num_frames":300,"num_objects":4}}'
curl -X POST localhost:8080/v1/pipelines -H "$H" -H 'Content-Type: application/json' \
  -d '{"pipeline_id":"p","camera_id":"demo","start":true,"config":{"realtime":false}}'
curl localhost:8080/v1/pipelines/p/analytics -H "$H"
curl localhost:8080/v1/events?limit=10 -H "$H"
```

Full reference: [API.md](API.md) (interactive OpenAPI docs at `/docs`).

## Running locally, testing and linting

```bash
python -m pytest -q             # unit + integration tests
ruff check . && ruff format --check .
```

Tests use synthetic numpy-generated scenes with exact ground truth and fake
capture backends for files and RTSP. OpenCV-specific tests are marked `opencv`
and skip cleanly when it is not installed. Details: [docs/testing.md](docs/testing.md).

## Deployment

A `Dockerfile` and `compose.yaml` are provided (non-root user, healthcheck,
volume for state). **They were not built or run locally because Docker is not
installed on the development machine.** Operations guidance (backups,
retention, metrics, tuning) is in [docs/operations.md](docs/operations.md).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map and data flow, and
[docs/algorithms.md](docs/algorithms.md) / [docs/data-model.md](docs/data-model.md)
for the deeper design notes.

## Limitations

- **RTSP was not tested against a live camera.** The adapter is exercised only
  with fake capture backends (connect failures, stalls, exceptions, end of
  stream, reconnect/backoff, attempt caps).
- **ONNX inference is not exercised end-to-end.** onnxruntime is not installed
  and no model file is shipped; pre/post-processing (letterbox, decoding, NMS)
  is tested with an injected fake session.
- Detection is classical. Background subtraction needs a warm-up window
  (default 40 frames), merges touching objects into one blob and eventually
  absorbs objects that stop moving (`absorb_after`, default 150 frames).
- Accuracy figures come from synthetic scenes (flat-coloured shapes on a static
  background); they do not predict real-world performance.
- Single-process design: pipelines are threads in one Python process (GIL-bound
  for pure-Python parts); there is no horizontal scaling, GPU support or
  distributed event bus.
- The MJPEG preview re-encodes the newest annotated frame on demand; it is a
  monitoring aid, not a video streaming server.
- Docker image, compose stack and the GitHub Actions workflow were written but
  not executed.

## Roadmap

- Validate the RTSP adapter against real cameras and add an FFmpeg-subprocess
  capture backend.
- End-to-end ONNX tests with a tiny exported model; optional GPU providers.
- Appearance features for re-identification (DeepSORT-style) and occlusion handling.
- Server-sent events / WebSocket push for events.
- Multi-process pipeline workers and an external message bus option.

## AI-assisted development

This project was developed with AI assistance (Claude by Anthropic). The code,
tests and documentation were generated and iterated in an AI-assisted session
and reviewed/verified by running the full test-suite, the linters and a manual
smoke test of the running service.

## License

MIT — see [LICENSE](LICENSE).
