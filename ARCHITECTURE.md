# Architecture

## Module map (`src/vap/`)

```
errors.py           typed error hierarchy (code + HTTP status) shared by library, CLI and API
geometry.py         BBox, Polygon, LineSegment, vectorised IoU
config.py           Settings (pydantic): defaults < TOML < env; roles
runtime.py          application container: wires storage, bus, sinks, registry, manager
cli.py              `vap` command-line interface

imaging/            frame.py (Frame, validation) · ops.py (ImageBackend ABC)
                    numpy_backend.py · opencv_backend.py · backend.py (selection/cache)
                    components.py (run-based labelling) · codec.py (Pillow) · draw.py · chain.py
sources/            base.py (FrameSource template, health) · synthetic.py · image_sequence.py
                    video_file.py · rtsp.py (reconnecting adapter) · specs.py (validated specs + factory)
extraction.py       every-N / interval / scene-change frame selection and export
detection/          base.py (Detector, Detection) · background.py · color.py · template.py
                    motion.py · onnx.py · nms.py · specs.py (validated specs + factory)
tracking/           assignment.py (Hungarian/greedy/scipy) · kalman.py · base.py (lifecycle) · trackers.py
motion/             differencing.py · heatmap.py · flow.py (block matching, Farneback) · kinematics.py
analytics/          zones.py · counters.py · engine.py (membership/dwell/crossings) · rules.py
events/             model.py · bus.py (bounded queue + dispatcher) · sinks.py (sqlite/recent/jsonl/webhook)
pipeline/           config.py · pipeline.py (per-camera thread) · annotate.py · manager.py
registry/           models.py (versioned artefacts, checksums, activation)
evaluation/         detection.py (P/R/AP) · tracking.py (CLEAR-MOT, IDF1) · runner.py
exporting.py        CSV/JSONL writers
storage/            database.py (sqlite3, migrations) · repositories.py
observability/      logs.py (JSON logs) · metrics.py (Prometheus text registry)
api/                app.py (factory, error envelopes) · middleware.py · security.py · deps.py
                    schemas.py · routes/{system,cameras,pipelines,events,models,analysis}.py
```

Dependency direction is strictly downward: `api` → `runtime` → `pipeline` →
`analytics/tracking/detection/motion/sources` → `imaging/geometry` → `errors`.
No library module imports FastAPI; the API layer is a thin adapter.

## Frame data flow

```
FrameSource.read() ─► CameraPipeline._prepare (frame skip, resize)
   ─► Detector.detect ─► Tracker.update ─► AnalyticsEngine.update ─► RuleEngine.evaluate
   ─► Events (+ annotated snapshot JPEG) ─► EventBus ─► sinks: SQLite · recent ring · webhook
   ├► heatmap / block flow / kinematics (live analytics)
   ├► finished tracks ─► SQLite `tracks`
   ├► periodic metric samples ─► SQLite `metric_samples` + Prometheus registry
   └► latest frame + tracks ─► MJPEG preview / snapshot endpoints (annotated on demand)
```

* Internally all colour images are `HxWx3 uint8` RGB. OpenCV sources convert
  BGR at the boundary (`bgr_to_frame_image`).
* Frame timestamps are *stream time* (index/fps for files and synthetic
  sources, wall clock for RTSP), so analytics are deterministic in tests.

## Concurrency model

* One worker thread per running pipeline (`pipeline-<id>`). RTSP sources add a
  reader thread per connection that keeps only the newest frame, so a slow
  pipeline never accumulates latency.
* The event bus has a bounded queue and a single dispatcher thread;
  publishers never block (events are dropped and counted when full). The
  webhook sink has its own queue/worker so slow receivers cannot stall other
  sinks.
* SQLite uses one shared connection serialised with an `RLock` in WAL mode.
* FastAPI sync endpoints run in the threadpool; MJPEG streams wait on a
  condition variable signalled by the pipeline.

## Error handling

Every error derives from `VapError` with a stable `code` and HTTP status. The
API maps them, plus request-validation and unexpected errors, onto one
envelope:

```json
{"error": {"code": "conflict", "message": "...", "details": {}}, "request_id": "..."}
```

Pipelines contain failures: a crashing detector or an exhausted RTSP source
moves the pipeline to `failed` (with a `source_down` event for the latter)
without affecting other pipelines.

## Backends and optional dependencies

| Feature | Without extras | With extras |
| --- | --- | --- |
| Image ops | numpy backend | `auto` picks OpenCV |
| Video files / RTSP | `backend_unavailable` (501) | OpenCV/FFmpeg capture |
| Background engine | running Gaussian | + OpenCV MOG2 |
| Template matching | FFT ZNCC | `cv2.matchTemplate` |
| Dense flow | block matching | + Farneback |
| Assignment | numpy Hungarian | + `solver="scipy"` |
| ONNX detector | pre/post-processing only | onnxruntime session |

Design notes for individual algorithms are in [docs/algorithms.md](docs/algorithms.md).
