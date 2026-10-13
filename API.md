# REST API reference

Base URL: `http://<host>:<port>`. OpenAPI schema: `/openapi.json`, Swagger UI: `/docs`.

## Conventions

**Authentication.** Send `X-API-Key: <key>` or `Authorization: Bearer <key>`.
Keys are configured with `VAP_API_KEYS="key:role,..."`. Roles are hierarchical:
`viewer` (read) < `operator` (cameras, pipelines, evaluations, analysis) <
`admin` (model registry writes). `/healthz` and `/readyz` are unauthenticated.

**Errors.** Every error uses the same envelope and echoes the request id:

```json
{"error": {"code": "invalid_input", "message": "request validation failed",
           "details": {"errors": [{"loc": "body.camera_id", "msg": "...", "type": "string_pattern_mismatch"}]}},
 "request_id": "0434ad7a5ae44137ba1eca18bf1c40c3"}
```

| HTTP | code | meaning |
| --- | --- | --- |
| 401 | `unauthenticated` | missing/invalid key (`WWW-Authenticate: ApiKey`) |
| 403 | `forbidden` | role too low, or path outside media roots |
| 404 | `not_found` | unknown resource or route |
| 409 | `conflict` | duplicate id, resource in use, pipeline not running |
| 413 | `payload_too_large` | body over `max_request_bytes` / `max_upload_bytes`, image over `max_image_pixels` |
| 422 | `invalid_input` / `checksum_mismatch` / `model_error` | validation failures |
| 429 | `limit_exceeded` | `max_pipelines` reached |
| 501 | `backend_unavailable` | optional dependency (OpenCV, onnxruntime) missing |
| 502/504 | `source_error` / `source_timeout` | camera could not be opened / probe timed out |

**Request ids.** A valid incoming `X-Request-ID` (`[A-Za-z0-9._-]{1,64}`) is
propagated; otherwise one is generated. It is returned in the `X-Request-ID`
response header and logged.

**Ids.** Camera/pipeline/zone/line/rule ids match `^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$`.

## System

| Method | Path | Role | Description |
| --- | --- | --- | --- |
| GET | `/healthz` | – | liveness |
| GET | `/readyz` | – | 200 when DB and event bus are healthy, else 503 |
| GET | `/v1/health` | viewer | detailed health: DB schema, bus stats, webhooks, backends |
| GET | `/v1/system` | viewer | version, backends, limits |
| GET | `/metrics` | viewer | Prometheus text exposition (`vap_*` metrics) |

## Cameras

| Method | Path | Role |
| --- | --- | --- |
| POST | `/v1/cameras` | operator |
| GET | `/v1/cameras` | viewer |
| GET / PUT / DELETE | `/v1/cameras/{camera_id}` | viewer / operator / operator |
| POST | `/v1/cameras/{camera_id}/probe?timeout_s=10` | operator |
| POST | `/v1/cameras/{camera_id}/extract` | operator |

Create body: `{"camera_id": "gate", "name": "Gate", "source": <source spec>}`. Source specs
(discriminated by `kind`):

```jsonc
{"kind": "synthetic", "width": 320, "height": 240, "fps": 15, "num_frames": 300, "loop": false,
 "seed": 0, "num_objects": 3, "noise_std": 2.0, "max_speed": 4.0,
 "objects": [{"object_id": 1, "x": 10, "y": 40, "width": 30, "height": 24, "vx": 2.5, "vy": 0,
              "shape": "rect", "color": [230, 60, 60], "label": "object"}]}   // objects optional
{"kind": "file", "path": "clip.mp4", "loop": false}                  // relative to a media root
{"kind": "images", "directory": "frames", "pattern": "*.png", "fps": 10}
{"kind": "rtsp", "url": "rtsp://user:pass@host/stream", "frame_timeout": 5, "open_timeout": 10,
 "reconnect_initial_delay": 0.5, "reconnect_max_delay": 30, "reconnect_multiplier": 2,
 "reconnect_jitter": 0.1, "max_reconnect_attempts": null}
```

`PUT` and `DELETE` return 409 while a running pipeline (or any pipeline, for
delete) uses the camera. Probe opens the source, reads one frame and returns
`{ok, latency_ms, frame: {width, height}, info, health}`.

Extraction body: `{"job": "job1", "spec": {"mode": "every_n" | "interval" | "scene_change",
"every_n": 10, "interval_s": 1.0, "scene_threshold": 0.3, "min_gap_frames": 5,
"max_frames": 100, "max_scan_frames": 10000, "image_format": "jpeg", "resize_width": null}}`.
Frames are written under `<state_dir>/exports/extractions/<camera>/<job>/` with a
`manifest.jsonl`; the response lists the selected frames.

## Pipelines

| Method | Path | Role | Description |
| --- | --- | --- | --- |
| POST | `/v1/pipelines` | operator | `{pipeline_id, camera_id, config, start}` |
| GET | `/v1/pipelines` | viewer | list with live status |
| GET / PUT / DELETE | `/v1/pipelines/{id}` | viewer / operator / operator | PUT/DELETE require stopped |
| POST | `/v1/pipelines/{id}/start` · `/stop` | operator | 409 if already running / not running |
| GET | `/v1/pipelines/{id}/status` | viewer | state, frames, fps, source health, detector |
| GET | `/v1/pipelines/{id}/analytics` | viewer | zones (occupancy, dwell), lines, tracks with kinematics, flow, heatmap hotspots |
| GET | `/v1/pipelines/{id}/metrics` | viewer | current counters |
| GET | `/v1/pipelines/{id}/metrics/history?since&limit` | viewer | stored samples |
| GET | `/v1/pipelines/{id}/runs` | viewer | run history (status, frames, error) |
| GET | `/v1/pipelines/{id}/tracks?run_id&limit` | viewer | finished track summaries |
| GET | `/v1/pipelines/{id}/snapshot.jpg` | viewer | latest annotated frame (`X-Frame-Index`) |
| GET | `/v1/pipelines/{id}/heatmap.png` | viewer | motion heatmap |
| GET | `/v1/pipelines/{id}/preview.mjpg?fps=5&max_frames` | viewer | `multipart/x-mixed-replace` stream |

States: `starting → running → stopping → stopped`, or `completed` (finite
source ended) / `failed` (error recorded in `status.error`).

Pipeline config (all fields optional):

```jsonc
{
  "detector": {"kind": "background", "engine": "gaussian", "learning_rate": 0.02, "k_sigma": 2.5,
                "min_diff": 18, "blur_sigma": 1.0, "morph_ksize": 3, "min_area": 30, "max_area_ratio": 0.5,
                "warmup_frames": 20, "warmup_stride": 2, "absorb_after": 150, "label": "object"},
  //  or {"kind": "color", "ranges": [{"label": "red", "hue_min": 170, "hue_max": 10, "sat_min": 80, ...}]}
  //  or {"kind": "template", "template_path": "logo.png" | "template_png_b64": "...", "threshold": 0.8, "scales": [1.0]}
  //  or {"kind": "motion", "threshold": 20, "mode": "three_frame"}
  //  or {"kind": "onnx", "model": "yolo", "version": null, "input_width": 640, "conf_threshold": 0.25, ...}
  "tracker": {"kind": "sort", "iou_threshold": 0.3, "min_hits": 3, "max_age": 10, "solver": "hungarian", "class_aware": false},
  "zones": [{"zone_id": "z", "points": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "normalized": true}],
  "lines": [{"line_id": "door", "points": [[160, 0], [160, 240]]}],
  "rules": [
    {"kind": "zone_intrusion", "rule_id": "r1", "zone_id": "z", "labels": [], "cooldown_s": 0, "severity": "warning"},
    {"kind": "line_crossing", "rule_id": "r2", "line_id": "door", "direction": "any"},
    {"kind": "loitering", "rule_id": "r3", "zone_id": "z", "min_dwell_s": 10},
    {"kind": "crowding", "rule_id": "r4", "zone_id": null, "min_count": 5, "min_duration_s": 2, "cooldown_s": 30}
  ],
  "process_every_n": 1, "resize_width": null, "realtime": true, "max_frames": null,
  "snapshots": true, "heatmap": true, "heatmap_decay": 0.998, "flow_every_n": 0,
  "pixels_per_meter": null, "counter_window_s": 60, "persist_tracks": true
}
```

Zone/line coordinates refer to the *processed* frame (after `resize_width`);
use `normalized: true` to stay resolution-independent. A tripwire's positive
side is the right-hand side when walking from its first to its second point as
drawn on screen. Rules referencing unknown zones/lines are rejected.

## Events and exports

| Method | Path | Description |
| --- | --- | --- |
| GET | `/v1/events?pipeline_id&camera_id&type&since&until&before&limit` | newest first; `next_before` cursor for pagination |
| GET | `/v1/events/recent?pipeline_id&limit` | in-memory ring buffer |
| GET | `/v1/events/stats?pipeline_id` | counts by type + bus statistics |
| GET | `/v1/events/{event_id}` | one event |
| GET | `/v1/events/{event_id}/snapshot` | annotated JPEG captured when the event fired |
| GET | `/v1/exports/{events|tracks|metrics}?format=csv|jsonl&pipeline_id&since&limit` | streamed download (metrics need `pipeline_id`) |

Event types: `zone_intrusion`, `line_crossing`, `loitering`, `crowding`,
`source_down`. Event fields: `event_id, type, severity, pipeline_id,
camera_id, rule_id, frame_index, stream_time, created_at, message, track_ids,
zone_id, line_id, data, snapshot`.

## Models (registry)

| Method | Path | Role |
| --- | --- | --- |
| GET | `/v1/models` · `/v1/models/{name}` | viewer |
| PUT | `/v1/models/{name}/versions/{version}?format=onnx&sha256=&activate=false` | admin |
| GET / DELETE | `/v1/models/{name}/versions/{version}` (`?force=true` to delete the active version) | viewer / admin |
| POST | `/v1/models/{name}/activate` `{"version": "1.2.0"}` · `/deactivate` | admin |
| POST | `/v1/models/{name}/versions/{version}/verify` | viewer |

Upload the artefact as the raw request body (streamed, limited by
`max_upload_bytes`); optional JSON metadata in the `X-Model-Metadata` header,
e.g. `{"class_names": ["person", "car"], "input_size": [640, 640]}`. Versions
are semantic versions; formats: `onnx`, `npz`, `json`, `bin`.

```bash
curl -X PUT "localhost:8080/v1/models/yolo/versions/1.0.0?activate=true&sha256=$(sha256sum yolo.onnx | cut -d' ' -f1)" \
  -H "X-API-Key: $ADMIN_KEY" -H 'X-Model-Metadata: {"class_names":["person"]}' --data-binary @yolo.onnx
```

## Evaluation and analysis

| Method | Path | Role | Description |
| --- | --- | --- | --- |
| POST | `/v1/evaluations` | operator | run detector + tracker on a synthetic scene and score it (stored) |
| GET | `/v1/evaluations` · `/v1/evaluations/{id}` | viewer | stored reports |
| POST | `/v1/analyze/image` | operator | `{"image_b64", "detector"}` → detections (colour/template/ONNX only) |
| POST | `/v1/imaging/process` | operator | `{"image_b64", "ops": [...], "output_format"}` → processed image (base64) |

Evaluation request: `{"scene": <synthetic source spec, finite>, "detector": <spec>,
"tracker": <spec>, "iou_thresholds": [0.5], "coco_map": false, "tracking_iou": 0.5,
"skip_frames": null, "backend": "auto"}`. The report contains detection metrics per
threshold (tp/fp/fn, precision, recall, F1, AP, mAP), tracking metrics (MOTA,
MOTP, IDF1, IDP, IDR, ID switches, fragmentations, MT/PT/ML) and timing.
`skip_frames` defaults to the background detector's warm-up window.

Image ops: `resize {width, height?, interpolation}`, `gray`, `convert {to: bgr|hsv}`,
`blur {kind: gaussian|box, sigma, ksize}`, `canny {low, high, sigma}`, `sobel`,
`threshold {value? (Otsu when omitted)}`, `morphology {operation, ksize, shape, iterations}`.
