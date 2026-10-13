# Operations runbook

## Running

```bash
export VAP_API_KEYS="<admin-key>:admin,<viewer-key>:viewer"
export VAP_STORAGE__STATE_DIR=/var/lib/vap
vap serve --config /etc/vap/vap.toml
```

The process is a single uvicorn worker on purpose: pipelines are threads that
own in-memory state, so **do not run multiple workers** against the same state
directory.

Shutdown (SIGINT/SIGTERM) stops all pipelines, flushes the event bus and closes
the database; pipelines keep `desired_state=running` and are restarted on the
next boot. Pipelines that fail on restore are logged and left stopped.

## Health and monitoring

| Endpoint | Use |
| --- | --- |
| `GET /healthz` | liveness probe |
| `GET /readyz` | readiness: 503 when the DB ping fails or the event bus thread is down |
| `GET /v1/health` | details: schema version, bus queue/drops/sink errors, webhook stats, backends |
| `GET /metrics` | Prometheus text format (needs a viewer key) |

Key metrics:

* `vap_frames_processed_total{pipeline}`, `vap_pipeline_fps{pipeline}` — throughput.
* `vap_frame_processing_seconds_bucket{pipeline}` — per-frame latency histogram.
* `vap_pipeline_errors_total{pipeline}` — crashed pipelines.
* `vap_events_total{pipeline,type}`, `vap_event_bus_dropped_total` — event volume / back-pressure.
* `vap_webhook_deliveries_total{outcome}` — `delivered`, `retry`, `failed`, `dropped`.
* `vap_http_requests_total{method,route,status}`, `vap_http_request_seconds` — API traffic.

Logs are JSON lines on stderr (`logging.format = "text"` for humans). Access
logs include `request_id`, which is echoed in the `X-Request-ID` header and in
error envelopes.

## Common situations

| Symptom | Check / action |
| --- | --- |
| Pipeline `failed` with `giving up after N connection attempts` | Camera unreachable. Check `status.source.health.last_error`; raise `max_reconnect_attempts` (null = retry forever) or fix the network. A `source_down` event was emitted. |
| `status.source.health.dropped_frames` grows | The pipeline is slower than the camera; only the newest frame is processed. Use `process_every_n`, `resize_width` or a lighter detector. |
| No detections for the first seconds | Background warm-up (`warmup_frames * warmup_stride` frames). Reduce for fast scenes. |
| Stationary objects disappear | Static absorption (`absorb_after` frames). Increase it or set `0` to disable (ghosts from the warm-up will then persist longer). |
| `backend_unavailable` (501) | Video files / RTSP / MOG2 / Farneback need OpenCV; ONNX needs onnxruntime. Install the extra. |
| `vap_event_bus_dropped_total` increases | Sinks are too slow or event rate too high. Add rule cooldowns; check webhook latency. |
| Webhooks rejected at start-up | Target resolves to a private address; set `webhooks.allow_private_networks = true` only for trusted receivers. |

## Backup and retention

* Back up `state_dir`. For a consistent copy while running use
  `sqlite3 vap.db ".backup backup.db"` (WAL-safe) and copy `models/` and `snapshots/`.
* Events and metric samples older than `storage.event_retention_days` are
  purged at start-up. Snapshot files are not purged automatically; remove old
  files with a cron job if disk space matters.
* Model artefacts can be checked at any time with
  `POST /v1/models/{name}/versions/{version}/verify` or
  `vap models verify --name ... --version ...`.

## Capacity notes

Measured on the development machine (Windows 11, Python 3.11, 320x240
synthetic frames): the background detector took roughly 3 ms per frame and a
full pipeline processed about 60 frames/s in non-real-time mode. These are
indicative only; real cameras at higher resolutions will be slower — use
`resize_width` to bound per-frame cost.

## Upgrades

Schema changes are applied automatically at start-up by numbered migrations.
Take a backup before upgrading; migrations are forward-only.
