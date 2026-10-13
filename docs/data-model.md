# Data model

State lives in one SQLite database (`<state_dir>/vap.db`, WAL mode) plus files
under `<state_dir>`:

```
var/
├── vap.db                     SQLite database (+ -wal / -shm)
├── snapshots/                 annotated JPEGs referenced by events.snapshot
├── models/<name>/<version>/   model artefacts (model.<ext>)
└── exports/extractions/...    frame-extraction jobs (images + manifest.jsonl)
```

## Schema (migration 1)

Migrations are applied in order inside `BEGIN IMMEDIATE` transactions and
recorded in `schema_version`; re-running them is a no-op.

### cameras
| column | type | notes |
| --- | --- | --- |
| camera_id | TEXT PK | validated id |
| name | TEXT | display name |
| spec_json | TEXT | validated source spec (RTSP credentials included — redacted on read by the API) |
| created_at, updated_at | REAL | unix seconds |

### pipelines
| column | type | notes |
| --- | --- | --- |
| pipeline_id | TEXT PK | |
| camera_id | TEXT FK → cameras (ON DELETE RESTRICT) | a camera cannot be deleted while referenced |
| config_json | TEXT | validated `PipelineConfig` |
| desired_state | TEXT | `running` / `stopped`; `running` pipelines are restarted on boot |
| created_at, updated_at | REAL | |

### pipeline_runs
One row per start: `run_id` (PK), `pipeline_id`, `started_at`, `stopped_at`,
`frames`, `status` (`running`, `completed`, `stopped`, `failed`), `error`.

### events
`event_id` (PK, uuid4 hex), `type`, `severity`, `pipeline_id`, `camera_id`,
`rule_id`, `frame_index`, `stream_time`, `created_at`, `message`, `track_ids`
(JSON array), `zone_id`, `line_id`, `data_json`, `snapshot` (file name).
Indexes: `(pipeline_id, created_at)`, `(type, created_at)`. Inserts are
idempotent (`INSERT OR IGNORE`). Rows older than `event_retention_days` are
purged at start-up. Pagination uses the `before=<created_at>` cursor.

### tracks
Summary of each confirmed track when it ends: `(run_id, track_id)` PK,
`pipeline_id`, `label`, first/last frame and timestamp, `hits`,
`path_length` (px), `mean_speed` (px/s along the path), `direction`
(8-way compass from the least-squares heading), `last_bbox` (JSON),
`created_at`.

### metric_samples
Periodic snapshots per pipeline (`metrics_interval_s`, plus a final sample):
`pipeline_id`, `ts`, `data_json` = `{frames_processed, detections, events, fps,
live_tracks, unique_tracks, zones: {zone_id: occupancy}, lines: {line_id: total}}`.
Purged with the same retention as events.

### models
`(name, version)` PK, `format`, `sha256`, `size_bytes`, `path`,
`metadata_json`, `active`, `created_at`. A partial unique index
`ON models(name) WHERE active = 1` guarantees at most one active version per
model; activation swaps flags in a single transaction.

### evaluations
`evaluation_id` PK, `created_at`, `status` (`running`/`completed`/`failed`),
`request_json`, `report_json`, `error`.

## In-memory structures

* **Event ring buffer** (`pipelines.recent_events`, default 500) backs
  `/v1/events/recent`.
* **Pipeline state** (latest frame, tracks, heatmap, analytics counters) lives
  in the `CameraPipeline` object; after a pipeline finishes, the last run stays
  queryable until the pipeline is restarted or deleted (or the process exits).
* **Prometheus registry**: counters/gauges/histograms are process-local and
  reset on restart.

## JSONL interchange formats

Ground truth and predictions for `vap evaluate --gt/--pred` and the files
written by `vap generate` / `vap analyze`:

```json
{"frame": 12, "id": 3, "bbox": [x1, y1, x2, y2], "label": "object", "score": 0.93}
```

`id` is optional for predictions (tracking metrics are computed only when all
predictions have one); `score` defaults to 1.0.
