# Getting started

## 1. Install

Requires Python 3.11+.

```bash
git clone <this repository> && cd vision-analytics-platform
python -m venv .venv
. .venv/bin/activate                 # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.lock     # exact versions used during development (incl. OpenCV, SciPy)
pip install --no-deps -e .
```

Minimal install (numpy + Pillow core, no OpenCV/SciPy): `pip install -e .`.
Extras: `pip install -e ".[opencv,scipy,dev]"`, `".[onnx]"` for onnxruntime.

Check what is available:

```bash
vap config --config examples/configs/vap.toml   # needs VAP_API_KEYS or [auth] enabled=false
```

## 2. Run the service

```bash
export VAP_API_KEYS="dev-admin-key-0123456789:admin,dev-viewer-key-0123456789:viewer"
vap serve                       # 127.0.0.1:8080, state in ./var
# or: uvicorn --factory vap.api.app:app_from_env --port 8080
```

Open `http://127.0.0.1:8080/docs` for the interactive API. Keys must be at
least 16 characters. For a throw-away local session you can disable auth
explicitly with `VAP_AUTH__ENABLED=false` (never on a shared network).

## 3. First pipeline

```bash
API_KEY=dev-admin-key-0123456789 ./examples/api_walkthrough.sh
python examples/python_client.py --key dev-admin-key-0123456789
```

The walkthrough registers a synthetic camera, runs a pipeline with zones, a
tripwire and four rules, then fetches analytics, events, CSV/JSONL exports, an
annotated snapshot, a heatmap and an evaluation report.

Watch the live preview in a browser (the browser must send the key, e.g. via
a header extension) or with curl:

```bash
curl -H "X-API-Key: $API_KEY" "http://127.0.0.1:8080/v1/pipelines/lobby-live/preview.mjpg?max_frames=50" -o preview.mjpg
```

## 4. Offline workflow (no server)

```bash
./examples/offline_cli.sh       # generate -> analyze -> evaluate -> extract -> process
```

## 5. Using your own media

File and image-sequence sources may only read below configured media roots:

```bash
export VAP_MEDIA__ROOTS='["/srv/videos"]'
curl -X POST localhost:8080/v1/cameras -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"camera_id":"clip","source":{"kind":"file","path":"parking.mp4"}}'
```

RTSP cameras:

```json
{"camera_id": "door", "source": {"kind": "rtsp", "url": "rtsp://user:pass@10.0.0.20:554/stream1",
 "frame_timeout": 5, "reconnect_initial_delay": 0.5, "reconnect_max_delay": 30}}
```

Credentials in RTSP URLs are stored in SQLite but redacted in every API
response and log line. (The RTSP adapter has only been tested with fake
capture backends; see the README limitations.)

## 6. Tests and lint

```bash
python -m pytest -q
ruff check . && ruff format --check .
```
