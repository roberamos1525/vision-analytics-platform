# Testing

```bash
python -m pytest -q                 # everything
python -m pytest -q tests/unit      # fast unit tests
python -m pytest -q -m "not opencv" # skip OpenCV-specific tests explicitly
```

## Layout

| Path | Contents |
| --- | --- |
| `tests/unit/` | module-level tests: geometry, imaging (both backends + parity), codecs/drawing/op chains, sources, RTSP adapter, detection, tracking, motion, analytics/rules, evaluation, events/webhooks, storage/registry, config/metrics/logging, exports/extraction, pipeline, API internals (auth, middleware, MJPEG generator) |
| `tests/integration/` | the FastAPI app through `TestClient` with real SQLite, the CLI through `vap.cli.main`, runtime/manager wiring (webhooks, restore after restart, probe timeouts) |
| `tests/helpers.py` | synthetic scene builders, API keys, `wait_until`, skip markers |
| `tests/fakes.py` | scriptable `FakeCapture` / `ScriptedFactory` standing in for `cv2.VideoCapture` |

## Synthetic data instead of fixtures

No binary fixtures are committed. Tests render scenes with
`vap.sources.synthetic`: moving rectangles/circles with known colours and
velocities, optional Gaussian noise, bouncing or leaving the frame, with
start/end frames. Ground-truth boxes are exact, which allows:

* detection tests asserting recall/precision on rendered frames,
* tracking tests with perfect detections (MOTA = IDF1 = 1),
* analytics tests with deterministic stream timestamps,
* evaluation tests with hand-computed AP / MOTA / IDF1 values.

## Optional dependencies

* OpenCV tests carry `@pytest.mark.opencv` and the `requires_opencv` skip
  marker; the `backend` fixture parametrises tests over `numpy` and (when
  installed) `opencv`. `tests/unit/test_imaging_parity.py` checks that the
  numpy backend matches OpenCV within ±1–2 grey levels (identical for
  morphology and connected components).
* SciPy-specific tests (`requires_scipy`) compare the numpy Hungarian solver
  with `scipy.optimize.linear_sum_assignment`.
* ONNX inference uses an injected fake session; onnxruntime is not required.

To verify the "no OpenCV" path without uninstalling it:

```bash
python -c "import sys; sys.modules['cv2'] = None; import pytest; raise SystemExit(pytest.main(['-q']))"
```

## RTSP

`tests/unit/test_rtsp.py` drives `RtspSource` with fake captures: retries with
recorded backoff delays, end-of-stream reconnects, stalled reads (frame
timeout), exceptions inside `read()`, attempt caps, `stop()` interrupting a
30-second backoff, credential redaction. `tests/integration/test_api_pipelines.py`
runs an RTSP camera through the API with a fake factory, including a
reconnect and the final `source_down` failure. No test touches a real camera.

## Results at release time

Recorded on Windows 11 / Python 3.11.9 with the pinned `requirements.lock`:

* `python -m pytest -q` → 403 passed (OpenCV 5.0 and SciPy installed).
* With `cv2` blocked as shown above → 331 passed, 72 skipped.
* `ruff check .` and `ruff format --check .` → clean.
