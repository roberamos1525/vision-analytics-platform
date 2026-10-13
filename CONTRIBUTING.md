# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .
```

## Workflow

1. Create a branch; keep changes focused.
2. Write or update tests first where practical. Unit tests live in
   `tests/unit`, end-to-end tests (API via `TestClient`, CLI via `main()`) in
   `tests/integration`. Use the synthetic scene generator and the fakes in
   `tests/fakes.py` instead of binary fixtures.
3. Run the checks:
   ```bash
   python -m pytest -q
   ruff check . && ruff format --check .
   ```
4. Update docs (`API.md`, `docs/`) and `CHANGELOG.md` for user-visible changes.

## Conventions

- Python 3.11, full type hints, `from __future__ import annotations`.
- Raise `vap.errors` types (never bare `Exception`) so the API maps them to the
  standard envelope; validate every external input (pydantic models at the API
  and config boundaries, explicit checks in library constructors).
- Images are `uint8` RGB `HxWx3` (or `HxW`) internally; convert at boundaries.
- Optional dependencies (OpenCV, SciPy, onnxruntime) must be imported lazily
  and must fail with `BackendUnavailableError`. Every OpenCV-backed feature
  needs a numpy path or a clear error, and its tests must be marked
  `@pytest.mark.opencv` + `requires_opencv` so they skip cleanly.
- Do not put package code in directories named `data`, `docs`, `examples`,
  `build`, `dist`, `out` or `vendor`.
- Keep the core free of network access; tests must not open sockets except to
  in-process test clients.

## Commit messages

Imperative mood, short subject (`Add crowding rule cooldown`), body explaining
why when it is not obvious.
