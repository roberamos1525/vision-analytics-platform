"""Minimal Python client for the REST API (uses httpx).

Usage:
    VAP_API_KEYS="dev-admin-key-0123456789:admin" vap serve &
    python examples/python_client.py --key dev-admin-key-0123456789

It registers a looping synthetic camera, starts a real-time pipeline, reads a
few MJPEG preview frames, prints live analytics and stops the pipeline.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--key", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base, headers={"X-API-Key": args.key}, timeout=30) as client:
        camera = {
            "camera_id": "lobby",
            "name": "Lobby (synthetic)",
            "source": {
                "kind": "synthetic",
                "num_frames": 300,
                "num_objects": 5,
                "seed": 11,
                "loop": True,
            },
        }
        r = client.post("/v1/cameras", json=camera)
        if r.status_code not in (201, 409):
            r.raise_for_status()

        config = {
            "zones": [
                {
                    "zone_id": "lobby",
                    "points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                    "normalized": True,
                }
            ],
            "rules": [
                {"kind": "loitering", "rule_id": "loiter", "zone_id": "lobby", "min_dwell_s": 3},
                {
                    "kind": "crowding",
                    "rule_id": "crowd",
                    "zone_id": "lobby",
                    "min_count": 3,
                    "cooldown_s": 10,
                },
            ],
            "realtime": True,
        }
        r = client.post(
            "/v1/pipelines",
            json={"pipeline_id": "lobby-live", "camera_id": "lobby", "config": config},
        )
        if r.status_code not in (201, 409):
            r.raise_for_status()
        client.post("/v1/pipelines/lobby-live/start").raise_for_status()

        with client.stream(
            "GET", "/v1/pipelines/lobby-live/preview.mjpg", params={"max_frames": 5, "fps": 5}
        ) as stream:
            size = sum(len(chunk) for chunk in stream.iter_bytes())
        print(f"received {size} bytes of MJPEG preview")

        time.sleep(args.seconds)
        analytics = client.get("/v1/pipelines/lobby-live/analytics").json()
        print(
            json.dumps(
                {k: analytics[k] for k in ("live_tracks", "unique_tracks", "zones")}, indent=2
            )
        )
        print(
            json.dumps(client.get("/v1/events/recent", params={"limit": 5}).json(), indent=2)[:1500]
        )

        status = client.post("/v1/pipelines/lobby-live/stop").json()
        print(f"stopped after {status['frames_processed']} frames")


if __name__ == "__main__":
    main()
