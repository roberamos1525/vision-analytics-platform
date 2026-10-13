#!/usr/bin/env bash
# End-to-end walkthrough against a running server.
#
#   VAP_API_KEYS="dev-admin-key-0123456789:admin" vap serve &
#   API_KEY=dev-admin-key-0123456789 ./examples/api_walkthrough.sh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8080}"
KEY="${API_KEY:?set API_KEY}"
HERE="$(cd "$(dirname "$0")" && pwd)"
auth=(-H "X-API-Key: ${KEY}")
json=(-H "Content-Type: application/json")

echo "== health"
curl -s "${BASE}/healthz"; echo

echo "== register a synthetic camera"
curl -s -X POST "${BASE}/v1/cameras" "${auth[@]}" "${json[@]}" \
  -d '{"camera_id":"demo","name":"Synthetic demo","source":{"kind":"synthetic","num_frames":300,"num_objects":4,"seed":7}}'
echo

echo "== probe it"
curl -s -X POST "${BASE}/v1/cameras/demo/probe" "${auth[@]}"; echo

echo "== create and start a pipeline with zones, a tripwire and rules"
config="$(cat "${HERE}/configs/pipeline_retail.json")"
curl -s -X POST "${BASE}/v1/pipelines" "${auth[@]}" "${json[@]}" \
  -d "{\"pipeline_id\":\"demo-p\",\"camera_id\":\"demo\",\"start\":true,\"config\":${config}}" > /dev/null

echo "== wait for completion"
for _ in $(seq 1 60); do
  state="$(curl -s "${BASE}/v1/pipelines/demo-p/status" "${auth[@]}" | python -c 'import sys,json;print(json.load(sys.stdin)["state"])')"
  [ "${state}" = "completed" ] && break
  sleep 0.5
done
echo "state=${state}"

echo "== live analytics / metrics"
curl -s "${BASE}/v1/pipelines/demo-p/analytics" "${auth[@]}" | head -c 600; echo
curl -s "${BASE}/v1/pipelines/demo-p/metrics" "${auth[@]}"; echo

echo "== events"
curl -s "${BASE}/v1/events/stats" "${auth[@]}"; echo
curl -s "${BASE}/v1/events?limit=3" "${auth[@]}" | head -c 600; echo

echo "== exports"
curl -s "${BASE}/v1/exports/events?format=csv&limit=5" "${auth[@]}"
curl -s "${BASE}/v1/exports/tracks?format=jsonl&limit=3" "${auth[@]}"

echo "== annotated snapshot + heatmap"
curl -s -o demo_snapshot.jpg "${BASE}/v1/pipelines/demo-p/snapshot.jpg" "${auth[@]}"
curl -s -o demo_heatmap.png "${BASE}/v1/pipelines/demo-p/heatmap.png" "${auth[@]}"
ls -l demo_snapshot.jpg demo_heatmap.png

echo "== evaluation on synthetic ground truth"
curl -s -X POST "${BASE}/v1/evaluations" "${auth[@]}" "${json[@]}" \
  -d @"${HERE}/configs/evaluation_request.json" | head -c 800; echo

echo "== Prometheus metrics"
curl -s "${BASE}/metrics" "${auth[@]}" | grep -E '^vap_(frames_processed|events)_total' || true
