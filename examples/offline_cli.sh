#!/usr/bin/env bash
# Offline workflow with the CLI only (no server):
# render a synthetic video with ground truth, analyse it, score it, extract keyframes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-./vap-demo}"
mkdir -p "${OUT}"

vap generate --out "${OUT}/scene" --params seed=3,objects=4,frames=200
vap analyze "${OUT}/scene" --fps 15 --config "${HERE}/configs/pipeline_retail.json" \
  --out "${OUT}/predictions.jsonl" --events "${OUT}/events.jsonl"
vap evaluate --gt "${OUT}/scene/ground_truth.jsonl" --pred "${OUT}/predictions.jsonl" --iou 0.3,0.5
vap evaluate --request "${HERE}/configs/evaluation_request.json" --min-mota 0.5
vap extract "${OUT}/scene" --fps 15 --mode scene_change --threshold 0.2 --out "${OUT}/keyframes"
vap process "${OUT}/scene/frame_00050.png" \
  --ops '[{"op":"gray"},{"op":"blur","sigma":1.2},{"op":"canny","low":40,"high":110}]' \
  --out "${OUT}/edges.png"
