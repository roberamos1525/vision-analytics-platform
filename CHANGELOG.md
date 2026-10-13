# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [0.1.0] - 2026-10-12

### Added
- Image-processing backend interface with pure-numpy and OpenCV implementations
  (resize, colour conversion, blur, Sobel, Canny, morphology, Otsu, connected
  components), Pillow codecs with pixel limits, drawing helpers and op chains.
- Video sources: synthetic scenes with exact ground truth, image sequences,
  OpenCV video files, and an RTSP adapter with reader thread, frame timeouts,
  exponential backoff, attempt caps and health reporting.
- Frame extraction (every N, interval, scene change) with manifests.
- Detectors: background subtraction (running Gaussian with median warm-up and
  static absorption, OpenCV MOG2), HSV colour ranges, ZNCC template matching,
  frame differencing and a YOLO-style ONNX adapter; numpy NMS.
- Trackers: IoU and SORT (Kalman + numpy Hungarian), track lifecycle.
- Motion: differencing, heatmaps, block-matching flow, kinematics.
- Streaming analytics engine (zones, tripwires, occupancy, dwell, rolling
  counters) and rules (zone intrusion, line crossing, loitering, crowding).
- Event bus with SQLite, in-memory, JSONL and signed webhook sinks.
- Model registry with semantic versions, checksums and activation.
- Evaluation: precision/recall/AP/mAP and CLEAR-MOT/IDF1.
- CSV/JSONL exports, SQLite persistence with migrations, Prometheus metrics,
  JSON logging.
- FastAPI service (auth roles, error envelopes, body limits, MJPEG preview)
  and `vap` CLI.
- Docs, examples, Dockerfile, compose file and CI workflow (container and CI
  not executed locally).
