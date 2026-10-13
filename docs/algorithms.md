# Algorithms and design notes

## Image processing (numpy backend)

* **Separable filters** — Gaussian and Sobel kernels are applied as two 1-D
  correlations over reflect-101 padded arrays (same border as OpenCV's
  default). Box blur uses cumulative sums, so its cost is independent of the
  kernel size.
* **Area resize** — exact box averaging for arbitrary (non-integer) scale
  factors by bilinearly sampling the *integral image* at fractional cell
  boundaries; the integral of a piecewise-constant image is piecewise
  bilinear, so this is exact. Upscaling with `area` falls back to bilinear.
* **HSV** — OpenCV conventions (H in 0..179, S and V in 0..255) so colour
  ranges are portable between backends.
* **Canny** — Gaussian smoothing, Sobel gradients, L2 magnitude, direction
  quantised into four bins, vectorised non-maximum suppression with shifted
  arrays, then hysteresis implemented as connected-component labelling of the
  weak-edge mask, keeping components that contain a strong pixel.
* **Morphology** — rank filters over the structuring element's offsets
  (separable for rectangles); borders behave like OpenCV (erosion pads with
  max, dilation with min), which the parity tests confirm bit-for-bit.
* **Connected components** — run-length labelling: each row becomes
  `[start, end)` runs via `np.diff`; overlapping runs between consecutive rows
  (with a one-pixel slack for 8-connectivity) are found with `searchsorted`
  and merged with union-find; labels are painted back with `np.repeat`.
  Python-level work scales with the number of runs, not pixels.

## Sources

* **Synthetic scenes** — object position is a closed-form function of the
  frame index (triangle-wave reflection for bouncing), so frames can be
  rendered out of order and ground truth is exact. Noise is seeded per frame.
* **RTSP adapter** — the capture factory defaults to
  `cv2.VideoCapture(url, CAP_FFMPEG, [OPEN_TIMEOUT, READ_TIMEOUT])`. A reader
  thread pushes into a mailbox that keeps only the newest frame (older unread
  frames are counted as dropped) and a terminal error that is delivered only
  after the pending frame. `read()` waits up to `frame_timeout`; timeouts and
  errors tear down the connection and reconnect. The backoff before attempt
  *n* is `min(initial * multiplier^(n-2), max)` with ±jitter, and failures that
  occurred before a reconnect cycle count towards *n*, so a camera that
  connects and immediately drops cannot cause a hot loop.

## Detection

* **Background model** — per-pixel running mean/variance of the blurred grey
  image. Foreground if `|x - μ| > k·σ` and `> min_diff`. Initialisation uses
  the per-pixel **median** of a strided warm-up window (median absolute
  deviation for σ), which rejects objects present during start-up as long as
  they cover a pixel in fewer than half the samples. Foreground pixels learn
  20× slower than background. Pixels that remain foreground and unchanged
  frame-to-frame for `absorb_after` frames are absorbed (removes ghosts, and
  eventually parked objects). The mask is opened/closed with elliptical
  kernels and converted to boxes via connected components.
* **Template matching** — zero-mean normalised cross-correlation. The
  numerator is a real-FFT convolution with the flipped zero-mean template; the
  denominator uses integral images of the image and its square for local
  energy. Equivalent to `TM_CCOEFF_NORMED` (agreement < 1e-3 in tests).
* **ONNX / YOLO** — letterbox to the model size (aspect preserved, grey 114
  padding), NCHW float input, decode `(1, N, 5+C)` (objectness × class) or
  `(1, 4+C, N)` layouts, class-aware NMS via the coordinate-offset trick, then
  map boxes back through the letterbox transform.

## Tracking

* **Hungarian** — shortest augmenting path with row/column potentials,
  O(n²m), inner column loop vectorised. Rectangular matrices are handled by
  transposing. Gating replaces disallowed pairs with a large finite cost
  before solving and filters them afterwards.
* **SORT** — state `[cx, cy, s, r, vcx, vcy, vs]` (area `s`, aspect `r`),
  constant velocity per frame, Joseph-form covariance update. Lost tracks
  follow the prediction so they can be re-associated after short occlusions.
* **Lifecycle** — tentative → confirmed after `min_hits` consecutive matches;
  confirmed tracks become lost on a miss and are deleted after `max_age`
  misses; tentative tracks are deleted on their first miss.

## Analytics and rules

* Anchor point: bottom-centre of the box (ground contact).
* Zones: even-odd ray casting; entries/exits update occupancy, peak and dwell
  (bounded reservoir for p50/p90). Finished tracks exit at their last-seen time.
* Tripwires: per (line, track) the last *non-zero* side is remembered; a
  crossing is counted when the side flips and the movement segment
  intersects the line segment (so touching the line or passing beside it does
  not count).
* Rules: intrusion (per-track cooldown), crossing (direction filter), loitering
  (once per visit, keyed by entry time), crowding (hold time, cooldown,
  re-arms when the count drops).

## Motion

* Three-frame differencing (`|f2-f1| ∧ |f3-f2|`) suppresses the ghost of
  two-frame differencing.
* Block matching: for each candidate displacement, one shifted copy of the
  current frame is compared against all blocks at once (SAD via reshape-sum);
  ties prefer the smallest displacement; low-texture blocks are marked invalid.
* Kinematics: least-squares slope of position vs. time over a window, which is
  robust to per-frame box jitter; heading 0° = image right, 90° = image up.

## Evaluation

* Detection: greedy score-ordered matching per frame (and per class when
  `class_aware`), cumulative precision/recall, AP with the precision envelope
  (all-point, or COCO 101-point), mAP over 0.50:0.95.
* CLEAR-MOT: correspondences from the previous frame are kept when still
  valid; the rest are assigned with the Hungarian algorithm; ID switches are
  counted against the last matched hypothesis of each ground-truth object.
* IDF1: global one-to-one matching of trajectories maximising identity true
  positives (equivalent to minimising IDFP + IDFN).
