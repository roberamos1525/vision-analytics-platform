"""Multi-object tracking metrics: CLEAR-MOT (MOTA/MOTP) and identity metrics (IDF1).

Inputs are per-frame lists of ``(id, bbox)`` for ground truth and hypotheses.
CLEAR-MOT matching keeps last frame's correspondences when they are still
valid (IoU >= threshold) and solves the rest with the Hungarian algorithm, as
in Bernardin & Stiefelhagen (2008). IDF1 follows Ristani et al. (2016):
trajectories are matched one-to-one globally to maximise identity true
positives.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from vap.errors import InvalidInputError
from vap.geometry import BBox, boxes_to_array, iou_matrix
from vap.tracking.assignment import hungarian, match

FrameObjects = Sequence[tuple[int, BBox]]


@dataclass(slots=True)
class MOTReport:
    num_frames: int
    num_gt: int
    num_hyp: int
    matches: int
    false_positives: int
    misses: int
    id_switches: int
    fragmentations: int
    mota: float
    motp: float
    idtp: int
    idfp: int
    idfn: int
    idf1: float
    idp: float
    idr: float
    gt_tracks: int
    hyp_tracks: int
    mostly_tracked: int
    partially_tracked: int
    mostly_lost: int

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        for key in ("mota", "motp", "idf1", "idp", "idr"):
            out[key] = round(out[key], 6)
        return out


def _ious(gt: FrameObjects, hyp: FrameObjects) -> np.ndarray:
    return iou_matrix(boxes_to_array(b for _, b in gt), boxes_to_array(b for _, b in hyp))


def evaluate_mot(
    gt_frames: Mapping[int, FrameObjects],
    hyp_frames: Mapping[int, FrameObjects],
    iou_threshold: float = 0.5,
) -> MOTReport:
    if not 0.0 < iou_threshold <= 1.0:
        raise InvalidInputError("IoU threshold must be in (0, 1]")
    frames = sorted(set(gt_frames) | set(hyp_frames))
    prev_match: dict[int, int] = {}  # gt id -> hyp id in the previous frame
    last_match: dict[int, int] = {}  # gt id -> last hyp id ever matched
    tracked_frames: dict[int, int] = defaultdict(int)
    present_frames: dict[int, int] = defaultdict(int)
    was_tracked: dict[int, bool] = {}
    frag = 0
    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    num_gt = num_hyp = 0
    pair_overlap: dict[tuple[int, int], int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    hyp_counts: dict[int, int] = defaultdict(int)

    for f in frames:
        gt = list(gt_frames.get(f, ()))
        hyp = list(hyp_frames.get(f, ()))
        num_gt += len(gt)
        num_hyp += len(hyp)
        ious = _ious(gt, hyp)
        for gi, (gid, _) in enumerate(gt):
            gt_counts[gid] += 1
            present_frames[gid] += 1
            for hi, (hid, _) in enumerate(hyp):
                if ious[gi, hi] >= iou_threshold:
                    pair_overlap[(gid, hid)] += 1
        for hid, _ in hyp:
            hyp_counts[hid] += 1

        # 1) keep previous correspondences that are still valid
        frame_pairs: list[tuple[int, int]] = []
        gt_idx = {gid: i for i, (gid, _) in enumerate(gt)}
        hyp_idx = {hid: i for i, (hid, _) in enumerate(hyp)}
        used_g: set[int] = set()
        used_h: set[int] = set()
        for gid, hid in prev_match.items():
            gi, hi = gt_idx.get(gid), hyp_idx.get(hid)
            if gi is None or hi is None or hi in used_h:
                continue
            if ious[gi, hi] >= iou_threshold:
                frame_pairs.append((gi, hi))
                used_g.add(gi)
                used_h.add(hi)
        # 2) Hungarian on the remainder
        rem_g = [i for i in range(len(gt)) if i not in used_g]
        rem_h = [i for i in range(len(hyp)) if i not in used_h]
        if rem_g and rem_h:
            sub = 1.0 - ious[np.ix_(rem_g, rem_h)]
            pairs, _, _ = match(sub, 1.0 - iou_threshold)
            frame_pairs.extend((rem_g[a], rem_h[b]) for a, b in pairs)

        new_prev: dict[int, int] = {}
        matched_g = set()
        for gi, hi in frame_pairs:
            gid, hid = gt[gi][0], hyp[hi][0]
            matched_g.add(gid)
            if gid in last_match and last_match[gid] != hid:
                idsw += 1
            last_match[gid] = hid
            new_prev[gid] = hid
            iou_sum += float(ious[gi, hi])
            tracked_frames[gid] += 1
        for gid, _ in gt:
            tracked = gid in matched_g
            if tracked and was_tracked.get(gid) is False:
                frag += 1
            if gid in was_tracked or tracked:
                was_tracked[gid] = tracked
        prev_match = new_prev
        tp += len(frame_pairs)
        fp += len(hyp) - len(frame_pairs)
        fn += len(gt) - len(frame_pairs)

    mota = 1.0 - (fn + fp + idsw) / num_gt if num_gt else 0.0
    motp = iou_sum / tp if tp else 0.0

    gt_ids = sorted(gt_counts)
    hyp_ids = sorted(hyp_counts)
    idtp = 0
    if gt_ids and hyp_ids:
        overlap = np.zeros((len(gt_ids), len(hyp_ids)))
        gpos = {g: i for i, g in enumerate(gt_ids)}
        hpos = {h: i for i, h in enumerate(hyp_ids)}
        for (g, h), n in pair_overlap.items():
            overlap[gpos[g], hpos[h]] = n
        rows, cols = hungarian(-overlap)
        idtp = int(overlap[rows, cols].sum())
    idfn = num_gt - idtp
    idfp = num_hyp - idtp
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (idtp + idfp + idfn) else 0.0
    idp = idtp / (idtp + idfp) if idtp + idfp else 0.0
    idr = idtp / (idtp + idfn) if idtp + idfn else 0.0

    mt = pt = ml = 0
    for gid in gt_ids:
        ratio = tracked_frames[gid] / present_frames[gid]
        if ratio >= 0.8:
            mt += 1
        elif ratio < 0.2:
            ml += 1
        else:
            pt += 1

    return MOTReport(
        num_frames=len(frames),
        num_gt=num_gt,
        num_hyp=num_hyp,
        matches=tp,
        false_positives=fp,
        misses=fn,
        id_switches=idsw,
        fragmentations=frag,
        mota=mota,
        motp=motp,
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idf1=idf1,
        idp=idp,
        idr=idr,
        gt_tracks=len(gt_ids),
        hyp_tracks=len(hyp_ids),
        mostly_tracked=mt,
        partially_tracked=pt,
        mostly_lost=ml,
    )
