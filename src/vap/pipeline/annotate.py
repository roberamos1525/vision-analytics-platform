"""Render analytics overlays (zones, tripwires, tracks) onto frames."""

from __future__ import annotations

import itertools

import numpy as np

from vap.analytics.zones import Tripwire, Zone
from vap.imaging import draw
from vap.motion.heatmap import MotionHeatmap
from vap.tracking.base import Track

ZONE_COLOR = (0, 200, 255)
LINE_COLOR = (255, 0, 160)
HIGHLIGHT = (255, 40, 40)


def annotate(
    image: np.ndarray,
    tracks: list[Track],
    zones: list[Zone],
    lines: list[Tripwire],
    *,
    heatmap: MotionHeatmap | None = None,
    highlight: set[int] | None = None,
    trails: bool = True,
) -> np.ndarray:
    """Return an annotated copy of ``image``."""
    out = heatmap.overlay(image, 0.45) if heatmap is not None and heatmap.updates else image.copy()
    for zone in zones:
        draw.draw_polygon(out, zone.polygon, ZONE_COLOR, thickness=1, fill_alpha=0.15)
    for line in lines:
        draw.draw_tripwire(out, line.segment, LINE_COLOR, thickness=2)
    marked = highlight or set()
    for track in tracks:
        color = HIGHLIGHT if track.track_id in marked else draw.color_for(track.track_id)
        draw.draw_rect(out, track.bbox, color, thickness=3 if track.track_id in marked else 2)
        draw.draw_label(out, track.bbox, f"#{track.track_id}", color)
        if trails and len(track.history) > 1:
            pts = list(track.history)[-30:]
            for a, b in itertools.pairwise(pts):
                draw.draw_line(out, (a.x, a.y), (b.x, b.y), color, 1)
    return out
