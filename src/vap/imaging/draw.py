"""Annotation primitives used for previews and event snapshots.

Everything draws in place on an RGB ``uint8`` array and clips to the image, so
callers can pass boxes that are partly outside the frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vap.geometry import BBox, LineSegment, Polygon

Color = tuple[int, int, int]

# 3x5 bitmap glyphs (rows top->bottom, 3 bits each, MSB = left column).
_GLYPHS: dict[str, tuple[int, int, int, int, int]] = {
    "0": (0b111, 0b101, 0b101, 0b101, 0b111),
    "1": (0b010, 0b110, 0b010, 0b010, 0b111),
    "2": (0b111, 0b001, 0b111, 0b100, 0b111),
    "3": (0b111, 0b001, 0b111, 0b001, 0b111),
    "4": (0b101, 0b101, 0b111, 0b001, 0b001),
    "5": (0b111, 0b100, 0b111, 0b001, 0b111),
    "6": (0b111, 0b100, 0b111, 0b101, 0b111),
    "7": (0b111, 0b001, 0b010, 0b010, 0b010),
    "8": (0b111, 0b101, 0b111, 0b101, 0b111),
    "9": (0b111, 0b101, 0b111, 0b001, 0b111),
    "#": (0b101, 0b111, 0b101, 0b111, 0b101),
    "-": (0b000, 0b000, 0b111, 0b000, 0b000),
    ".": (0b000, 0b000, 0b000, 0b000, 0b010),
    ":": (0b000, 0b010, 0b000, 0b010, 0b000),
    " ": (0b000, 0b000, 0b000, 0b000, 0b000),
}

PALETTE: tuple[Color, ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
)


def color_for(index: int) -> Color:
    return PALETTE[index % len(PALETTE)]


def draw_rect(image: np.ndarray, box: BBox, color: Color, thickness: int = 1) -> None:
    h, w = image.shape[:2]
    x1, y1 = round(box.x1), round(box.y1)
    x2, y2 = round(box.x2) - 1, round(box.y2) - 1
    t = max(1, thickness)
    for i in range(t):
        top, bottom, left, right = y1 + i, y2 - i, x1 + i, x2 - i
        if left > right or top > bottom:
            break
        cl, cr = max(left, 0), min(right, w - 1)
        ct, cb = max(top, 0), min(bottom, h - 1)
        if cl <= cr:
            if 0 <= top < h:
                image[top, cl : cr + 1] = color
            if 0 <= bottom < h:
                image[bottom, cl : cr + 1] = color
        if ct <= cb:
            if 0 <= left < w:
                image[ct : cb + 1, left] = color
            if 0 <= right < w:
                image[ct : cb + 1, right] = color


def fill_rect(image: np.ndarray, box: BBox, color: Color, alpha: float = 1.0) -> None:
    h, w = image.shape[:2]
    clipped = box.clip(w, h)
    x1, y1, x2, y2 = (round(v) for v in clipped.to_list())
    if x2 <= x1 or y2 <= y1:
        return
    region = image[y1:y2, x1:x2]
    blend_into(region, np.ones(region.shape[:2], dtype=bool), color, alpha)


def blend_into(region: np.ndarray, mask: np.ndarray, color: Color, alpha: float) -> None:
    if alpha >= 1.0:
        region[mask] = color
        return
    src = region[mask].astype(np.float32)
    col = np.asarray(color, dtype=np.float32)
    region[mask] = np.clip(src * (1 - alpha) + col * alpha + 0.5, 0, 255).astype(np.uint8)


def draw_line(
    image: np.ndarray, p0: Sequence[float], p1: Sequence[float], color: Color, thickness: int = 1
) -> None:
    h, w = image.shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int64)
    r = max(thickness, 1) // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            px, py = xs + dx, ys + dy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            image[py[ok], px[ok]] = color


def draw_polygon(
    image: np.ndarray, polygon: Polygon, color: Color, thickness: int = 1, fill_alpha: float = 0.0
) -> None:
    if fill_alpha > 0:
        h, w = image.shape[:2]
        blend_into(image, polygon.mask(h, w), color, fill_alpha)
    pts = polygon.points
    for i, p in enumerate(pts):
        draw_line(image, p, pts[(i + 1) % len(pts)], color, thickness)


def draw_tripwire(image: np.ndarray, line: LineSegment, color: Color, thickness: int = 2) -> None:
    draw_line(image, line.start, line.end, color, thickness)
    mx = (line.start[0] + line.end[0]) / 2
    my = (line.start[1] + line.end[1]) / 2
    dx, dy = line.end[0] - line.start[0], line.end[1] - line.start[1]
    norm = max(line.length, 1e-9)
    # Tick pointing towards the positive side (right-hand side as drawn on screen).
    nx, ny = -dy / norm, dx / norm
    draw_line(image, (mx, my), (mx + nx * 10, my + ny * 10), color, 1)


def draw_text(
    image: np.ndarray, text: str, origin: Sequence[float], color: Color, scale: int = 2
) -> None:
    """Render digits and a few symbols with a 3x5 bitmap font (unknown chars become spaces)."""
    h, w = image.shape[:2]
    x, y = round(origin[0]), round(origin[1])
    s = max(1, scale)
    for ch in text:
        glyph = _GLYPHS.get(ch, _GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for col in range(3):
                if bits & (1 << (2 - col)):
                    gx, gy = x + col * s, y + row * s
                    xa, xb = max(gx, 0), min(gx + s, w)
                    ya, yb = max(gy, 0), min(gy + s, h)
                    if xa < xb and ya < yb:
                        image[ya:yb, xa:xb] = color
        x += 4 * s


def text_size(text: str, scale: int = 2) -> tuple[int, int]:
    s = max(1, scale)
    return (max(len(text) * 4 * s - s, 0), 5 * s)


def draw_label(image: np.ndarray, box: BBox, text: str, color: Color, scale: int = 2) -> None:
    tw, th = text_size(text, scale)
    top = box.y1 - th - 2 if box.y1 - th - 2 >= 0 else box.y1
    fill_rect(image, BBox(box.x1, top, box.x1 + tw + 2, top + th + 2), color)
    draw_text(image, text, (box.x1 + 1, top + 1), (0, 0, 0), scale)
