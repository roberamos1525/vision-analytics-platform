"""Image decoding/encoding through Pillow with decompression-bomb guards."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from vap.errors import InvalidInputError, PayloadTooLargeError
from vap.imaging.frame import ensure_image

ImageFormat = Literal["jpeg", "png", "bmp"]
SUPPORTED_SUFFIXES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".bmp": "bmp"}
DEFAULT_MAX_PIXELS = 40_000_000


def decode_image(data: bytes, *, max_pixels: int = DEFAULT_MAX_PIXELS) -> np.ndarray:
    """Decode encoded image bytes into an RGB ``uint8`` array.

    The header is inspected before pixel data is decoded so oversized images
    are rejected without allocating their full buffer.
    """
    if not data:
        raise InvalidInputError("image payload is empty")
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise PayloadTooLargeError(
                    "image exceeds the maximum pixel count",
                    details={"width": width, "height": height, "max_pixels": max_pixels},
                )
            img.load()
            rgb = img.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidInputError(f"could not decode image: {exc}") from exc


def encode_image(image: np.ndarray, fmt: ImageFormat = "jpeg", quality: int = 85) -> bytes:
    image = ensure_image(image)
    if fmt not in ("jpeg", "png", "bmp"):
        raise InvalidInputError(f"unsupported image format {fmt!r}")
    if not 1 <= quality <= 100:
        raise InvalidInputError("quality must be in [1, 100]")
    buf = io.BytesIO()
    pil = Image.fromarray(image)
    if fmt == "jpeg":
        pil.save(buf, format="JPEG", quality=quality)
    else:
        pil.save(buf, format=fmt.upper())
    return buf.getvalue()


def format_for_path(path: Path) -> ImageFormat:
    fmt = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        raise InvalidInputError(f"unsupported image extension {path.suffix!r}")
    return fmt  # type: ignore[return-value]


def load_image(path: str | Path, *, max_pixels: int = DEFAULT_MAX_PIXELS) -> np.ndarray:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read image {p}: {exc}") from exc
    return decode_image(data, max_pixels=max_pixels)


def save_image(path: str | Path, image: np.ndarray, quality: int = 90) -> Path:
    p = Path(path)
    fmt = format_for_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(encode_image(image, fmt, quality))
    tmp.replace(p)
    return p
