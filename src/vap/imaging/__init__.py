"""Image processing: frame container, backends, codecs and drawing helpers."""

from vap.imaging.backend import available_backends, get_backend
from vap.imaging.components import Blob, find_blobs, label_components
from vap.imaging.frame import ColorSpace, Frame, ensure_gray, ensure_image
from vap.imaging.ops import ImageBackend

__all__ = [
    "Blob",
    "ColorSpace",
    "Frame",
    "ImageBackend",
    "available_backends",
    "ensure_gray",
    "ensure_image",
    "find_blobs",
    "get_backend",
    "label_components",
]
