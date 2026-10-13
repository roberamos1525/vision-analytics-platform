"""Shared fixtures: synthetic scenes, backends, settings and API clients."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helpers import (
    ADMIN_KEY,
    HAS_OPENCV,
    OPERATOR_KEY,
    VIEWER_KEY,
    requires_opencv,
    two_object_scene,
)
from vap.config import Settings
from vap.imaging.backend import get_backend
from vap.imaging.ops import ImageBackend
from vap.sources.synthetic import SyntheticScene


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "opencv" in item.keywords and not HAS_OPENCV:
            item.add_marker(pytest.mark.skip(reason="opencv-python-headless not installed"))


BACKENDS = ["numpy", pytest.param("opencv", marks=[pytest.mark.opencv, requires_opencv])]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> ImageBackend:
    return get_backend(request.param)


@pytest.fixture
def numpy_backend() -> ImageBackend:
    return get_backend("numpy")


@pytest.fixture
def scene() -> SyntheticScene:
    return SyntheticScene(two_object_scene())


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    media = tmp_path / "media"
    media.mkdir()
    return Settings.model_validate(
        {
            "storage": {"state_dir": str(tmp_path / "state")},
            "pipelines": {"metrics_interval_s": 0.05, "image_backend": "numpy"},
            "media": {"roots": [str(media)]},
        }
    )


@pytest.fixture
def auth_settings(settings: Settings) -> Settings:
    data = settings.model_dump(mode="json")
    data["auth"] = {
        "enabled": True,
        "api_keys": {ADMIN_KEY: "admin", OPERATOR_KEY: "operator", VIEWER_KEY: "viewer"},
    }
    return Settings.model_validate(data)


@pytest.fixture
def client_factory() -> Iterator[Callable[..., Any]]:
    """Create TestClients that are closed (lifespan shutdown) after the test."""
    from fastapi.testclient import TestClient

    from vap.api.app import create_app
    from vap.runtime import Runtime

    clients: list[TestClient] = []

    def make(settings: Settings, **runtime_kwargs: Any) -> TestClient:
        runtime = Runtime(settings, **runtime_kwargs)
        client = TestClient(create_app(runtime=runtime))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)
