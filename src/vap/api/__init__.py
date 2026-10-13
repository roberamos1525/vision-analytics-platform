"""REST API (FastAPI)."""

from vap.api.app import app_from_env, create_app

__all__ = ["app_from_env", "create_app"]
