"""Configuration: defaults < TOML file < environment variables.

Environment overrides use ``VAP_<SECTION>__<FIELD>`` (double underscore), e.g.
``VAP_SERVER__PORT=9000``. Two shortcuts exist for secrets: ``VAP_API_KEYS``
(``key:role,key2:role``) and ``VAP_WEBHOOK_SECRET``. List values accept JSON
(``["a","b"]``) or comma-separated strings.
"""

from __future__ import annotations

import enum
import json
import os
import tomllib
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from vap.errors import ConfigError

ENV_PREFIX = "VAP_"


class Role(enum.StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return {"viewer": 0, "operator": 1, "admin": 2}[self.value]

    def allows(self, required: Role) -> bool:
        return self.rank >= required.rank


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerSettings(_Section):
    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    max_request_bytes: int = Field(2 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    max_upload_bytes: int = Field(200 * 1024 * 1024, ge=1024, le=4 * 1024 * 1024 * 1024)
    max_image_pixels: int = Field(25_000_000, ge=1024, le=200_000_000)


class AuthSettings(_Section):
    enabled: bool = True
    api_keys: dict[str, Role] = Field(default_factory=dict)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_keys(cls, value: Any) -> Any:
        if isinstance(value, str):
            out: dict[str, str] = {}
            for item in filter(None, (p.strip() for p in value.split(","))):
                key, sep, role = item.rpartition(":")
                if not sep or not key:
                    raise ValueError("api_keys entries must look like 'key:role'")
                out[key] = role
            return out
        return value

    @field_validator("api_keys")
    @classmethod
    def _check_keys(cls, value: dict[str, Role]) -> dict[str, Role]:
        for key in value:
            if len(key) < 16:
                raise ValueError("API keys must be at least 16 characters")
        return value

    @model_validator(mode="after")
    def _keys_when_enabled(self) -> AuthSettings:
        if self.enabled and not self.api_keys:
            raise ValueError("auth is enabled but no API keys are configured (set VAP_API_KEYS)")
        return self


class StorageSettings(_Section):
    state_dir: Path = Path("var")
    database: str = "vap.db"
    event_retention_days: int = Field(30, ge=1, le=3650)

    @property
    def database_path(self) -> Path:
        return self.state_dir / self.database

    @property
    def snapshots_dir(self) -> Path:
        return self.state_dir / "snapshots"

    @property
    def models_dir(self) -> Path:
        return self.state_dir / "models"

    @property
    def exports_dir(self) -> Path:
        return self.state_dir / "exports"


class PipelineSettings(_Section):
    max_pipelines: int = Field(8, ge=1, le=256)
    image_backend: Literal["auto", "numpy", "opencv"] = "auto"
    metrics_interval_s: float = Field(5.0, gt=0, le=3600)
    snapshot_quality: int = Field(85, ge=10, le=100)
    preview_quality: int = Field(70, ge=10, le=100)
    recent_events: int = Field(500, ge=10, le=100_000)
    stop_timeout_s: float = Field(5.0, gt=0, le=120)


class WebhookSettings(_Section):
    urls: list[str] = Field(default_factory=list, max_length=16)
    secret: str | None = None
    timeout_s: float = Field(5.0, gt=0, le=60)
    max_retries: int = Field(3, ge=0, le=10)
    backoff_s: float = Field(0.5, ge=0, le=60)
    allow_private_networks: bool = False
    queue_size: int = Field(1000, ge=1, le=100_000)
    event_types: list[str] = Field(default_factory=list)


class MediaSettings(_Section):
    roots: list[Path] = Field(default_factory=list, max_length=32)


class LoggingSettings(_Section):
    level: Literal["debug", "info", "warning", "error"] = "info"
    format: Literal["json", "text"] = "json"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=lambda: AuthSettings(enabled=False))
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pipelines: PipelineSettings = Field(default_factory=PipelineSettings)
    webhooks: WebhookSettings = Field(default_factory=WebhookSettings)
    media: MediaSettings = Field(default_factory=MediaSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def ensure_dirs(self) -> None:
        for path in (
            self.storage.state_dir,
            self.storage.snapshots_dir,
            self.storage.models_dir,
            self.storage.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _is_list_field(section: str, field: str) -> bool:
    model = Settings.model_fields.get(section)
    if model is None or model.annotation is None:
        return False
    sub = typing.cast(type[BaseModel], model.annotation)
    info = sub.model_fields.get(field)
    return info is not None and typing.get_origin(info.annotation) is list


def _coerce_env(section: str, field: str, raw: str) -> Any:
    text = raw.strip()
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid JSON in environment override for {section}.{field}"
            ) from exc
    if _is_list_field(section, field):
        return [p.strip() for p in text.split(",") if p.strip()]
    return text


def env_overrides(env: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        rest = name[len(ENV_PREFIX) :].lower()
        if rest == "api_keys":
            out.setdefault("auth", {})["api_keys"] = raw
            continue
        if rest == "webhook_secret":
            out.setdefault("webhooks", {})["secret"] = raw
            continue
        if "__" not in rest:
            continue
        section, field = rest.split("__", 1)
        if section not in Settings.model_fields:
            raise ConfigError(f"unknown configuration section in {name}")
        out.setdefault(section, {})[field] = _coerce_env(section, field, raw)
    return out


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_path: str | Path | None = None, env: Mapping[str, str] | None = None
) -> Settings:
    """Load and validate settings; raises :class:`ConfigError` on any problem."""
    env = os.environ if env is None else env
    data: dict[str, Any] = {}
    path = config_path or env.get("VAP_CONFIG")
    if path:
        p = Path(path)
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {p}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {p}: {exc}") from exc
    data = _merge(data, env_overrides(env))
    # Servers loaded from configuration are secure by default: authentication is on
    # unless explicitly disabled. (``Settings()`` built in code defaults to off for
    # embedding and tests.)
    data.setdefault("auth", {}).setdefault("enabled", True)
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        problems = [
            {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]} for e in exc.errors()
        ]
        raise ConfigError("invalid configuration", details={"errors": problems}) from exc
