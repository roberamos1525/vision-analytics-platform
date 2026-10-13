from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from vap.config import Role, Settings, env_overrides, load_settings
from vap.errors import ConfigError, InvalidInputError, VapError, validation_details
from vap.observability.logs import JsonFormatter, TextFormatter, configure_logging, get_logger
from vap.observability.metrics import Counter, Gauge, Histogram, MetricsRegistry

KEY = "k" * 20


class TestConfig:
    def test_defaults_require_keys_when_loaded(self) -> None:
        with pytest.raises(ConfigError) as exc:
            load_settings(env={})
        assert "API keys" in json.dumps(exc.value.details)
        s = load_settings(env={"VAP_AUTH__ENABLED": "false"})
        assert not s.auth.enabled and s.server.port == 8080
        assert not Settings().auth.enabled  # programmatic default for embedding/tests

    def test_toml_file_and_env_precedence(self, tmp_path: Path) -> None:
        cfg = tmp_path / "vap.toml"
        cfg.write_text('[server]\nport = 9000\nhost = "0.0.0.0"\n[pipelines]\nmax_pipelines = 3\n')
        env = {
            "VAP_CONFIG": str(cfg),
            "VAP_SERVER__PORT": "9100",
            "VAP_API_KEYS": f"{KEY}:admin, {'v' * 20}:viewer",
            "VAP_WEBHOOK_SECRET": "shh",
            "VAP_WEBHOOKS__URLS": "https://a.example/x,https://b.example/y",
            "VAP_MEDIA__ROOTS": '["/data/a"]',
            "OTHER": "ignored",
        }
        s = load_settings(env=env)
        assert s.server.port == 9100 and s.server.host == "0.0.0.0"
        assert s.pipelines.max_pipelines == 3
        assert s.auth.api_keys[KEY] == Role.ADMIN and len(s.auth.api_keys) == 2
        assert s.webhooks.secret == "shh" and len(s.webhooks.urls) == 2
        assert s.media.roots == [Path("/data/a")]

    @pytest.mark.parametrize(
        "env",
        [
            {"VAP_AUTH__ENABLED": "false", "VAP_SERVER__PORT": "99999"},
            {"VAP_AUTH__ENABLED": "false", "VAP_NOPE__X": "1"},
            {"VAP_API_KEYS": "short:admin"},
            {"VAP_API_KEYS": f"{KEY}:superuser"},
            {"VAP_API_KEYS": "missing-colon-entry-xxxxxxxx"},
            {"VAP_AUTH__ENABLED": "false", "VAP_SERVER__EXTRA": "1"},
            {"VAP_AUTH__ENABLED": "false", "VAP_MEDIA__ROOTS": "[broken"},
        ],
    )
    def test_invalid_configuration(self, env: dict[str, str]) -> None:
        with pytest.raises(ConfigError):
            load_settings(env=env)

    def test_bad_files(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_settings(tmp_path / "missing.toml", env={})
        bad = tmp_path / "bad.toml"
        bad.write_text("[server\n")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_settings(bad, env={})

    def test_roles_and_paths(self, tmp_path: Path) -> None:
        assert Role.ADMIN.allows(Role.VIEWER) and not Role.VIEWER.allows(Role.OPERATOR)
        s = Settings.model_validate({"storage": {"state_dir": str(tmp_path / "st")}})
        s.ensure_dirs()
        assert s.storage.snapshots_dir.is_dir() and s.storage.database_path.name == "vap.db"
        assert env_overrides({"VAP_FOO": "x"}) == {}


class TestErrors:
    def test_envelope_and_validation_details(self) -> None:
        err = InvalidInputError("bad", details={"field": "x"})
        assert err.to_dict() == {
            "code": "invalid_input",
            "message": "bad",
            "details": {"field": "x"},
        }
        assert VapError("x").status_code == 500
        details = validation_details(
            [{"loc": ("body", "a", 0), "msg": "nope", "type": "t", "ctx": {"e": object()}}]
        )
        assert details == [{"loc": "body.a.0", "msg": "nope", "type": "t"}]


class TestMetrics:
    def test_exposition_format(self) -> None:
        reg = MetricsRegistry("t")
        c = reg.counter("requests_total", "Requests", ["route"])
        c.inc(route="/a")
        c.inc(2, route='/b"x')
        g = reg.gauge("temp", "Temperature")
        g.set(1.5)
        g.dec(0.5)
        h = reg.histogram("latency_seconds", "Latency", buckets=(0.1, 1.0))
        h.observe(0.05)
        h.observe(0.5)
        h.observe(5)
        text = reg.render()
        assert "# TYPE t_requests_total counter" in text
        assert 't_requests_total{route="/b\\"x"} 2' in text
        assert "t_temp 1" in text
        assert 't_latency_seconds_bucket{le="0.1"} 1' in text
        assert 't_latency_seconds_bucket{le="1"} 2' in text
        assert 't_latency_seconds_bucket{le="+Inf"} 3' in text
        assert "t_latency_seconds_count 3" in text
        assert h.snapshot()["count"] == 3
        c.remove(route="/a")
        assert c.value(route="/a") == 0

    def test_validation_and_reregistration(self) -> None:
        reg = MetricsRegistry()
        a = reg.counter("x_total", "x", ["a"])
        assert reg.counter("x_total", "x", ["a"]) is a
        with pytest.raises(InvalidInputError):
            reg.gauge("x_total", "x", ["a"])
        with pytest.raises(InvalidInputError):
            a.inc(-1, a="1")
        with pytest.raises(InvalidInputError):
            a.inc(a="1", b="2")
        with pytest.raises(InvalidInputError):
            Counter("bad name", "x")
        with pytest.raises(InvalidInputError):
            Gauge("ok", "x", ["__reserved"])
        with pytest.raises(InvalidInputError):
            Histogram("h", "x", buckets=(0,))
        with pytest.raises(InvalidInputError):
            MetricsRegistry("bad-ns")


class TestLogging:
    def test_json_formatter_includes_extras(self) -> None:
        record = logging.LogRecord("vap.x", logging.INFO, "f", 1, "hello %s", ("world",), None)
        record.pipeline = "p1"
        payload = json.loads(JsonFormatter().format(record))
        assert (
            payload["msg"] == "hello world"
            and payload["pipeline"] == "p1"
            and payload["level"] == "info"
        )
        assert "pipeline=p1" in TextFormatter().format(record)

    def test_configure_and_get_logger(self) -> None:
        configure_logging("debug", "text")
        assert get_logger("thing").name == "vap.thing"
        assert logging.getLogger("vap").level == logging.DEBUG
        configure_logging("info", "json")
