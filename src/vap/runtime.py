"""Application container wiring settings, storage, event bus, registry and pipelines."""

from __future__ import annotations

import time
from typing import Any

import httpx

from vap import __version__
from vap.config import Settings
from vap.events.bus import EventBus
from vap.events.sinks import RecentEventsSink, Resolver, SqliteEventSink, WebhookSink
from vap.imaging.backend import available_backends
from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics
from vap.pipeline.manager import PipelineManager
from vap.registry.models import ModelRegistry
from vap.sources.video_file import CaptureFactory
from vap.storage.database import Database
from vap.storage.repositories import Repositories

log = get_logger("runtime")


class Runtime:
    def __init__(
        self,
        settings: Settings,
        *,
        capture_factory: CaptureFactory | None = None,
        webhook_client: httpx.Client | None = None,
        webhook_resolver: Resolver | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.metrics = PlatformMetrics()
        self.db = Database(settings.storage.database_path)
        self.repos = Repositories(self.db)
        self.registry = ModelRegistry(
            self.db, settings.storage.models_dir, max_bytes=settings.server.max_upload_bytes
        )
        self.bus = EventBus(metrics=self.metrics)
        self.recent = RecentEventsSink(settings.pipelines.recent_events)
        self.bus.subscribe(SqliteEventSink(self.repos.events))
        self.bus.subscribe(self.recent)
        self.webhook: WebhookSink | None = None
        wh = settings.webhooks
        if wh.urls:
            kwargs: dict[str, Any] = {}
            if webhook_resolver is not None:
                kwargs["resolver"] = webhook_resolver
            self.webhook = WebhookSink(
                wh.urls,
                secret=wh.secret,
                timeout_s=wh.timeout_s,
                max_retries=wh.max_retries,
                backoff_s=wh.backoff_s,
                queue_size=wh.queue_size,
                event_types=wh.event_types,
                allow_private=wh.allow_private_networks,
                client=webhook_client,
                metrics=self.metrics,
                **kwargs,
            )
            self.bus.subscribe(self.webhook)
        self.manager = PipelineManager(
            settings,
            self.repos,
            self.bus,
            self.metrics,
            self.registry,
            capture_factory=capture_factory,
        )
        self.started_at = time.time()
        self._started = False

    def start(self, *, restore: bool = True) -> None:
        if self._started:
            return
        self.bus.start()
        cutoff = time.time() - self.settings.storage.event_retention_days * 86_400
        purged = self.repos.events.purge_before(cutoff) + self.repos.metrics.purge_before(cutoff)
        if purged:
            log.info("purged expired rows", extra={"rows": purged})
        if restore:
            restored = self.manager.restore()
            if restored:
                log.info("restored pipelines", extra={"pipelines": restored})
        self._started = True

    def close(self) -> None:
        self.manager.shutdown()
        self.bus.flush(timeout=5.0)
        self.bus.stop()
        self.db.close()
        self._started = False

    def health(self) -> dict[str, Any]:
        db_ok = self.db.ping()
        status = {
            "status": "ok" if db_ok and self.bus.running else "degraded",
            "version": __version__,
            "uptime_s": round(time.time() - self.started_at, 1),
            "database": {"ok": db_ok, "schema_version": self.db.schema_version if db_ok else None},
            "event_bus": self.bus.stats(),
            "pipelines_running": len(self.manager.running_ids()),
            "image_backends": available_backends(),
            "image_backend": self.settings.pipelines.image_backend,
        }
        if self.webhook is not None:
            status["webhooks"] = self.webhook.stats()
        return status
