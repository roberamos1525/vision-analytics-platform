"""The event record emitted by analytics rules."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class EventType(enum.StrEnum):
    ZONE_INTRUSION = "zone_intrusion"
    LINE_CROSSING = "line_crossing"
    LOITERING = "loitering"
    CROWDING = "crowding"
    SOURCE_DOWN = "source_down"


class Severity(enum.StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(slots=True)
class Event:
    type: EventType
    pipeline_id: str
    camera_id: str
    frame_index: int
    stream_time: float
    message: str
    severity: Severity = Severity.WARNING
    rule_id: str = ""
    track_ids: list[int] = field(default_factory=list)
    zone_id: str | None = None
    line_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    snapshot: str | None = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "pipeline_id": self.pipeline_id,
            "camera_id": self.camera_id,
            "rule_id": self.rule_id,
            "frame_index": self.frame_index,
            "stream_time": round(self.stream_time, 4),
            "created_at": round(self.created_at, 3),
            "message": self.message,
            "track_ids": list(self.track_ids),
            "zone_id": self.zone_id,
            "line_id": self.line_id,
            "data": dict(self.data),
            "snapshot": self.snapshot,
        }
