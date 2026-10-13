"""Pipeline configuration (validated with pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vap.analytics.rules import (
    CrowdingRuleSpec,
    LineCrossingRuleSpec,
    LoiteringRuleSpec,
    RuleSpec,
    ZoneIntrusionRuleSpec,
)
from vap.analytics.zones import LineSpec, ZoneSpec
from vap.detection.specs import BackgroundDetectorSpec, DetectorSpec
from vap.tracking.trackers import TrackerSpec

ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector: DetectorSpec = Field(default_factory=BackgroundDetectorSpec)
    tracker: TrackerSpec = Field(default_factory=TrackerSpec)
    zones: list[ZoneSpec] = Field(default_factory=list, max_length=64)
    lines: list[LineSpec] = Field(default_factory=list, max_length=64)
    rules: list[RuleSpec] = Field(default_factory=list, max_length=128)
    process_every_n: int = Field(1, ge=1, le=100)
    resize_width: int | None = Field(None, ge=32, le=4096)
    realtime: bool = True
    max_frames: int | None = Field(None, ge=1)
    snapshots: bool = True
    heatmap: bool = True
    heatmap_decay: float = Field(0.998, gt=0, le=1)
    flow_every_n: int = Field(0, ge=0, le=1000)
    pixels_per_meter: float | None = Field(None, gt=0)
    counter_window_s: float = Field(60.0, gt=0, le=86_400)
    persist_tracks: bool = True

    @model_validator(mode="after")
    def _references(self) -> PipelineConfig:
        zone_ids = [z.zone_id for z in self.zones]
        line_ids = [ln.line_id for ln in self.lines]
        rule_ids = [r.rule_id for r in self.rules]
        for kind, ids in (("zone", zone_ids), ("line", line_ids), ("rule", rule_ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"duplicate {kind} ids: {', '.join(dupes)}")
        zones, lines = set(zone_ids), set(line_ids)
        for rule in self.rules:
            if (
                isinstance(rule, ZoneIntrusionRuleSpec | LoiteringRuleSpec)
                and rule.zone_id not in zones
            ):
                raise ValueError(f"rule {rule.rule_id} references unknown zone {rule.zone_id!r}")
            if (
                isinstance(rule, CrowdingRuleSpec)
                and rule.zone_id is not None
                and rule.zone_id not in zones
            ):
                raise ValueError(f"rule {rule.rule_id} references unknown zone {rule.zone_id!r}")
            if isinstance(rule, LineCrossingRuleSpec) and rule.line_id not in lines:
                raise ValueError(f"rule {rule.rule_id} references unknown line {rule.line_id!r}")
        return self
