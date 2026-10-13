"""Event rules: zone intrusion, line crossing, loitering and crowding."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from vap.analytics.engine import AnalyticsEngine, Observation
from vap.events.model import Event, EventType, Severity

_RULE_ID = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"


class _RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(pattern=_RULE_ID)
    severity: Severity = Severity.WARNING
    labels: list[str] = Field(default_factory=list, max_length=32)


class ZoneIntrusionRuleSpec(_RuleSpec):
    kind: Literal["zone_intrusion"] = "zone_intrusion"
    zone_id: str
    cooldown_s: float = Field(0.0, ge=0, le=86_400)


class LineCrossingRuleSpec(_RuleSpec):
    kind: Literal["line_crossing"] = "line_crossing"
    line_id: str
    direction: Literal["any", "positive", "negative"] = "any"


class LoiteringRuleSpec(_RuleSpec):
    kind: Literal["loitering"] = "loitering"
    zone_id: str
    min_dwell_s: float = Field(gt=0, le=86_400)


class CrowdingRuleSpec(_RuleSpec):
    kind: Literal["crowding"] = "crowding"
    zone_id: str | None = None
    min_count: int = Field(ge=1, le=10_000)
    min_duration_s: float = Field(0.0, ge=0, le=86_400)
    cooldown_s: float = Field(30.0, ge=0, le=86_400)


AnyRuleSpec = ZoneIntrusionRuleSpec | LineCrossingRuleSpec | LoiteringRuleSpec | CrowdingRuleSpec
RuleSpec = Annotated[AnyRuleSpec, Field(discriminator="kind")]


@dataclass(slots=True)
class RuleContext:
    pipeline_id: str
    camera_id: str
    frame_index: int
    timestamp: float
    observations: list[Observation]
    engine: AnalyticsEngine


class Rule(abc.ABC):
    def __init__(self, spec: AnyRuleSpec) -> None:
        self.spec = spec
        self._labels = set(spec.labels)

    def label_ok(self, label: str) -> bool:
        return not self._labels or label in self._labels

    def event(self, ctx: RuleContext, etype: EventType, message: str, **kwargs: object) -> Event:
        return Event(
            type=etype,
            pipeline_id=ctx.pipeline_id,
            camera_id=ctx.camera_id,
            frame_index=ctx.frame_index,
            stream_time=ctx.timestamp,
            message=message,
            severity=self.spec.severity,
            rule_id=self.spec.rule_id,
            **kwargs,  # type: ignore[arg-type]
        )

    @abc.abstractmethod
    def evaluate(self, ctx: RuleContext) -> list[Event]: ...

    def reset(self) -> None:
        return None


class ZoneIntrusionRule(Rule):
    spec: ZoneIntrusionRuleSpec

    def __init__(self, spec: ZoneIntrusionRuleSpec) -> None:
        super().__init__(spec)
        self._last: dict[int, float] = {}

    def reset(self) -> None:
        self._last.clear()

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        events: list[Event] = []
        for ob in ctx.observations:
            if (
                ob.kind != "zone_enter"
                or ob.zone_id != self.spec.zone_id
                or not self.label_ok(ob.label)
            ):
                continue
            last = self._last.get(ob.track_id)
            if last is not None and ob.timestamp - last < self.spec.cooldown_s:
                continue
            self._last[ob.track_id] = ob.timestamp
            events.append(
                self.event(
                    ctx,
                    EventType.ZONE_INTRUSION,
                    f"track {ob.track_id} ({ob.label}) entered zone {ob.zone_id}",
                    track_ids=[ob.track_id],
                    zone_id=ob.zone_id,
                    data={"label": ob.label},
                )
            )
        return events


class LineCrossingRule(Rule):
    spec: LineCrossingRuleSpec

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        wanted = {"any": (1, -1), "positive": (1,), "negative": (-1,)}[self.spec.direction]
        events: list[Event] = []
        for ob in ctx.observations:
            if ob.kind != "line_cross" or ob.line_id != self.spec.line_id:
                continue
            if ob.direction not in wanted or not self.label_ok(ob.label):
                continue
            direction = "positive" if ob.direction > 0 else "negative"
            events.append(
                self.event(
                    ctx,
                    EventType.LINE_CROSSING,
                    f"track {ob.track_id} ({ob.label}) crossed line {ob.line_id} ({direction})",
                    track_ids=[ob.track_id],
                    line_id=ob.line_id,
                    data={"label": ob.label, "direction": direction},
                )
            )
        return events


class LoiteringRule(Rule):
    """Fires once per zone visit when a track's dwell exceeds ``min_dwell_s``."""

    spec: LoiteringRuleSpec

    def __init__(self, spec: LoiteringRuleSpec) -> None:
        super().__init__(spec)
        self._fired: set[tuple[int, float]] = set()

    def reset(self) -> None:
        self._fired.clear()

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        occupants = ctx.engine.occupants(self.spec.zone_id)
        events: list[Event] = []
        for tid, entered in occupants.items():
            visit = (tid, entered)
            dwell = ctx.timestamp - entered
            if visit in self._fired or dwell < self.spec.min_dwell_s:
                continue
            label = ctx.engine.label_of(tid)
            if not self.label_ok(label):
                continue
            self._fired.add(visit)
            events.append(
                self.event(
                    ctx,
                    EventType.LOITERING,
                    f"track {tid} ({label}) loitering in zone {self.spec.zone_id} for {dwell:.1f}s",
                    track_ids=[tid],
                    zone_id=self.spec.zone_id,
                    data={"label": label, "dwell_s": round(dwell, 3)},
                )
            )
        # Forget visits that ended so memory stays bounded.
        self._fired = {v for v in self._fired if occupants.get(v[0]) == v[1]}
        return events


class CrowdingRule(Rule):
    """Fires when at least ``min_count`` objects are present for ``min_duration_s``.

    The rule re-arms when the count drops below the threshold or after
    ``cooldown_s`` of continuous crowding.
    """

    spec: CrowdingRuleSpec

    def __init__(self, spec: CrowdingRuleSpec) -> None:
        super().__init__(spec)
        self._since: float | None = None
        self._last_fired: float | None = None

    def reset(self) -> None:
        self._since = None
        self._last_fired = None

    def _present(self, ctx: RuleContext) -> list[int]:
        if self.spec.zone_id is None:
            ids = ctx.engine.recently_seen(ctx.timestamp)
        else:
            ids = sorted(ctx.engine.occupants(self.spec.zone_id))
        return [t for t in ids if self.label_ok(ctx.engine.label_of(t))]

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        present = self._present(ctx)
        if len(present) < self.spec.min_count:
            self._since = None
            self._last_fired = None
            return []
        if self._since is None:
            self._since = ctx.timestamp
        held = ctx.timestamp - self._since
        if held < self.spec.min_duration_s:
            return []
        if self._last_fired is not None and ctx.timestamp - self._last_fired < self.spec.cooldown_s:
            return []
        if self._last_fired is not None and self.spec.cooldown_s == 0:
            return []
        self._last_fired = ctx.timestamp
        where = f"zone {self.spec.zone_id}" if self.spec.zone_id else "frame"
        return [
            self.event(
                ctx,
                EventType.CROWDING,
                f"{len(present)} objects in {where} for {held:.1f}s (threshold {self.spec.min_count})",
                track_ids=present,
                zone_id=self.spec.zone_id,
                data={"count": len(present), "held_s": round(held, 3)},
            )
        ]


def build_rule(spec: AnyRuleSpec) -> Rule:
    if isinstance(spec, ZoneIntrusionRuleSpec):
        return ZoneIntrusionRule(spec)
    if isinstance(spec, LineCrossingRuleSpec):
        return LineCrossingRule(spec)
    if isinstance(spec, LoiteringRuleSpec):
        return LoiteringRule(spec)
    return CrowdingRule(spec)


class RuleEngine:
    def __init__(self, specs: list[AnyRuleSpec]) -> None:
        self.rules = [build_rule(s) for s in specs]

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        events: list[Event] = []
        for rule in self.rules:
            events.extend(rule.evaluate(ctx))
        return events
