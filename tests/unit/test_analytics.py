from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from vap.analytics.counters import DwellStats, LineCounters, RollingCounter
from vap.analytics.engine import AnalyticsEngine
from vap.analytics.rules import RuleContext, RuleEngine, RuleSpec
from vap.analytics.zones import LineSpec, ZoneSpec, resolve_line, resolve_zone
from vap.errors import InvalidInputError
from vap.events.model import EventType
from vap.geometry import BBox
from vap.tracking.base import Track, TrackPoint, TrackState

RULES: TypeAdapter[Any] = TypeAdapter(RuleSpec)

LEFT_ZONE = ZoneSpec(zone_id="left", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
MID_LINE = LineSpec(line_id="mid", points=[(150, 0), (150, 100)])


def make_track(tid: int, cx: float, bottom: float, ts: float, label: str = "person") -> Track:
    box = BBox(cx - 5, bottom - 10, cx + 5, bottom)
    t = Track(tid, box, label, 0.9, 0, ts, 0, ts, state=TrackState.CONFIRMED, ever_confirmed=True)
    t.history.append(TrackPoint(0, ts, cx, bottom - 5))
    return t


class Driver:
    """Feeds tracks through the engine and rules, mimicking the pipeline."""

    def __init__(self, rules: list[dict[str, Any]], zones: list[ZoneSpec] | None = None) -> None:
        self.engine = AnalyticsEngine(
            zones if zones is not None else [LEFT_ZONE], [MID_LINE], window_s=10
        )
        self.engine.configure(200, 100)
        self.rules = RuleEngine([RULES.validate_python(r) for r in rules])
        self.frame = 0

    def step(
        self,
        ts: float,
        positions: dict[int, float],
        finished: list[int] | None = None,
        label: str = "person",
    ) -> list[Any]:
        tracks = [make_track(tid, x, 50, ts, label) for tid, x in positions.items()]
        done = [make_track(tid, 0, 50, ts) for tid in finished or []]
        obs = self.engine.update(tracks, self.frame, ts, done)
        ctx = RuleContext("p", "c", self.frame, ts, obs, self.engine)
        self.frame += 1
        return self.rules.evaluate(ctx)


class TestEngine:
    def test_zone_entry_exit_and_dwell(self) -> None:
        d = Driver([])
        d.step(0.0, {1: 150})
        d.step(1.0, {1: 50})
        assert d.engine.occupants("left") == {1: 1.0}
        d.step(4.0, {1: 50})
        d.step(5.0, {1: 180})
        snap = d.engine.snapshot()["zones"][0]
        assert snap["entries"] == 1 and snap["exits"] == 1 and snap["occupancy"] == 0
        assert snap["dwell"]["max_s"] == pytest.approx(4.0)

    def test_finished_track_exits_zone_at_last_seen_time(self) -> None:
        d = Driver([])
        d.step(0.0, {7: 50})
        d.step(2.0, {7: 60})
        d.step(3.0, {}, finished=[7])
        z = d.engine.zone_counters["left"]
        assert z.exits == 1 and z.dwell.max_s == pytest.approx(2.0)
        assert d.engine.live_track_ids() == []

    def test_line_crossing_both_directions(self) -> None:
        d = Driver([])
        d.step(0.0, {1: 140, 2: 160})
        d.step(1.0, {1: 150, 2: 150})  # on the line: ignored
        d.step(2.0, {1: 165, 2: 120})
        line = d.engine.snapshot()["lines"][0]
        assert line["positive"] == 1 and line["negative"] == 1 and line["net"] == 0

    def test_requires_configure_and_normalized_geometry(self) -> None:
        engine = AnalyticsEngine([LEFT_ZONE])
        with pytest.raises(RuntimeError):
            engine.update([], 0, 0.0)
        norm = ZoneSpec(zone_id="z", points=[(0, 0), (0.5, 0), (0.5, 1), (0, 1)], normalized=True)
        zone = resolve_zone(norm, 200, 100)
        assert zone.contains((99, 50)) and not zone.contains((101, 50))
        line = resolve_line(
            LineSpec(line_id="l", points=[(0, 0.5), (1, 0.5)], normalized=True), 200, 100
        )
        assert line.segment.end == (200, 50)


class TestRules:
    def test_zone_intrusion_with_label_filter_and_cooldown(self) -> None:
        d = Driver(
            [
                {
                    "kind": "zone_intrusion",
                    "rule_id": "r",
                    "zone_id": "left",
                    "labels": ["person"],
                    "cooldown_s": 10,
                }
            ]
        )
        assert d.step(0.0, {1: 150}) == []
        events = d.step(1.0, {1: 50})
        assert [e.type for e in events] == [EventType.ZONE_INTRUSION]
        assert events[0].track_ids == [1] and events[0].zone_id == "left"
        d.step(2.0, {1: 150})
        assert d.step(3.0, {1: 50}) == []  # cooldown
        assert d.step(4.0, {2: 50}, label="car") == []  # label filtered

    def test_line_crossing_direction_filter(self) -> None:
        d = Driver(
            [{"kind": "line_crossing", "rule_id": "r", "line_id": "mid", "direction": "positive"}]
        )
        d.step(0.0, {1: 140, 2: 160})
        events = d.step(1.0, {1: 160, 2: 140})
        assert len(events) == 1 and events[0].data["direction"] == "positive"
        assert events[0].track_ids == [2]

    def test_loitering_fires_once_per_visit(self) -> None:
        d = Driver(
            [
                {
                    "kind": "loitering",
                    "rule_id": "r",
                    "zone_id": "left",
                    "min_dwell_s": 2.0,
                    "severity": "critical",
                }
            ]
        )
        fired = [len(d.step(t, {1: 50})) for t in (0.0, 1.0, 2.5, 3.0)]
        assert fired == [0, 0, 1, 0]
        d.step(4.0, {1: 150})
        fired_again = [len(d.step(t, {1: 50})) for t in (5.0, 7.5)]
        assert fired_again == [0, 1]

    def test_crowding_duration_cooldown_and_rearm(self) -> None:
        d = Driver(
            [
                {
                    "kind": "crowding",
                    "rule_id": "r",
                    "min_count": 2,
                    "min_duration_s": 1.0,
                    "cooldown_s": 5.0,
                }
            ]
        )
        assert d.step(0.0, {1: 50, 2: 60}) == []
        events = d.step(1.0, {1: 50, 2: 60})
        assert len(events) == 1 and events[0].data["count"] == 2
        assert d.step(3.0, {1: 50, 2: 60}) == []
        assert len(d.step(6.0, {1: 50, 2: 60})) == 1
        d.step(7.0, {1: 50}, finished=[2])
        assert d.step(7.5, {1: 50, 3: 70}) == []  # re-armed, waits min_duration again
        assert len(d.step(8.6, {1: 50, 3: 70})) == 1

    def test_crowding_in_zone(self) -> None:
        d = Driver(
            [
                {
                    "kind": "crowding",
                    "rule_id": "r",
                    "zone_id": "left",
                    "min_count": 2,
                    "cooldown_s": 0,
                }
            ]
        )
        assert d.step(0.0, {1: 50, 2: 160}) == []
        assert len(d.step(1.0, {1: 50, 2: 60})) == 1
        assert d.step(2.0, {1: 50, 2: 60}) == []  # cooldown 0 = once per episode

    @pytest.mark.parametrize(
        "rule",
        [
            {"kind": "loitering", "rule_id": "r", "zone_id": "z"},
            {"kind": "crowding", "rule_id": "bad id!", "min_count": 1},
            {"kind": "teleport", "rule_id": "r"},
        ],
    )
    def test_spec_validation(self, rule: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="validation"):
            RULES.validate_python(rule)


class TestZoneSpecs:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="normalized"):
            ZoneSpec(zone_id="z", points=[(0, 0), (2, 0), (0, 1)], normalized=True)
        with pytest.raises(ValueError, match="degenerate"):
            ZoneSpec(zone_id="z", points=[(0, 0), (1, 1), (2, 2)])
        with pytest.raises(ValueError, match="differ"):
            LineSpec(line_id="l", points=[(1, 1), (1, 1)])


class TestCounters:
    def test_rolling_window_expiry(self) -> None:
        c = RollingCounter(window_s=10, bucket_s=1)
        for t in (0.0, 0.5, 3.0, 9.0):
            c.add(t)
        assert c.count(9.5) == 4
        assert c.count(12.0) == 2
        assert c.count(30.0) == 0
        assert c.total == 4
        c.add(29.0)
        c.add(28.0)  # late event folds into the newest bucket
        assert c.count(29.5) == 2 and c.rate_per_minute(29.5) == pytest.approx(12.0)
        with pytest.raises(InvalidInputError):
            RollingCounter(window_s=1, bucket_s=2)

    def test_dwell_stats_and_line_counters(self) -> None:
        s = DwellStats()
        for v in (1.0, 2.0, 3.0, -1.0):
            s.add(v)
        assert s.count == 4 and s.max_s == 3.0 and s.mean_s == pytest.approx(1.5)
        assert s.to_dict()["p50_s"] == pytest.approx(1.5)
        assert DwellStats().quantile(0.5) == 0.0
        lc = LineCounters("l")
        lc.record(1, 0.0)
        lc.record(-1, 1.0)
        lc.record(0, 2.0)
        assert lc.snapshot(2.0)["total"] == 2
