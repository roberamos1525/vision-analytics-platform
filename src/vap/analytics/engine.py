"""Streaming analytics over confirmed tracks: zone membership, dwell, line crossings.

The engine is fed one frame at a time with the tracks seen in that frame and
the tracks the tracker has finished. It maintains per-zone occupancy and
dwell statistics, per-line directional counts, and returns low-level
:class:`Observation` records that the rule engine turns into events.

Anchor point: the bottom-centre of a track's box (where an object touches the
ground), the conventional reference for floor-plane zones and tripwires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vap.analytics.counters import LineCounters, RollingCounter, ZoneCounters
from vap.analytics.zones import LineSpec, Tripwire, Zone, ZoneSpec, resolve_line, resolve_zone
from vap.motion.kinematics import estimate_kinematics
from vap.tracking.base import Track

ObservationKind = Literal["zone_enter", "zone_exit", "line_cross"]


@dataclass(frozen=True, slots=True)
class Observation:
    kind: ObservationKind
    track_id: int
    label: str
    frame_index: int
    timestamp: float
    zone_id: str | None = None
    line_id: str | None = None
    direction: int = 0
    dwell_s: float = 0.0


@dataclass(slots=True)
class _TrackMemo:
    label: str
    first_ts: float
    last_ts: float
    anchor: tuple[float, float]


class AnalyticsEngine:
    def __init__(
        self,
        zones: list[ZoneSpec] | None = None,
        lines: list[LineSpec] | None = None,
        *,
        window_s: float = 60.0,
        pixels_per_meter: float | None = None,
    ) -> None:
        self.zone_specs = list(zones or [])
        self.line_specs = list(lines or [])
        self.window_s = window_s
        self.pixels_per_meter = pixels_per_meter
        self.zones: dict[str, Zone] = {}
        self.lines: dict[str, Tripwire] = {}
        self.zone_counters = {z.zone_id: ZoneCounters(z.zone_id, window_s) for z in self.zone_specs}
        self.line_counters = {
            ln.line_id: LineCounters(ln.line_id, window_s) for ln in self.line_specs
        }
        self._membership: dict[str, dict[int, float]] = {z.zone_id: {} for z in self.zone_specs}
        self._line_memory: dict[tuple[str, int], tuple[int, tuple[float, float]]] = {}
        self._tracks: dict[int, _TrackMemo] = {}
        self._speeds: dict[int, float] = {}
        self.unique_tracks = 0
        self.track_window = RollingCounter(window_s)
        self.frame_size: tuple[int, int] | None = None
        self.last_ts = 0.0

    # -- geometry ---------------------------------------------------------------------

    def configure(self, width: int, height: int) -> None:
        """Resolve (possibly normalised) geometry for the given frame size."""
        if self.frame_size == (width, height):
            return
        self.frame_size = (width, height)
        self.zones = {z.zone_id: resolve_zone(z, width, height) for z in self.zone_specs}
        self.lines = {ln.line_id: resolve_line(ln, width, height) for ln in self.line_specs}

    # -- per-frame update -------------------------------------------------------------

    def update(
        self,
        tracks: list[Track],
        frame_index: int,
        timestamp: float,
        finished: list[Track] | None = None,
    ) -> list[Observation]:
        if self.frame_size is None:
            raise RuntimeError("AnalyticsEngine.configure() must be called before update()")
        self.last_ts = timestamp
        obs: list[Observation] = []
        for track in tracks:
            obs.extend(self._update_track(track, frame_index, timestamp))
        for track in finished or []:
            obs.extend(self._finish_track(track.track_id, frame_index))
        for zone_id, members in self._membership.items():
            self.zone_counters[zone_id].occupancy = len(members)
        return obs

    def _update_track(self, track: Track, frame_index: int, ts: float) -> list[Observation]:
        tid = track.track_id
        anchor = track.anchor
        memo = self._tracks.get(tid)
        if memo is None:
            self.unique_tracks += 1
            self.track_window.add(ts)
            memo = _TrackMemo(track.label, ts, ts, anchor)
            self._tracks[tid] = memo
        memo.last_ts = ts
        memo.anchor = anchor
        kin = estimate_kinematics(track.history, pixels_per_meter=self.pixels_per_meter)
        self._speeds[tid] = kin.speed_px_s
        out: list[Observation] = []
        for zone_id, zone in self.zones.items():
            members = self._membership[zone_id]
            inside = zone.contains(anchor)
            if inside and tid not in members:
                members[tid] = ts
                counters = self.zone_counters[zone_id]
                counters.entries += 1
                counters.entries_window.add(ts)
                counters.peak_occupancy = max(counters.peak_occupancy, len(members))
                out.append(
                    Observation("zone_enter", tid, track.label, frame_index, ts, zone_id=zone_id)
                )
            elif not inside and tid in members:
                out.append(self._exit(zone_id, tid, frame_index, ts))
        for line_id, wire in self.lines.items():
            side = wire.segment.side(anchor)
            if side == 0:
                continue
            key = (line_id, tid)
            prev = self._line_memory.get(key)
            self._line_memory[key] = (side, anchor)
            if prev is None or prev[0] == side:
                continue
            direction = wire.segment.crossing_direction(prev[1], anchor)
            if direction:
                self.line_counters[line_id].record(direction, ts)
                out.append(
                    Observation(
                        "line_cross",
                        tid,
                        track.label,
                        frame_index,
                        ts,
                        line_id=line_id,
                        direction=direction,
                    )
                )
        return out

    def _exit(self, zone_id: str, tid: int, frame_index: int, ts: float) -> Observation:
        entered = self._membership[zone_id].pop(tid)
        counters = self.zone_counters[zone_id]
        counters.exits += 1
        dwell = max(ts - entered, 0.0)
        counters.dwell.add(dwell)
        label = self._tracks[tid].label if tid in self._tracks else "object"
        return Observation("zone_exit", tid, label, frame_index, ts, zone_id=zone_id, dwell_s=dwell)

    def _finish_track(self, tid: int, frame_index: int) -> list[Observation]:
        memo = self._tracks.pop(tid, None)
        self._speeds.pop(tid, None)
        for key in [k for k in self._line_memory if k[1] == tid]:
            del self._line_memory[key]
        if memo is None:
            return []
        return [
            self._exit(zone_id, tid, frame_index, memo.last_ts)
            for zone_id, members in self._membership.items()
            if tid in members
        ]

    # -- queries ------------------------------------------------------------------------

    def occupants(self, zone_id: str) -> dict[int, float]:
        return dict(self._membership.get(zone_id, {}))

    def label_of(self, track_id: int) -> str:
        memo = self._tracks.get(track_id)
        return memo.label if memo else "object"

    def live_track_ids(self) -> list[int]:
        return sorted(self._tracks)

    def recently_seen(self, now: float, max_age_s: float = 1.0) -> list[int]:
        return sorted(t for t, m in self._tracks.items() if now - m.last_ts <= max_age_s)

    def snapshot(self) -> dict[str, Any]:
        now = self.last_ts
        speeds = list(self._speeds.values())
        return {
            "stream_time": round(now, 3),
            "live_tracks": len(self._tracks),
            "unique_tracks": self.unique_tracks,
            "tracks_per_min": round(self.track_window.rate_per_minute(now), 3),
            "mean_speed_px_s": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "zones": [c.snapshot(now) for c in self.zone_counters.values()],
            "lines": [c.snapshot(now) for c in self.line_counters.values()],
        }
