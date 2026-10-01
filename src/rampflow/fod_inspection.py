from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping

from .timeutil import utc


class SegmentState(str, Enum):
    OPEN = "open"
    CONTROLLED = "controlled"


class CheckResult(str, Enum):
    CLEAR = "clear"
    ATTENTION = "attention"


class FodEventType(str, Enum):
    CHECK = "check"
    FIND = "find"
    CLEAR = "clear"


@dataclass(frozen=True)
class InspectionSegment:
    segment_id: str
    route_id: str
    position: int
    name: str = ""


@dataclass(frozen=True)
class InspectionRoute:
    route_id: str
    segments: tuple[InspectionSegment, ...]

    def __post_init__(self):
        if not self.route_id:
            raise ValueError("route id required")
        if not self.segments:
            raise ValueError("route needs at least one segment")
        ids = [s.segment_id for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate segment in route")
        routes = {s.route_id for s in self.segments}
        if routes != {self.route_id}:
            raise ValueError("segment belongs to a different route")
        positions = sorted(s.position for s in self.segments)
        if positions != list(range(len(self.segments))):
            raise ValueError("segment positions must be a dense zero-based order")
        object.__setattr__(self, "segments", tuple(sorted(self.segments, key=lambda s: s.position)))


@dataclass(frozen=True)
class InspectionRound:
    round_id: str
    inspector_id: str
    vehicle_id: str
    started_at: datetime

    def __post_init__(self):
        if not self.round_id or not self.inspector_id or not self.vehicle_id:
            raise ValueError("round, inspector and vehicle are required")
        object.__setattr__(self, "started_at", utc(self.started_at))


@dataclass(frozen=True)
class SegmentCheck:
    event_id: str
    segment_id: str
    round_id: str
    inspector_id: str
    vehicle_id: str
    at: datetime
    result: CheckResult
    note: str = ""


@dataclass(frozen=True)
class FodFinding:
    fod_id: str
    event_id: str
    round_id: str
    inspector_id: str
    vehicle_id: str
    at: datetime
    description: str = ""


@dataclass(frozen=True)
class ClearanceRecord:
    event_id: str
    round_id: str
    inspector_id: str
    vehicle_id: str
    at: datetime
    note: str = ""


@dataclass(frozen=True)
class ContainmentCycle:
    """One controlled interval of a segment.

    Opens when FOD is reported (the segment is locally contained at the same
    instant) and can only close when a different round, driven by a different
    inspector, re-checks the segment and confirms clearance.
    """

    cycle_no: int
    segment_id: str
    finding: FodFinding
    controlled_at: datetime
    extra_findings: tuple[FodFinding, ...] = ()
    clearance: ClearanceRecord | None = None
    version: int = 1

    @property
    def state(self) -> SegmentState:
        return SegmentState.CONTROLLED if self.clearance is None else SegmentState.OPEN

    @property
    def opened(self) -> bool:
        return self.clearance is None

    @property
    def findings(self) -> tuple[FodFinding, ...]:
        return (self.finding, *self.extra_findings)


@dataclass(frozen=True)
class ContainmentOrder:
    """The outward local-control order linked to an active FOD finding."""

    order_id: str
    segment_id: str
    cycle_no: int
    issued_at: datetime
    reason_fod_id: str
    restricted: bool


@dataclass(frozen=True)
class AuditEntry:
    at: datetime
    segment_id: str
    cycle_no: int | None
    code: str
    round_id: str
    inspector_id: str
    vehicle_id: str
    detail: str


@dataclass(frozen=True)
class FodEvent:
    """A raw patrol-vehicle event, possibly delivered long after it happened."""

    event_id: str
    kind: FodEventType
    occurred_at: datetime
    segment_id: str
    round_id: str
    seq: int = 0
    fod_id: str = ""
    result: CheckResult = CheckResult.CLEAR
    note: str = ""
    received_at: datetime | None = None

    def __post_init__(self):
        object.__setattr__(self, "occurred_at", utc(self.occurred_at))
        if self.received_at is not None:
            object.__setattr__(self, "received_at", utc(self.received_at))


@dataclass(frozen=True)
class BackfillOutcome:
    event_id: str
    status: str  # applied | duplicate | rejected
    reason: str = ""
    segment_state: SegmentState | None = None


@dataclass(frozen=True)
class _Effect:
    applied: bool
    reason: str = ""


class FodInspectionService:
    """State machine for segmented FOD patrols with offline-safe replay.

    Every state change is an append-only event keyed by an idempotent event id.
    Segment state is derived by folding each segment's events in their actual
    occurrence order (with the vehicle monotonic ``seq`` breaking ties), so a
    batch uploaded by an offline patrol vehicle cannot reopen or re-control a
    segment that an earlier cycle already closed: stale events land inside the
    closed cycle's timeline and leave the current state untouched.
    """

    def __init__(self):
        self.routes: dict[str, InspectionRoute] = {}
        self._segments: dict[str, InspectionSegment] = {}
        self.rounds: dict[str, InspectionRound] = {}
        self._events: dict[str, list[FodEvent]] = {}
        # derived projections, rebuilt by folding
        self._checks: dict[str, list[SegmentCheck]] = {}
        self._cycles: dict[str, list[ContainmentCycle]] = {}
        self._effects: dict[str, dict[str, _Effect]] = {}
        self._segment_audit: dict[str, list[AuditEntry]] = {}
        self._audit: list[AuditEntry] = []

    # ----- registration -----

    def register_route(self, route: InspectionRoute) -> InspectionRoute:
        if route.route_id in self.routes:
            raise ValueError(f"route {route.route_id} already registered")
        for segment in route.segments:
            if segment.segment_id in self._segments:
                raise ValueError(f"segment {segment.segment_id} already registered")
        self.routes[route.route_id] = route
        for segment in route.segments:
            self._segments[segment.segment_id] = segment
            self._events.setdefault(segment.segment_id, [])
            self._checks.setdefault(segment.segment_id, [])
            self._cycles.setdefault(segment.segment_id, [])
            self._effects.setdefault(segment.segment_id, {})
            self._segment_audit.setdefault(segment.segment_id, [])
        return route

    def register_round(self, round_: InspectionRound) -> InspectionRound:
        if round_.round_id in self.rounds:
            raise ValueError(f"round {round_.round_id} already registered")
        self.rounds[round_.round_id] = round_
        return round_

    def route_segments(self, route_id: str) -> tuple[InspectionSegment, ...]:
        return self.routes[route_id].segments

    def _segment(self, segment_id: str) -> InspectionSegment:
        try:
            return self._segments[segment_id]
        except KeyError:
            raise KeyError(f"unknown segment {segment_id}") from None

    def _round(self, round_id: str) -> InspectionRound:
        try:
            return self.rounds[round_id]
        except KeyError:
            raise KeyError(f"unknown round {round_id}") from None

    # ----- online operations -----

    def record_check(
        self,
        segment_id: str,
        round_id: str,
        at: datetime,
        event_id: str,
        result: CheckResult = CheckResult.CLEAR,
        note: str = "",
        seq: int = 0,
    ) -> SegmentCheck:
        self._accept(
            FodEvent(
                event_id=event_id,
                kind=FodEventType.CHECK,
                occurred_at=at,
                segment_id=segment_id,
                round_id=round_id,
                seq=seq,
                result=result,
                note=note,
                received_at=utc(at),
            )
        )
        return self._find_check(event_id, segment_id)

    def report_finding(
        self,
        segment_id: str,
        round_id: str,
        fod_id: str,
        at: datetime,
        event_id: str,
        description: str = "",
        seq: int = 0,
    ) -> ContainmentCycle:
        event = self._accept(
            FodEvent(
                event_id=event_id,
                kind=FodEventType.FIND,
                occurred_at=at,
                segment_id=segment_id,
                round_id=round_id,
                seq=seq,
                fod_id=fod_id,
                note=description,
                received_at=utc(at),
            )
        )
        effect = self._effect(event)
        if not effect.applied:
            raise RuntimeError(f"finding rejected: {effect.reason}")
        return self._cycles[segment_id][-1]

    def verify_clearance(
        self,
        segment_id: str,
        verifier_round_id: str,
        at: datetime,
        event_id: str,
        note: str = "",
        seq: int = 0,
    ) -> ContainmentCycle:
        event = self._accept(
            FodEvent(
                event_id=event_id,
                kind=FodEventType.CLEAR,
                occurred_at=at,
                segment_id=segment_id,
                round_id=verifier_round_id,
                seq=seq,
                note=note,
                received_at=utc(at),
            )
        )
        effect = self._effect(event)
        if not effect.applied:
            raise RuntimeError(f"clearance rejected: {effect.reason}")
        return self._cycles[segment_id][-1]

    # ----- offline backfill -----

    def ingest_offline(
        self,
        events: Iterable[FodEvent],
        received_at: datetime | None = None,
    ) -> tuple[BackfillOutcome, ...]:
        """Merge a vehicle's offline batch, replaying by occurrence time.

        Events may arrive in any upload order.  They are validated against the
        registered routes/rounds, deduplicated by event id, then every affected
        segment is re-folded chronologically.  Returns one outcome per event in
        the order the caller supplied them.
        """
        received = utc(received_at) if received_at is not None else None
        batch = tuple(events)
        outcomes: dict[str, BackfillOutcome] = {}
        touched: set[str] = set()
        known_ids = {e.event_id for evs in self._events.values() for e in evs}

        for event in batch:
            if event.event_id in outcomes:
                raise ValueError(f"duplicate event id within batch: {event.event_id}")
            stamped = event if received is None or event.received_at is not None else self._stamp(event, received)
            reason = self._validate(stamped)
            if reason:
                outcomes[stamped.event_id] = BackfillOutcome(stamped.event_id, "rejected", reason)
                continue
            if stamped.event_id in known_ids:
                outcomes[stamped.event_id] = BackfillOutcome(
                    stamped.event_id, "duplicate", "event already recorded", self.state(stamped.segment_id)
                )
                continue
            self._events[stamped.segment_id].append(stamped)
            known_ids.add(stamped.event_id)
            touched.add(stamped.segment_id)
            outcomes[stamped.event_id] = BackfillOutcome(stamped.event_id, "applied")

        for segment_id in touched:
            self._refold(segment_id)
        # annotate applied outcomes with the post-replay state and flag events
        # that were well-formed but produced no effect on a finished cycle
        for event_id, outcome in list(outcomes.items()):
            if outcome.status != "applied":
                continue
            event = self._find_event(event_id)
            effect = self._effect(event)
            state = self.state(event.segment_id)
            if effect.applied:
                outcomes[event_id] = BackfillOutcome(event_id, "applied", "", state)
            else:
                outcomes[event_id] = BackfillOutcome(event_id, "applied", f"stale: {effect.reason}", state)
        return tuple(outcomes[e.event_id] for e in batch)

    @staticmethod
    def _stamp(event: FodEvent, received_at: datetime) -> FodEvent:
        return FodEvent(
            event_id=event.event_id,
            kind=event.kind,
            occurred_at=event.occurred_at,
            segment_id=event.segment_id,
            round_id=event.round_id,
            seq=event.seq,
            fod_id=event.fod_id,
            result=event.result,
            note=event.note,
            received_at=received_at,
        )

    # ----- queries -----

    def state(self, segment_id: str) -> SegmentState:
        self._segment(segment_id)
        cycle = self._cycles[segment_id][-1:]
        return cycle[0].state if cycle and cycle[0].opened else SegmentState.OPEN

    def controlled_segments(self) -> tuple[ContainmentOrder, ...]:
        orders = []
        for segment_id in sorted(self._segments):
            cycles = self._cycles.get(segment_id, ())
            if cycles and cycles[-1].opened:
                cycle = cycles[-1]
                orders.append(
                    ContainmentOrder(
                        order_id=f"CTRL-{segment_id}-{cycle.cycle_no}",
                        segment_id=segment_id,
                        cycle_no=cycle.cycle_no,
                        issued_at=cycle.controlled_at,
                        reason_fod_id=cycle.finding.fod_id,
                        restricted=True,
                    )
                )
        return tuple(orders)

    def checks(self, segment_id: str) -> tuple[SegmentCheck, ...]:
        self._segment(segment_id)
        return tuple(self._checks.get(segment_id, ()))

    def cycles(self, segment_id: str) -> tuple[ContainmentCycle, ...]:
        self._segment(segment_id)
        return tuple(self._cycles.get(segment_id, ()))

    def audit_trail(self, route_id: str | None = None) -> tuple[AuditEntry, ...]:
        if route_id is None:
            return tuple(self._audit)
        allowed = {s.segment_id for s in self.routes[route_id].segments}
        return tuple(e for e in self._audit if e.segment_id in allowed)

    def segment_coverage(self, route_id: str) -> Mapping[str, tuple[SegmentCheck, ...]]:
        """Per-segment evidence of which round/inspector last covered it."""
        return {s.segment_id: self.checks(s.segment_id) for s in self.routes[route_id].segments}

    # ----- internals -----

    def _validate(self, event: FodEvent) -> str:
        if not event.event_id:
            return "event id required"
        if event.segment_id not in self._segments:
            return f"unknown segment {event.segment_id}"
        if event.round_id not in self.rounds:
            return f"unknown round {event.round_id}"
        if event.kind is FodEventType.FIND and not event.fod_id:
            return "finding requires a fod id"
        return ""

    def _accept(self, event: FodEvent) -> FodEvent:
        reason = self._validate(event)
        if reason:
            raise ValueError(reason)
        existing = self._find_event(event.event_id)
        if existing is not None:
            raise ValueError(f"duplicate event id {event.event_id}")
        self._events[event.segment_id].append(event)
        self._refold(event.segment_id)
        return event

    def _effect(self, event: FodEvent) -> _Effect:
        return self._effects[event.segment_id].get(event.event_id, _Effect(False, "event not found"))

    def _find_event(self, event_id: str) -> FodEvent | None:
        for events in self._events.values():
            for event in events:
                if event.event_id == event_id:
                    return event
        return None

    def _find_check(self, event_id: str, segment_id: str) -> SegmentCheck:
        return next(c for c in self._checks[segment_id] if c.event_id == event_id)

    def _ordered(self, segment_id: str) -> list[FodEvent]:
        return sorted(
            self._events[segment_id],
            key=lambda e: (e.occurred_at, e.seq, e.event_id),
        )

    def _refold(self, segment_id: str) -> None:
        checks: list[SegmentCheck] = []
        cycles: list[ContainmentCycle] = []
        effects: dict[str, _Effect] = {}
        audit: list[AuditEntry] = []
        known_fod: set[str] = set()

        for event in self._ordered(segment_id):
            round_ = self.rounds[event.round_id]
            actor = (round_.round_id, round_.inspector_id, round_.vehicle_id)

            if event.kind is FodEventType.CHECK:
                checks.append(
                    SegmentCheck(
                        event.event_id,
                        segment_id,
                        round_.round_id,
                        round_.inspector_id,
                        round_.vehicle_id,
                        event.occurred_at,
                        event.result,
                        event.note,
                    )
                )
                effects[event.event_id] = _Effect(True)
                audit.append(
                    AuditEntry(
                        event.occurred_at, segment_id, len(cycles) or None,
                        "segment.checked", *actor,
                        f"result={event.result.value}",
                    )
                )
                continue

            if event.kind is FodEventType.FIND:
                if event.fod_id in known_fod:
                    effects[event.event_id] = _Effect(False, "fod already reported in this segment")
                    continue
                known_fod.add(event.fod_id)
                finding = FodFinding(
                    event.fod_id, event.event_id, round_.round_id, round_.inspector_id,
                    round_.vehicle_id, event.occurred_at, event.note,
                )
                active = cycles[-1] if cycles and cycles[-1].opened else None
                if active is None:
                    cycle = ContainmentCycle(
                        cycle_no=len(cycles) + 1,
                        segment_id=segment_id,
                        finding=finding,
                        controlled_at=event.occurred_at,
                    )
                    cycles.append(cycle)
                    audit.append(
                        AuditEntry(
                            event.occurred_at, segment_id, cycle.cycle_no,
                            "fod.reported", *actor, f"fod={event.fod_id}",
                        )
                    )
                    audit.append(
                        AuditEntry(
                            event.occurred_at, segment_id, cycle.cycle_no,
                            "segment.controlled", *actor,
                            f"local containment for fod={event.fod_id}",
                        )
                    )
                else:
                    cycle = ContainmentCycle(
                        cycle_no=active.cycle_no,
                        segment_id=segment_id,
                        finding=active.finding,
                        controlled_at=active.controlled_at,
                        extra_findings=(*active.extra_findings, finding),
                        clearance=active.clearance,
                        version=active.version + 1,
                    )
                    cycles[-1] = cycle
                    audit.append(
                        AuditEntry(
                            event.occurred_at, segment_id, cycle.cycle_no,
                            "fod.reported", *actor, f"fod={event.fod_id} added to active containment",
                        )
                    )
                effects[event.event_id] = _Effect(True)
                continue

            # clearance
            active = cycles[-1] if cycles and cycles[-1].opened else None
            if active is None:
                effects[event.event_id] = _Effect(False, "segment has no open containment")
                continue
            finder = active.finding
            if round_.round_id == finder.round_id:
                effects[event.event_id] = _Effect(False, "clearance must come from another inspection round")
                continue
            if round_.inspector_id == finder.inspector_id:
                effects[event.event_id] = _Effect(False, "clearance inspector must differ from the finder")
                continue
            if event.occurred_at < finder.at:
                effects[event.event_id] = _Effect(False, "clearance cannot predate the finding")
                continue
            clearance = ClearanceRecord(
                event.event_id, round_.round_id, round_.inspector_id,
                round_.vehicle_id, event.occurred_at, event.note,
            )
            cycles[-1] = ContainmentCycle(
                cycle_no=active.cycle_no,
                segment_id=segment_id,
                finding=active.finding,
                controlled_at=active.controlled_at,
                extra_findings=active.extra_findings,
                clearance=clearance,
                version=active.version + 1,
            )
            effects[event.event_id] = _Effect(True)
            audit.append(
                AuditEntry(
                    event.occurred_at, segment_id, active.cycle_no,
                    "segment.reopened", *actor,
                    f"verified clearance of {len(active.findings)} finding(s)",
                )
            )

        self._checks[segment_id] = checks
        self._cycles[segment_id] = cycles
        self._effects[segment_id] = effects
        self._segment_audit[segment_id] = audit
        self._rebuild_audit()

    def _rebuild_audit(self) -> None:
        entries = [entry for trail in self._segment_audit.values() for entry in trail]
        entries.sort(key=lambda e: (e.at, e.segment_id))
        self._audit = entries
