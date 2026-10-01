from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Iterable, NamedTuple

from .timeutil import utc


class FodInspectionError(RuntimeError):
    """Base class for apron foreign-object-debris patrol failures."""


class UnknownSegment(FodInspectionError, KeyError):
    pass


class DuplicateEvent(FodInspectionError):
    pass


class SegmentClosed(FodInspectionError):
    pass


class NoOpenCase(FodInspectionError):
    pass


class SameRoundReinspection(FodInspectionError):
    """A segment may only be reopened by a patrol round other than the finding one."""


class SameInspectorReinspection(FodInspectionError):
    """The confirming patrol must carry a different inspector than the reporter."""


class StaleEvent(FodInspectionError):
    """An online command arrived older than events already accepted for the segment."""


class CheckOutcome(str, Enum):
    CLEAR = "clear"
    FINDING = "finding"
    CLEARANCE = "clearance"


class SegmentState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class CaseState(str, Enum):
    OPEN = "open"
    CONFIRMED_CLEAR = "confirmed-clear"


@dataclass(frozen=True)
class RouteSegment:
    """One auditable stretch of an inspection route."""

    segment_id: str
    route_id: str = ""
    label: str = ""

    def __post_init__(self):
        if not self.segment_id:
            raise ValueError("segment id required")
        if not self.label:
            object.__setattr__(self, "label", self.segment_id)


@dataclass(frozen=True)
class InspectionEvent:
    """Immutable patrol observation.

    Events are ordered by when the patrol actually happened (``occurred_at``),
    breaking ties with the vehicle-local ``client_seq``, so records captured
    offline and uploaded later are still applied in real-world order.
    """

    event_id: str
    segment_id: str
    round_id: str
    inspector_id: str
    occurred_at: datetime
    outcome: str
    client_seq: int = 0
    vehicle_id: str = ""
    case_id: str = ""
    finding_id: str = ""
    description: str = ""
    note: str = ""

    def __post_init__(self):
        if not self.event_id:
            raise ValueError("event id required")
        if not self.segment_id or not self.round_id or not self.inspector_id:
            raise ValueError("segment, round and inspector are required")
        object.__setattr__(self, "occurred_at", utc(self.occurred_at))
        if self.outcome not in (CheckOutcome.CLEAR, CheckOutcome.FINDING, CheckOutcome.CLEARANCE):
            raise ValueError(f"unknown inspection outcome {self.outcome}")
        if self.client_seq < 0:
            raise ValueError("client sequence must be non-negative")

    @property
    def order_key(self) -> tuple[datetime, int, str]:
        return (self.occurred_at, self.client_seq, self.event_id)

    @classmethod
    def check(
        cls,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        *,
        client_seq: int = 0,
        vehicle_id: str = "",
        note: str = "",
    ) -> "InspectionEvent":
        return cls(
            event_id, segment_id, round_id, inspector_id, occurred_at,
            CheckOutcome.CLEAR, client_seq, vehicle_id, note=note,
        )

    @classmethod
    def finding(
        cls,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        *,
        case_id: str,
        finding_id: str = "",
        description: str = "",
        client_seq: int = 0,
        vehicle_id: str = "",
    ) -> "InspectionEvent":
        if not case_id:
            raise ValueError("finding requires a case id")
        return cls(
            event_id, segment_id, round_id, inspector_id, occurred_at,
            CheckOutcome.FINDING, client_seq, vehicle_id,
            case_id=case_id, finding_id=finding_id or event_id, description=description,
        )

    @classmethod
    def clearance(
        cls,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        case_id: str,
        *,
        client_seq: int = 0,
        vehicle_id: str = "",
        note: str = "",
    ) -> "InspectionEvent":
        if not case_id:
            raise ValueError("clearance requires a case id")
        return cls(
            event_id, segment_id, round_id, inspector_id, occurred_at,
            CheckOutcome.CLEARANCE, client_seq, vehicle_id, case_id=case_id, note=note,
        )


@dataclass(frozen=True)
class FodCase:
    """A foreign-object-debris case that holds the segment closed until re-inspection."""

    case_id: str
    segment_id: str
    reported_by_event: str
    report_round: str
    report_inspector: str
    opened_at: datetime
    state: str
    clearance_event: str = ""
    clearance_round: str = ""
    clearance_inspector: str = ""
    closed_at: datetime | None = None
    observation_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceEntry:
    """One line of the per-segment audit trail: who checked what and when."""

    event: InspectionEvent
    accepted: bool
    effect: str
    detail: str = ""


@dataclass(frozen=True)
class SegmentTrace:
    segment: RouteSegment
    state: str
    entries: tuple[TraceEntry, ...]
    open_case: FodCase | None


@dataclass(frozen=True)
class BackfillReport:
    received: int
    accepted: tuple[InspectionEvent, ...]
    duplicates: tuple[InspectionEvent, ...]
    anomalies: tuple[TraceEntry, ...]

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)


class _Projection(NamedTuple):
    states: dict[str, str]
    traces: dict[str, list[TraceEntry]]
    cases: dict[str, FodCase]
    open_case: dict[str, FodCase]
    latest: dict[str, InspectionEvent]


class FodInspectionLedger:
    """Event-sourced registry of segmented apron FOD patrols.

    Every patrol observation is an immutable event.  Segments are open until a
    finding opens a case that locally closes the segment; the segment reopens
    only when a *different* patrol round (and a different inspector) confirms
    clearance.  Online submissions reject out-of-order commands; offline
    vehicles upload a batch with :meth:`backfill`, which is folded into the
    timeline in actual-occurrence order.  Because state is always projected by
    replaying the ordered timeline, late data extends history rather than
    re-closing segments that an independent round has already settled.
    """

    def __init__(self, segments: Iterable[RouteSegment] = ()):
        self._segments: dict[str, RouteSegment] = {}
        self._events: dict[str, InspectionEvent] = {}
        self._lock = RLock()
        for segment in segments:
            self.add_segment(segment)

    def add_segment(self, segment: RouteSegment) -> None:
        with self._lock:
            if segment.segment_id in self._segments:
                raise DuplicateEvent(f"segment {segment.segment_id} already registered")
            self._segments[segment.segment_id] = segment

    def segments(self) -> tuple[RouteSegment, ...]:
        with self._lock:
            return tuple(self._segments[k] for k in sorted(self._segments))

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def _ordered(self) -> list[InspectionEvent]:
        return sorted(self._events.values(), key=lambda e: e.order_key)

    def _project(self) -> _Projection:
        states: dict[str, str] = {}
        traces: dict[str, list[TraceEntry]] = {}
        cases: dict[str, FodCase] = {}
        open_case: dict[str, FodCase] = {}
        latest: dict[str, InspectionEvent] = {}
        for segment_id in self._segments:
            states[segment_id] = SegmentState.OPEN
            traces[segment_id] = []
        for event in self._ordered():
            latest[event.segment_id] = event
            entries = traces.setdefault(event.segment_id, [])
            state = states.get(event.segment_id, SegmentState.OPEN)
            entry = self._apply(event, state, states, cases, open_case)
            entries.append(entry)
        return _Projection(states, traces, cases, open_case, latest)

    @staticmethod
    def _apply(
        event: InspectionEvent,
        state: str,
        states: dict[str, str],
        cases: dict[str, FodCase],
        open_case: dict[str, FodCase],
    ) -> TraceEntry:
        if event.outcome == CheckOutcome.CLEAR:
            if state == SegmentState.CLOSED:
                return TraceEntry(
                    event, False, "rejected-clear-while-closed",
                    "clear recorded on a closed segment; case still awaits independent reinspection",
                )
            return TraceEntry(event, True, "checked-clear", "segment inspected, no debris found")

        if event.outcome == CheckOutcome.FINDING:
            active = open_case.get(event.segment_id)
            if active is not None:
                if active.case_id == event.case_id:
                    case = cases[event.case_id]
                    updated = FodCase(
                        case_id=case.case_id, segment_id=case.segment_id,
                        reported_by_event=case.reported_by_event,
                        report_round=case.report_round,
                        report_inspector=case.report_inspector,
                        opened_at=case.opened_at, state=case.state,
                        clearance_event=case.clearance_event,
                        clearance_round=case.clearance_round,
                        clearance_inspector=case.clearance_inspector,
                        closed_at=case.closed_at,
                        observation_events=case.observation_events + (event.event_id,),
                    )
                    cases[event.case_id] = updated
                    open_case[event.segment_id] = updated
                    return TraceEntry(event, True, "finding-added-to-open-case", "additional debris observation attached to active case")
                return TraceEntry(
                    event, False, "rejected-finding-while-closed",
                    f"segment already closed under {active.case_id}; clear that case first",
                )
            prior = cases.get(event.case_id)
            if prior is not None:
                # The case was already confirmed clear by another round; a late
                # observation never re-closes a settled segment.
                return TraceEntry(
                    event, False, "late-finding-after-clearance",
                    f"case {event.case_id} was confirmed clear at {prior.closed_at}; open a new case",
                )
            case = FodCase(
                case_id=event.case_id,
                segment_id=event.segment_id,
                reported_by_event=event.event_id,
                report_round=event.round_id,
                report_inspector=event.inspector_id,
                opened_at=event.occurred_at,
                state=CaseState.OPEN,
            )
            cases[event.case_id] = case
            open_case[event.segment_id] = case
            states[event.segment_id] = SegmentState.CLOSED
            return TraceEntry(event, True, "case-opened", f"local closure for case {event.case_id}")

        # clearance / reinspection
        case = open_case.get(event.segment_id)
        if case is None or case.case_id != event.case_id:
            return TraceEntry(
                event, False, "clearance-without-open-case",
                f"no open case {event.case_id} on segment",
            )
        if event.round_id == case.report_round:
            return TraceEntry(
                event, False, "same-round-reinspection",
                "reopening requires a separate patrol round",
            )
        if event.inspector_id == case.report_inspector:
            return TraceEntry(
                event, False, "same-inspector-reinspection",
                "reopening requires a different inspector",
            )
        settled = FodCase(
            case_id=case.case_id, segment_id=case.segment_id,
            reported_by_event=case.reported_by_event,
            report_round=case.report_round,
            report_inspector=case.report_inspector,
            opened_at=case.opened_at,
            state=CaseState.CONFIRMED_CLEAR,
            clearance_event=event.event_id,
            clearance_round=event.round_id,
            clearance_inspector=event.inspector_id,
            closed_at=event.occurred_at,
            observation_events=case.observation_events,
        )
        cases[event.case_id] = settled
        open_case.pop(event.segment_id, None)
        states[event.segment_id] = SegmentState.OPEN
        return TraceEntry(event, True, "case-confirmed-clear", "independent reinspection reopened the segment")

    def _require_segment(self, event: InspectionEvent) -> RouteSegment:
        try:
            return self._segments[event.segment_id]
        except KeyError:
            raise UnknownSegment(event.segment_id) from None

    def _submit(self, event: InspectionEvent, *, enforce_order: bool) -> TraceEntry:
        with self._lock:
            self._require_segment(event)
            existing = self._events.get(event.event_id)
            if existing is not None:
                if existing != event:
                    raise DuplicateEvent(f"event id {event.event_id} reused with different content")
                return next(
                    entry for entry in self._project().traces[event.segment_id]
                    if entry.event.event_id == event.event_id
                )
            if enforce_order:
                last = self._project().latest.get(event.segment_id)
                if last is not None and event.order_key < last.order_key:
                    raise StaleEvent(
                        f"{event.event_id} occurred {event.occurred_at:%H:%M:%S} "
                        f"before latest {last.event_id} at {last.occurred_at:%H:%M:%S}; use backfill"
                    )
            self._events[event.event_id] = event
            return next(
                entry for entry in self._project().traces[event.segment_id]
                if entry.event.event_id == event.event_id
            )

    def record_check(
        self,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        *,
        client_seq: int = 0,
        vehicle_id: str = "",
        note: str = "",
    ) -> TraceEntry:
        """Record a routine (clear) patrol observation while the vehicle is online."""
        event = InspectionEvent.check(
            event_id, segment_id, round_id, inspector_id, occurred_at,
            client_seq=client_seq, vehicle_id=vehicle_id, note=note,
        )
        entry = self._submit(event, enforce_order=True)
        if not entry.accepted and entry.effect == "rejected-clear-while-closed":
            raise SegmentClosed(entry.detail)
        return entry

    def report_fod(
        self,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        case_id: str,
        *,
        description: str = "",
        finding_id: str = "",
        client_seq: int = 0,
        vehicle_id: str = "",
    ) -> TraceEntry:
        """Report debris: opens a case and locally closes the segment."""
        event = InspectionEvent.finding(
            event_id, segment_id, round_id, inspector_id, occurred_at,
            case_id=case_id, finding_id=finding_id, description=description,
            client_seq=client_seq, vehicle_id=vehicle_id,
        )
        entry = self._submit(event, enforce_order=True)
        if not entry.accepted and entry.effect == "rejected-finding-while-closed":
            raise SegmentClosed(entry.detail)
        return entry

    def confirm_clearance(
        self,
        event_id: str,
        segment_id: str,
        round_id: str,
        inspector_id: str,
        occurred_at: datetime,
        case_id: str,
        *,
        client_seq: int = 0,
        vehicle_id: str = "",
        note: str = "",
    ) -> TraceEntry:
        """Re-inspection by another round/inspector; the only path back to open."""
        event = InspectionEvent.clearance(
            event_id, segment_id, round_id, inspector_id, occurred_at, case_id,
            client_seq=client_seq, vehicle_id=vehicle_id, note=note,
        )
        entry = self._submit(event, enforce_order=True)
        if not entry.accepted:
            if entry.effect == "clearance-without-open-case":
                raise NoOpenCase(entry.detail)
            if entry.effect == "same-round-reinspection":
                raise SameRoundReinspection(entry.detail)
            if entry.effect == "same-inspector-reinspection":
                raise SameInspectorReinspection(entry.detail)
        return entry

    def backfill(self, events: Iterable[InspectionEvent]) -> BackfillReport:
        """Fold offline-captured patrol observations into the timeline.

        The batch is validated, deduplicated against already-stored event ids,
        and committed atomistically.  Application order follows actual
        occurrence time (then vehicle-local sequence), not upload order, so a
        finding uploaded late cannot re-close a segment that an independent
        reinspection has already reopened.
        """
        candidates = tuple(events)
        with self._lock:
            accepted_events: list[InspectionEvent] = []
            duplicates: list[InspectionEvent] = []
            staged: dict[str, InspectionEvent] = dict(self._events)
            seen: set[str] = set()
            for event in candidates:
                self._require_segment(event)
                if event.event_id in self._events:
                    stored = self._events[event.event_id]
                    if stored != event:
                        raise DuplicateEvent(f"event id {event.event_id} reused with different content")
                    duplicates.append(stored)
                    continue
                if event.event_id in seen:
                    raise DuplicateEvent(f"duplicate event id inside batch: {event.event_id}")
                seen.add(event.event_id)
                staged[event.event_id] = event
                accepted_events.append(event)
            # Project over the merged timeline to read the effects of the batch.
            saved = dict(self._events)
            self._events = staged
            try:
                projection = self._project()
            finally:
                self._events = saved
            accepted_ids = {e.event_id for e in accepted_events}
            anomalies = tuple(
                entry
                for entries in projection.traces.values()
                for entry in entries
                if entry.event.event_id in accepted_ids and not entry.accepted
            )
            self._events = staged
            return BackfillReport(
                received=len(candidates),
                accepted=tuple(sorted(accepted_events, key=lambda e: e.order_key)),
                duplicates=tuple(duplicates),
                anomalies=anomalies,
            )

    def segment_state(self, segment_id: str) -> str:
        with self._lock:
            if segment_id not in self._segments:
                raise UnknownSegment(segment_id)
            return self._project().states[segment_id]

    def open_segments(self) -> tuple[str, ...]:
        with self._lock:
            projection = self._project()
            return tuple(sorted(s for s, state in projection.states.items() if state == SegmentState.OPEN))

    def closed_segments(self) -> tuple[str, ...]:
        with self._lock:
            projection = self._project()
            return tuple(sorted(s for s, state in projection.states.items() if state == SegmentState.CLOSED))

    def open_case(self, segment_id: str) -> FodCase | None:
        with self._lock:
            if segment_id not in self._segments:
                raise UnknownSegment(segment_id)
            return self._project().open_case.get(segment_id)

    def case(self, case_id: str) -> FodCase:
        with self._lock:
            try:
                return self._project().cases[case_id]
            except KeyError:
                raise NoOpenCase(case_id) from None

    def segment_trace(self, segment_id: str) -> SegmentTrace:
        with self._lock:
            try:
                segment = self._segments[segment_id]
            except KeyError:
                raise UnknownSegment(segment_id) from None
            projection = self._project()
            return SegmentTrace(
                segment,
                projection.states[segment_id],
                tuple(projection.traces[segment_id]),
                projection.open_case.get(segment_id),
            )

    def audit_trail(self) -> tuple[SegmentTrace, ...]:
        """Every segment with its full who/when/effect history."""
        with self._lock:
            projection = self._project()
            return tuple(
                SegmentTrace(
                    self._segments[segment_id],
                    projection.states[segment_id],
                    tuple(projection.traces[segment_id]),
                    projection.open_case.get(segment_id),
                )
                for segment_id in sorted(self._segments)
            )

    def timeline(self) -> tuple[InspectionEvent, ...]:
        with self._lock:
            return tuple(self._ordered())
