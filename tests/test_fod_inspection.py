import unittest
from datetime import datetime, timezone

from rampflow.fod_inspection import (
    RouteSegment,
    InspectionEvent,
    FodInspectionLedger,
    SegmentState,
    CaseState,
    UnknownSegment,
    DuplicateEvent,
    SegmentClosed,
    NoOpenCase,
    SameRoundReinspection,
    SameInspectorReinspection,
    StaleEvent,
)
from rampflow.workflows.control import RampWorkflowControl

UTC = timezone.utc
D = lambda h, m=0, s=0: datetime(2026, 1, 1, h, m, s, tzinfo=UTC)


class FodInspectionTests(unittest.TestCase):
    def setUp(self):
        self.ledger = FodInspectionLedger([
            RouteSegment("A1", "R-NORTH", "north apron lane 1"),
            RouteSegment("A2", "R-NORTH", "north apron lane 2"),
        ])

    def test_clear_round_leaves_segment_open_and_traced(self):
        entry = self.ledger.record_check("e1", "A1", "round-1", "inspector-ma", D(9))
        self.assertTrue(entry.accepted)
        self.assertEqual(entry.effect, "checked-clear")
        trace = self.ledger.segment_trace("A1")
        self.assertEqual(trace.state, SegmentState.OPEN)
        self.assertEqual(len(trace.entries), 1)
        self.assertEqual(trace.entries[0].event.inspector_id, "inspector-ma")
        self.assertEqual(self.ledger.closed_segments(), ())
        self.assertEqual(self.ledger.open_segments(), ("A1", "A2"))

    def test_finding_locally_closes_only_its_segment(self):
        entry = self.ledger.report_fod(
            "f1", "A1", "round-1", "inspector-ma", D(9, 5), case_id="C1",
            description="metal strip near stand 12",
        )
        self.assertEqual(entry.effect, "case-opened")
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.CLOSED)
        self.assertEqual(self.ledger.segment_state("A2"), SegmentState.OPEN)
        case = self.ledger.open_case("A1")
        self.assertIsNotNone(case)
        self.assertEqual(case.case_id, "C1")
        self.assertEqual(case.state, CaseState.OPEN)

    def test_routine_clear_rejected_while_segment_closed(self):
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        with self.assertRaises(SegmentClosed):
            self.ledger.record_check("e2", "A1", "round-1", "ma", D(9, 10))
        # the rejected attempt is still visible in the trail, nothing was hidden
        entries = self.ledger.segment_trace("A1").entries
        self.assertEqual(entries[-1].effect, "rejected-clear-while-closed")
        self.assertFalse(entries[-1].accepted)

    def test_second_finding_while_closed_is_attached_or_rejected(self):
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        # another debris sighting of the same open case is attached as evidence
        attached = self.ledger.report_fod(
            "f2", "A1", "round-2", "mb", D(9, 20), case_id="C1",
        )
        self.assertEqual(attached.effect, "finding-added-to-open-case")
        self.assertIn("f2", self.ledger.case("C1").observation_events)
        # a competing new case cannot be opened while the segment is closed
        with self.assertRaises(SegmentClosed):
            self.ledger.report_fod("f3", "A1", "round-2", "mb", D(9, 25), case_id="C2")

    def test_reopen_requires_different_round_and_inspector(self):
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        # same round, even with another inspector, cannot reopen
        with self.assertRaises(SameRoundReinspection):
            self.ledger.confirm_clearance("c1", "A1", "round-1", "mb", D(9, 30), "C1")
        # new round but the original inspector alone is not enough
        with self.assertRaises(SameInspectorReinspection):
            self.ledger.confirm_clearance("c2", "A1", "round-2", "ma", D(9, 35), "C1")
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.CLOSED)
        # independent round + independent inspector reopens
        entry = self.ledger.confirm_clearance("c3", "A1", "round-2", "mb", D(9, 40), "C1")
        self.assertEqual(entry.effect, "case-confirmed-clear")
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        case = self.ledger.case("C1")
        self.assertEqual(case.state, CaseState.CONFIRMED_CLEAR)
        self.assertEqual(case.clearance_round, "round-2")
        self.assertEqual(case.clearance_inspector, "mb")
        self.assertIsNone(self.ledger.open_case("A1"))

    def test_clearance_without_case_rejected(self):
        with self.assertRaises(NoOpenCase):
            self.ledger.confirm_clearance("c9", "A2", "round-2", "mb", D(10), "CX")

    def test_online_commands_must_arrive_in_occurrence_order(self):
        self.ledger.record_check("e1", "A1", "round-1", "ma", D(10))
        with self.assertRaises(StaleEvent):
            self.ledger.record_check("e0", "A1", "round-1", "ma", D(9, 50))
        # ordering is per segment; a different segment keeps its own clock
        self.ledger.record_check("x1", "A2", "round-1", "ma", D(8))

    def test_event_ids_are_idempotent(self):
        first = self.ledger.record_check("dup", "A1", "round-1", "ma", D(10), note="first")
        second = self.ledger.record_check("dup", "A1", "round-1", "ma", D(10), note="first")
        self.assertEqual(first, second)
        self.assertEqual(self.ledger.event_count, 1)
        with self.assertRaises(DuplicateEvent):
            self.ledger.record_check("dup", "A1", "round-1", "mb", D(10))

    def test_unknown_segment_rejected(self):
        with self.assertRaises(UnknownSegment):
            self.ledger.record_check("e1", "A9", "round-1", "ma", D(10))

    def test_audit_trail_reconstructs_who_checked_what_when(self):
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1",
                               description="plastic fragment")
        self.ledger.confirm_clearance("c1", "A1", "round-2", "mb", D(10), "C1")
        trails = {t.segment.segment_id: t for t in self.ledger.audit_trail()}
        effects = [(entry.event.round_id, entry.event.inspector_id, entry.effect)
                   for entry in trails["A1"].entries]
        self.assertEqual(effects, [
            ("round-1", "ma", "case-opened"),
            ("round-2", "mb", "case-confirmed-clear"),
        ])

    def test_backfill_applied_in_real_world_order(self):
        # Everything captured offline by the patrol car; uploaded shuffled.
        batch = [
            InspectionEvent.clearance("c1", "A1", "round-2", "mb", D(10), "C1"),
            InspectionEvent.finding("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1"),
            InspectionEvent.check("e0", "A1", "round-1", "ma", D(9)),
        ]
        report = self.ledger.backfill(batch)
        self.assertEqual(report.received, 3)
        self.assertEqual(len(report.accepted), 3)
        self.assertEqual(report.anomalies, ())
        # projected by occurrence time: opened at 09:05, reopened at 10:00
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        case = self.ledger.case("C1")
        self.assertEqual(case.state, CaseState.CONFIRMED_CLEAR)
        self.assertEqual(
            [e.event_id for e in self.ledger.timeline()],
            ["e0", "f1", "c1"],
        )

    def test_backfill_uses_client_seq_for_same_second_ties(self):
        batch = [
            InspectionEvent.clearance(
                "c1", "A1", "round-2", "mb", D(9, 30, 0), "C1", client_seq=2),
            InspectionEvent.finding(
                "f1", "A1", "round-1", "ma", D(9, 30, 0), case_id="C1", client_seq=1),
        ]
        report = self.ledger.backfill(batch)
        self.assertEqual(report.anomalies, ())
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)

    def test_late_offline_finding_does_not_reclose_settled_segment(self):
        # The case was opened and independently cleared while the car was online...
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        self.ledger.confirm_clearance("c1", "A1", "round-2", "mb", D(9, 40), "C1")
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        # ...then the car uploads a residual sighting of the SAME case it
        # captured at 09:50 (after the clearance) but only now synced.
        # The settled segment must stay open; the upload is flagged, not applied.
        late = InspectionEvent.finding(
            "f-late", "A1", "round-1", "ma", D(9, 50), case_id="C1",
        )
        report = self.ledger.backfill([late])
        self.assertEqual(len(report.accepted), 1)
        self.assertEqual(len(report.anomalies), 1)
        self.assertEqual(report.anomalies[0].effect, "late-finding-after-clearance")
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        self.assertIsNone(self.ledger.open_case("A1"))
        # but the late observation is preserved in the segment history
        effects = [e.effect for e in self.ledger.segment_trace("A1").entries]
        self.assertIn("late-finding-after-clearance", effects)

    def test_late_observation_while_case_was_still_open_attaches_as_evidence(self):
        # At 09:10 the case was genuinely still open (cleared 09:40), so a
        # same-case sighting uploaded afterwards is evidence, not a re-closure.
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        self.ledger.confirm_clearance("c1", "A1", "round-2", "mb", D(9, 40), "C1")
        report = self.ledger.backfill([
            InspectionEvent.finding("f-mid", "A1", "round-1", "ma", D(9, 10), case_id="C1"),
        ])
        self.assertEqual(report.anomalies, ())
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        self.assertIn("f-mid", self.ledger.case("C1").observation_events)

    def test_late_finding_of_new_case_still_closes_after_clearance(self):
        # A genuinely new debris case observed after the first one was cleared
        # (even if uploaded later) must close the segment and demand a fresh round.
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        self.ledger.confirm_clearance("c1", "A1", "round-2", "mb", D(9, 40), "C1")
        report = self.ledger.backfill([
            InspectionEvent.finding("f2", "A1", "round-3", "mc", D(10, 5), case_id="C2"),
        ])
        self.assertEqual(report.anomalies, ())
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.CLOSED)
        self.assertEqual(self.ledger.open_case("A1").case_id, "C2")

    def test_backfill_is_atomic_and_deduplicates(self):
        self.ledger.record_check("e1", "A1", "round-1", "ma", D(9))
        good = InspectionEvent.finding("f1", "A9", "round-1", "ma", D(9, 10), case_id="C1")
        # batch references an unknown segment: nothing is committed
        with self.assertRaises(UnknownSegment):
            self.ledger.backfill([good])
        self.assertEqual(self.ledger.event_count, 1)
        # replaying the same batch is a no-op
        batch = [
            InspectionEvent.check("e1", "A1", "round-1", "ma", D(9)),
            InspectionEvent.finding("f2", "A1", "round-2", "mb", D(10, 5), case_id="C9"),
        ]
        report = self.ledger.backfill(batch)
        self.assertEqual(len(report.duplicates), 1)
        self.assertEqual(len(report.accepted), 1)
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.CLOSED)

    def test_mixed_online_then_offline_history_stays_consistent(self):
        # online: opened
        self.ledger.report_fod("f1", "A1", "round-1", "ma", D(9, 5), case_id="C1")
        # car goes offline; on return it delivers an earlier routine check and
        # the independent clearance
        report = self.ledger.backfill([
            InspectionEvent.check("e0", "A1", "round-1", "ma", D(9), client_seq=0),
            InspectionEvent.clearance("c1", "A1", "round-2", "mb", D(10), "C1"),
        ])
        self.assertEqual(report.anomalies, ())
        self.assertEqual(self.ledger.segment_state("A1"), SegmentState.OPEN)
        timeline = self.ledger.timeline()
        self.assertEqual([e.event_id for e in timeline], ["e0", "f1", "c1"])
        # after the merged history, online commands resume with the newest clock
        self.ledger.record_check("e9", "A1", "round-3", "mc", D(10, 30))


class FodWorkflowTests(unittest.TestCase):
    def test_patrol_change_publishes_once_and_is_idempotent(self):
        control = RampWorkflowControl()
        ledger = FodInspectionLedger([RouteSegment("A1")])
        calls = []

        def file_finding():
            calls.append("f")
            return ledger.report_fod(
                "f1", "A1", "round-1", "ma",
                datetime(2026, 1, 1, 9, 5, tzinfo=UTC), case_id="C1",
            )

        at = datetime(2026, 1, 1, 9, 6, tzinfo=UTC)
        first = control.commit_fod_patrol("op-fod-1", "A1", at, file_finding)
        second = control.commit_fod_patrol("op-fod-1", "A1", at, file_finding)
        self.assertIs(first, second)
        self.assertEqual(calls, ["f"])
        self.assertEqual(control.stream.head, 1)
        self.assertEqual(first.event.topic, "apron.fod.patrol.committed")
        self.assertEqual(ledger.segment_state("A1"), SegmentState.CLOSED)

    def test_rejected_patrol_publishes_nothing(self):
        control = RampWorkflowControl()
        ledger = FodInspectionLedger([RouteSegment("A1")])
        ledger.report_fod("f1", "A1", "round-1", "ma",
                          datetime(2026, 1, 1, 9, 5, tzinfo=UTC), case_id="C1")

        def same_round_reopen():
            ledger.confirm_clearance(
                "c-bad", "A1", "round-1", "mb",
                datetime(2026, 1, 1, 9, 20, tzinfo=UTC), "C1",
            )

        with self.assertRaises(SameRoundReinspection):
            control.commit_fod_patrol(
                "op-fod-2", "A1",
                datetime(2026, 1, 1, 9, 21, tzinfo=UTC),
                same_round_reopen,
            )
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})
        self.assertEqual(ledger.segment_state("A1"), SegmentState.CLOSED)


if __name__ == "__main__":
    unittest.main()
