import unittest
from datetime import datetime, timezone

from rampflow.fod_inspection import (
    CheckResult,
    FodEvent,
    FodEventType,
    FodInspectionService,
    InspectionRoute,
    InspectionRound,
    InspectionSegment,
    SegmentState,
)
from rampflow.workflows.control import RampWorkflowControl


UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 3, 10, h, m, tzinfo=UTC)


def build_service():
    service = FodInspectionService()
    segments = (
        InspectionSegment("S1", "R1", 0, "taxi-lane-a"),
        InspectionSegment("S2", "R1", 1, "stand-row-b"),
        InspectionSegment("S3", "R1", 2, "ramp-exit-c"),
    )
    service.register_route(InspectionRoute("R1", segments))
    service.register_round(InspectionRound("round-1", "inspector-alice", "VAN-7", D(8)))
    service.register_round(InspectionRound("round-2", "inspector-bob", "VAN-3", D(9)))
    return service


class FodInspectionTests(unittest.TestCase):
    def test_segmented_patrol_leaves_per_segment_trail(self):
        service = build_service()
        service.record_check("S1", "round-1", D(8, 10), "e-s1-1")
        service.record_check("S2", "round-1", D(8, 20), "e-s2-1", CheckResult.ATTENTION, "metal strip seen")

        trail = service.audit_trail("R1")
        codes = [entry.code for entry in trail]
        self.assertEqual(codes, ["segment.checked", "segment.checked"])
        self.assertEqual([entry.segment_id for entry in trail], ["S1", "S2"])
        self.assertTrue(all(entry.round_id == "round-1" for entry in trail))

        coverage = service.segment_coverage("R1")
        self.assertEqual(coverage["S1"][0].inspector_id, "inspector-alice")
        self.assertEqual(coverage["S2"][0].result, CheckResult.ATTENTION)
        self.assertEqual(coverage["S3"], ())
        self.assertEqual(service.controlled_segments(), ())

    def test_finding_orders_local_containment(self):
        service = build_service()
        cycle = service.report_finding("S2", "round-1", "FOD-9", D(8, 25), "e-find-1", "nut on lane")

        self.assertEqual(service.state("S2"), SegmentState.CONTROLLED)
        self.assertEqual(service.state("S1"), SegmentState.OPEN)
        orders = service.controlled_segments()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].segment_id, "S2")
        self.assertEqual(orders[0].reason_fod_id, "FOD-9")
        self.assertTrue(orders[0].restricted)
        self.assertEqual(cycle.cycle_no, 1)
        self.assertTrue(cycle.opened)

        trail_codes = [e.code for e in service.audit_trail("R1") if e.segment_id == "S2"]
        self.assertEqual(trail_codes, ["fod.reported", "segment.controlled"])

    def test_clearance_requires_another_round_and_inspector(self):
        service = build_service()
        service.report_finding("S2", "round-1", "FOD-9", D(8, 25), "e-find-1")

        # the original round cannot clear its own finding
        with self.assertRaises(RuntimeError):
            service.verify_clearance("S2", "round-1", D(8, 40), "e-clear-self")
        self.assertEqual(service.state("S2"), SegmentState.CONTROLLED)

        # an independent round/inspector re-checks and confirms
        service.record_check("S2", "round-2", D(9, 5), "e-recheck", note="removed and swept")
        reopened = service.verify_clearance("S2", "round-2", D(9, 6), "e-clear-ok")

        self.assertEqual(service.state("S2"), SegmentState.OPEN)
        self.assertEqual(service.controlled_segments(), ())
        self.assertIsNotNone(reopened.clearance)
        self.assertEqual(reopened.clearance.inspector_id, "inspector-bob")
        self.assertEqual([e.code for e in service.audit_trail("R1") if e.segment_id == "S2"][-1],
                         "segment.reopened")

    def test_clearance_without_open_containment_is_rejected(self):
        service = build_service()
        with self.assertRaises(RuntimeError):
            service.verify_clearance("S1", "round-2", D(9, 10), "e-clear-nothing")

    def test_offline_batch_replays_in_occurrence_order(self):
        service = build_service()
        # online: alice reports FOD at 08:25; bob verifies at 09:06 -> reopened
        service.report_finding("S2", "round-1", "FOD-9", D(8, 25), "e-find-1")
        service.verify_clearance("S2", "round-2", D(9, 6), "e-clear-ok")
        self.assertEqual(service.state("S2"), SegmentState.OPEN)

        # van-7 comes back online and uploads its whole trip out of upload order:
        # a check at 08:05, the finding at 08:25 (duplicate), and a late check
        # at 08:40 that belongs to the already-closed cycle
        batch = (
            FodEvent("e-late-check", FodEventType.CHECK, D(8, 40), "S2", "round-1", seq=30,
                     result=CheckResult.CLEAR, note="still swept by alice"),
            FodEvent("e-early-check", FodEventType.CHECK, D(8, 5), "S2", "round-1", seq=10),
            FodEvent("e-find-1", FodEventType.FIND, D(8, 25), "S2", "round-1", seq=20, fod_id="FOD-9"),
        )
        outcomes = service.ingest_offline(batch, received_at=D(10))
        statuses = {o.event_id: o.status for o in outcomes}
        self.assertEqual(statuses["e-find-1"], "duplicate")
        self.assertEqual(statuses["e-late-check"], "applied")
        self.assertEqual(statuses["e-early-check"], "applied")

        # the closed segment must not be re-controlled or re-opened
        self.assertEqual(service.state("S2"), SegmentState.OPEN)
        self.assertEqual(service.controlled_segments(), ())
        cycle = service.cycles("S2")[-1]
        self.assertIsNotNone(cycle.clearance)

        # but the evidence is preserved in chronological order for review
        checks = service.checks("S2")
        self.assertEqual([c.at for c in checks], sorted(c.at for c in checks))
        self.assertEqual([c.event_id for c in checks],
                         ["e-early-check", "e-late-check"])
        trail = service.audit_trail("R1")
        self.assertEqual([e.at for e in trail], sorted(e.at for e in trail))

    def test_late_clearance_event_does_not_reopen_closed_segment(self):
        service = build_service()
        service.report_finding("S3", "round-1", "FOD-1", D(8, 20), "e-find-a")
        service.verify_clearance("S3", "round-2", D(9, 0), "e-clear-a")
        # a duplicate of bob's clearance re-uploaded offline
        outcomes = service.ingest_offline(
            [FodEvent("e-clear-a", FodEventType.CLEAR, D(9, 0), "S3", "round-2", seq=40)]
        )
        self.assertEqual(outcomes[0].status, "duplicate")
        self.assertEqual(service.state("S3"), SegmentState.OPEN)

    def test_new_finding_after_reopen_starts_a_fresh_cycle(self):
        service = build_service()
        service.report_finding("S1", "round-1", "FOD-1", D(8, 20), "e-find-a")
        service.verify_clearance("S1", "round-2", D(9, 0), "e-clear-a")

        # genuinely new finding after the segment was reopened, uploaded offline
        outcomes = service.ingest_offline(
            [FodEvent("e-find-b", FodEventType.FIND, D(9, 30), "S1", "round-2", seq=50,
                      fod_id="FOD-2", note="debris reappeared")]
        )
        self.assertEqual(outcomes[0].status, "applied")
        self.assertEqual(service.state("S1"), SegmentState.CONTROLLED)
        cycles = service.cycles("S1")
        self.assertEqual(len(cycles), 2)
        self.assertIsNone(cycles[1].clearance)
        self.assertEqual(cycles[1].finding.fod_id, "FOD-2")

    def test_duplicate_online_event_id_is_rejected(self):
        service = build_service()
        service.record_check("S1", "round-1", D(8, 10), "dup")
        with self.assertRaises(ValueError):
            service.record_check("S1", "round-1", D(8, 12), "dup")

    def test_workflow_boundary_publishes_containment_and_reopen_once(self):
        service = build_service()
        control = RampWorkflowControl()

        receipt = control.order_fod_containment(
            "op-1", "S2", D(8, 25),
            lambda: service.report_finding("S2", "round-1", "FOD-9", D(8, 25), "e-find-1"),
        )
        self.assertEqual(receipt.event.topic, "apron.fod.containment.ordered")
        # replay of the same operation must not double-control the segment
        again = control.order_fod_containment(
            "op-1", "S2", D(8, 25),
            lambda: service.report_finding("S2", "round-1", "FOD-9", D(8, 25), "e-find-1"),
        )
        self.assertIs(again, receipt)
        self.assertEqual(len(service.cycles("S2")), 1)

        reopen = control.publish_fod_reopen(
            "op-2", "S2", D(9, 6),
            lambda: service.verify_clearance("S2", "round-2", D(9, 6), "e-clear-ok"),
        )
        self.assertEqual(reopen.event.topic, "apron.fod.segment.reopened")
        self.assertEqual(service.state("S2"), SegmentState.OPEN)
        self.assertTrue(control.stream.verify_chain())


if __name__ == "__main__":
    unittest.main()
