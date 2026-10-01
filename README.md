# RampFlow Ground Operations Lab

RampFlow is a dependency-free Python domain library for airport turnaround and ramp operations. It models operational resources and constraints rather than exposing a toy CRUD service. The package includes turnaround task scheduling, resource reservations, runway slots, stand allocation, taxi routing, fueling, deicing, baggage reconciliation, maintenance deferrals, passenger connections, crew duty, mobile resource dispatch, apron safety, segmented FOD patrols with local closure and independent reinspection, NOTAM recurrence, disruption recovery, and immutable operational snapshots.

The implementation uses only the Python standard library at runtime. State-changing services use explicit version, generation, reservation, or replay semantics where the real-world workflow requires them. Cross-domain workflows publish outward events only after their domain change succeeds and use operation IDs for replay safety. Public tests exercise representative invariants, import every packaged module, and the CLI provides a dependency-free health check and snapshot digest utility.

## Quick check

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m rampflow.cli health --json
```

## Package layout

- `src/rampflow/`: production domain code.
- `src/rampflow/workflows/`: cross-domain commit and event-consistency boundary.
- `tests/`: public regression and smoke tests.
- `pyproject.toml`: package metadata. Runtime dependencies are empty.

## Segmented FOD patrols

`rampflow.fod_inspection.FodInspectionLedger` replaces paper slips and chat
threads with an event-sourced trail per route segment:

- Every patrol observation (clear check, debris finding, reinspection clearance)
  is an immutable event carrying segment, patrol round, inspector and time.
  `segment_trace()` / `audit_trail()` show exactly which stretch was checked,
  by whom, and what changed.
- A finding opens a FOD case that **locally closes just that segment**; routine
  checks and competing cases are rejected while it is closed, and additional
  same-case sightings are attached as evidence.
- Reopening requires a **different patrol round and a different inspector**
  (`confirm_clearance`); the original round self-clearing is rejected.
- Online commands must arrive in occurrence order. Offline patrol cars upload a
  batch with `backfill()`, which is deduped and replayed in actual-occurrence
  order (vehicle-local sequence breaks same-timestamp ties), so late uploads
  extend history rather than re-closing a segment an independent round has
  already settled. Late same-case sightings after clearance are recorded as
  anomalies, never applied as closures.
- The cross-domain boundary exposes the workflow as
  `RampWorkflowControl.commit_fod_patrol`, which publishes an outward event only
  after the domain change succeeds and is idempotent by operation id.

The repository is intentionally self-contained and does not require network access or external services for its domain logic.
