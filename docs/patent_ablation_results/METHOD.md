# Patent ablation method notes

## Scope

Device-free queue simulation only (`tests/patent_ablation_sim.rs`).
These results support **mechanism ablation / expected effect** language in the
disclosure. They are **not** live PD cluster measurements.

## Reproducibility

| Item | Value |
|------|-------|
| Branch | `feat/soft-length-aware` |
| Test | `cargo test --features cpu,ready --test patent_ablation_sim -- --nocapture` |
| Seeds | 1, 7, 21, 99, 20251103 (5 repeats, report means) |
| Prefill jobs/seed | 800 |
| Decode jobs/seed | 400 |
| Handoff jobs/seed | 300 |
| Artifacts | `SUMMARY.md`, `metrics.csv` |

## Experiment mapping to review checklist

1. **Prefill soft-tier**: Hard partition vs SoftNoCross (soft preference, no cross-tier sample) vs SoftFull (soft preference + preferred-vs-other compare).
2. **Decode joint cost**: RR vs Power-of-Two(load) vs KV+queue (no L_req) vs Full KV cost.
3. **Stale/handoff**: NoFallback vs RecheckOnly vs RecheckReselect under injected stale/block anomalies during Prefill→Decode handoff.

## Topology / load (fixed across arms)

- Prefill: 1 short-only worker + 2 open workers; short-heavy arrivals (≈88% short).
- Decode: 2 workers with induced KV-hot/queue-light vs KV-cooler/queue-heavy asymmetry.
- Handoff: 2 decode workers; after initial select, freeze metrics and spike true pressure / unavailability.

## Live cluster (deferred)

Hosts `39.156.164.107–110` can host a follow-up real PD nerdctl run with the
same trajectory/topology. Until then, disclosure text should keep
“预期效果、可由实验验证”, not “已实测提升”.
