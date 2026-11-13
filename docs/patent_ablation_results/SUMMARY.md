# Patent ablation simulation results

> Device-free queue simulation only. Do **not** phrase these as live PD cluster measurements in the disclosure.

- Seeds: [1, 7, 21, 99, 20251103]
- Prefill jobs/seed: 800
- Decode jobs/seed: 400
- Handoff jobs/seed: 300

## Exp1 Prefill soft-tier ablation

| Arm | High-bucket occupancy | Soft borrows | Short mean wait | Short P99 wait | Length misalloc |
|-----|----------------------:|-------------:|----------------:|---------------:|----------------:|
| Hard | 0.569 | 361.8 | 129114.7 | 296240.0 | 0.0 |
| SoftNoCross | 0.112 | 0.0 | 222199.3 | 440046.4 | 0.0 |
| SoftFull | 0.660 | 433.4 | 130290.0 | 322160.8 | 0.0 |

**Read:** SoftFull keeps misalloc=0 like Hard, borrows more than SoftNoCross (433 vs 0), and cuts short mean/P99 wait vs SoftNoCross by ~41%/27%. SoftFull stays within ~10% of Hard on short waits while enabling earlier cross-tier borrow (not delayed spill-only).

## Exp2 Decode joint-cost ablation

| Arm | KV overflows | Reject rate | P99 TPOT proxy | Large-on-hot | Usage σ |
|-----|-------------:|------------:|---------------:|-------------:|--------:|
| RoundRobin | 200.0 | 0.250 | 175.0 | 67.0 | 0.180 |
| PowerOfTwoLoad | 312.0 | 0.667 | 150.0 | 134.0 | 0.507 |
| KvQueueNoLength | 305.0 | 0.000 | 173.0 | 96.0 | 0.694 |
| FullKvCost | 133.0 | 0.000 | 177.0 | 19.0 | 0.234 |

**Read:** FullKvCost cuts KV overflows vs RR/PoT/NoLength (133 vs 200/312/305) and large-on-hot vs NoLength (19 vs 96), showing both full-pool KV scoring and the L_req admission term matter.

## Exp3 Stale fallback + handoff reselection

| Arm | Wrong rejects | Stale scored | Handoff fail rate | Mean recovery ticks | Reselect success |
|-----|--------------:|-------------:|------------------:|--------------------:|-----------------:|
| NoFallback | 200.0 | 100.0 | 1.000 | 30.0 | 0.0 |
| RecheckOnly | 0.0 | 0.0 | 1.000 | 30.0 | 0.0 |
| RecheckReselect | 0.0 | 0.0 | 0.000 | 2.0 | 300.0 |

**Read:** RecheckReselect drives handoff fail rate 1.0→0.0 and recovery ticks 30→2, with 300 successful reselections and zero stale scoring / wrong rejects.

## Figure notes (2–3 key charts)

1. Prefill short P99 wait bars: Hard vs SoftNoCross vs SoftFull
2. Decode large-on-hot + KV overflows grouped bars across four arms
3. Handoff fail rate + mean recovery ticks across three arms

CSV companions in this directory can be plotted offline.
