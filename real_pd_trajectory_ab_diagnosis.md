# Real-PD trajectory A/B diagnosis + E1 remesaure

## Locked criteria
At R_sat: experiment.output_tok_s > control.output_tok_s AND experiment.ttft_p90 ≤ control.ttft_p90.

## Prior root cause (unchanged)
- Prefill hard partition pinned shorts onto 2/4 workers under ~95%-short mix.
- Spill abs=2/rel=1.0 on 2+2 helped long TTFT but still FAIL vs RR.
- Decode remaining-output-load is NOT the fix.

## Intervention E1: 1 short + 3 open
- Topology: **1×4096 (18001 bucket 0) + 3×16384 (18002–04 open)** + 4 decode.
- Worker restarted: `pd-real-prefill-1` 4096→16384. Left `vllm-ascend-*` alone. No mock PD.
- Router: `--prefill-bucket-worker http://127.0.0.1:18001=0` only; spill abs=2 rel=1.0; bucket max 2048.
- E2 (0 short + 4 open) **skipped**: no soft length preference without new code (`power_of_two` is load-only).

## E1 remesaure (seed 20251031, offsets 440/500/560, N=60)

| rate | RR tok/s | exp tok/s | RR TTFT p90 | exp TTFT p90 | exp short/open prompts | PASS |
|------|----------|-----------|-------------|--------------|------------------------|------|
| 4.0 | 338.2 | 319.9 | 0.764 | 0.779 | 22334 / 22092 | FAIL |
| 6.4 | 399.0 | 394.3 | 1.579 | 1.801 | 11891 / 30011 | FAIL |
| 8.0 | 403.6 | 447.3 | 1.093 | 1.128 | 16824 / 23119 | FAIL |

- **passed_at_R_sat=False**
- R_sat=8.0: ots_win=True (447.3 > 403.6) but ttft_ok=False (1.128 > 1.093) → dual criteria FAIL.
- At 4.0 experiment still overloads the single short card (22334 vs RR’s balanced ~9–12k/worker) before spill catches up.
- At 6.4 spill nearly empties the short card relative to open, but system TTFT and tok/s both worse than RR.

## Honest stop (E1 still fails both criteria)
Do **not** keep tweaking spill. Required next change:
1. Soft score-based placement over all 4 prefills (prefer shorter max-len softly, no hard pin), **or**
2. Homogenize all 4 prefills to 16384 and accept RR vs power_of_two / decode_kv_cost-only (pure load-balance A/B, not length routing).
