# Patent live ablation (host 110 / Qwen3-0.6B)

- host=`bms-84092057-007` (`39.156.164.110`)
- topology (disclosure-aligned differentiation):
  - Prefill **short** ×2 — NPU0–1, `:18001–02`, `max_model_len=2048`
  - Prefill **open** ×2 — NPU2–3, `:18003–04`, `max_model_len=8192`
  - Decode ×2 — NPU4–5, `:18011–12`, `max_model_len=8192`
- model=`Qwen3-0.6B` / served=`qwen3-0.6b`, ceiling=`1024`, rate=`1.5` rps, n=`40`
- image=`quay.io/ascend/vllm-ascend:v0.30.0rc2`, connector=`MooncakeConnectorV1`
- note: **measured live** on differentiated Ascend 910B3 cluster; use with device-free sim — do not claim sole patent proof from this small-n run alone

## Headline metrics

| arm | n_ok | err_rate | output_tok_s | ttft_p50 (s) | ttft_p90 (s) | tpot_p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| A0 (RR) | 39 | 0.025 | 267.1 | 0.199 | **5.607** | 0.020 |
| Hard | 40 | 0.000 | 272.5 | 0.197 | **0.470** | 0.020 |
| SoftFull | 40 | 0.000 | 276.5 | 0.188 | **0.208** | 0.020 |
| SoftPlusDecode | 40 | 0.000 | 275.3 | 0.188 | **0.207** | 0.020 |

## Per-bucket TTFT p90

| arm | short_le_2048 n | short ttft_p90 | long_gt_2048 n | long ttft_p90 |
|---|---:|---:|---:|---:|
| A0 | 36 | 7.254 | 3 | 0.255 |
| Hard | 36 | 1.278 | 4 | 0.250 |
| SoftFull | 36 | 0.207 | 4 | 0.249 |
| SoftPlusDecode | 36 | 0.203 | 4 | 0.253 |

## Verdict

- Differentiated Prefill short/open pools match the disclosure topology and force length-aware routing to matter.
- RR (A0) piles short-pool contention (short TTFT p90 ≈ 7.3 s) and can 400 when a long prompt lands on `max_len=2048` workers.
- Hard / SoftFull / SoftPlusDecode keep `err_rate=0` and cut system TTFT p90 by ~12–27× vs A0 at this load.
- SoftFull ≈ SoftPlusDecode here; Decode `L_req` needs stronger Decode asymmetry / higher pressure to separate.

## Artifacts

- `metrics.csv`, `A0_r1.5.json`, `Hard_r1.5.json`, `SoftFull_r1.5.json`, `SoftPlusDecode_r1.5.json`
- remote: `/root/gyh_dev/vllm_router/logs/patent_live/`
