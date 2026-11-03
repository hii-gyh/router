# Decode KV-cost — best-fit & high-KV pressure remesaure

## Current decode scheme

`decode_kv_cost` always uses **full-pool min-cost** over eligible decode workers
(`1.0·U + 0.35·q + 0.50·a`). There is no low-load round-robin floor.

KV-cost wins when:

1. **Decode KV pressure is real** — mean-of-max `kv_usage` in ~**0.40–0.70+** (not the prior ~0.10–0.12).
2. **Request shape is heterogeneous** — long prompts (multi-k) + **uneven long outputs** so RR creates a hotspot D while another stays cool; short probes arrive while residents still hold KV.
3. **Pool is small (≤8 D)** — full-pool scan matters (power-of-two previously *worsened* `kv_std` on 4D).
4. **Arrival rate high enough to overlap residents**, but not so pathologically queued that TTFT is dominated by prefill/transfer noise alone.

It does **not** win on the prior ShareGPT-ish traj (seed `20251031`, rates 4.0/6.4/8.0): peak KV ~10–12%, mostly short prompts, little pressure for cost-based placement to beat RR.

## Pressure environment designed here

| Lever | Value |
|--|--|
| Topology | unchanged real Ascend PD: 2P@4096 + 2P@16384 + 4D@16384 |
| Traj | `prepare-high-kv` seed **20251102** — 16–24 preload residents + ~50% resident (prompt~4–10k, out~1–5k), ~25% medium, ~25% probes |
| Client | `--ignore-eos` so residents actually hold `max_tokens` |
| Rates | **1.0 / 1.6 / 2.4** (R_sat=2.4); N=60; offsets 0/60/120 |
| Arms | A0 RR+RR vs A1 RR+`decode_kv_cost` (full-pool min-cost) |
| Pack | `/root/gyh_dev/vllm_router/logs/decode_high_kv_pack(.tgz)` |

Occupancy target: push **mean max kv_usage into 0.40–0.70** and `kv_frac_max_ge_0_25` high enough that KV pressure is real.

## How to run (remote)

```bash
# on NPU host, after workers are healthy and decode_kv_cost binary is installed
cd /root/gyh_dev/vllm_router
# upload updated client + this script first
chmod +x run_real_pd_decode_high_kv_remeasure.sh
setsid nohup bash ./run_real_pd_decode_high_kv_remeasure.sh \
  > ./logs/decode_high_kv_outer.log 2>&1 < /dev/null &
```

Local dry-run of traj + occupancy model:

```bash
python3 real_pd_trajectory_client.py --mode prepare-high-kv \
  --traj /tmp/trajectory_high_kv.json --seed 20251102
python3 tmp_estimate_high_kv_occupancy.py
```

## Success gate

Same dual criteria at **R_sat=2.4**: A1 `output_tok_s` &gt; A0 **and** A1 `ttft_p90` ≤ A0. Also require `pressure_target_hit=true` (KV left the light band — prefer `kv_mean_of_max≥0.35` or `kv_frac_max_ge_0_25≥0.40`, not inflight-only); otherwise the arm is not a valid best-case claim.
