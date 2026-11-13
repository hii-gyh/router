# Live method — host 110 Qwen3-0.6B

Agent ops runbook (SSH, bootstrap, pitfalls): [`../../agent_remote_patent_live.md`](../../agent_remote_patent_live.md).

## Cluster

- Single Ascend 910B3 host (`39.156.164.110`), IF `data0.3001` / `192.168.151.55`
- Image `quay.io/ascend/vllm-ascend:v0.30.0rc2`, `MooncakeConnectorV1`, TP=1
- Router binary built on-host (`cargo build --release --bin vllm-router`), run host-native (container lacks `libssl.so.1.1`)

## Differentiation (matches disclosure)

| role | containers | NPUs | HTTP | KV | max_model_len |
|------|------------|------|------|-----|---------------|
| Prefill short | pd-pat-prefill-0/1 | 0–1 | 18001–02 | 30000–01 | 2048 |
| Prefill open | pd-pat-prefill-2/3 | 2–3 | 18003–04 | 30002–03 | 8192 |
| Decode | pd-pat-decode-0/1 | 4–5 | 18011–12 | 30100–01 | 8192 |

Bucket ceiling for soft/hard Prefill policies: `1024`. Short-pool workers labeled bucket `0` via `--prefill-bucket-worker`.

## Arms

- **A0**: Prefill RR + Decode RR
- **Hard**: `prefill_bucket` + `--prefill-hard-partition` + spill
- **SoftFull**: `prefill_bucket` soft preference (`--prefill-preference-weight 2.0`)
- **SoftPlusDecode**: SoftFull Prefill + `decode_kv_cost`

## Load

- Trajectory: synthetic ShareGPT-ish mixture (`seed=20251113`), client `real_pd_trajectory_client.py`
- `MODEL_NAME=qwen3-0.6b`, `rate=1.5`, `n=40`, `--ignore-eos`
- Script: `scripts/910b3/qwen06_pd_patent_1host.sh`

## Caveats

- Small-n / small-model for fast verification; not a substitute for large-model production figures.
- SoftPlusDecode vs SoftFull not separated under this Decode symmetry.
- Do not phrase as the only quantitative support in the disclosure; pair with device-free sims in the parent directory.
