# Agent guide: patent live remote (Ascend 910B3)

Operational runbook for **other agents** to reuse the lab cluster used for PD length-aware routing live retests. Policy theory and experiment design: [`910b3_pd_length_policy_test.md`](910b3_pd_length_policy_test.md). Device-free sims: [`patent_ablation_results/`](patent_ablation_results/). Latest live numbers: [`patent_ablation_results/live_110_qwen06/`](patent_ablation_results/live_110_qwen06/).

**Do not commit passwords, keys, or `.env` files.** Use `SSH_PASS` in the environment only.

---

## 1. Host inventory

| Public IP | Role (typical) | Notes |
|-----------|----------------|-------|
| `39.156.164.107` | spare / alternate | Probe before use; disk can fill |
| `39.156.164.108` | spare / alternate | |
| `39.156.164.109` | spare / alternate | |
| `39.156.164.110` | **primary live retest** | Validated Nov 2026 |

Primary host facts (`110`):

| Item | Value |
|------|--------|
| Hostname | `bms-84092057-007` |
| Arch | `aarch64` |
| NPUs | 8× Ascend 910B3 (`/dev/davinci0`–`7`) |
| RDMA / data IF | `data0.3001` |
| Data-plane IP | `192.168.151.55` |
| Weights | `/mnt/weights/Qwen3-0.6B` (mounted into containers as `/home/weights/...`) |
| Container runtime | `nerdctl` (OK to stop conflicting containers) |
| Image | `quay.io/ascend/vllm-ascend:v0.30.0rc2` |
| Router workspace | `/root/gyh_dev/vllm_router` |
| Live logs | `/root/gyh_dev/vllm_router/logs/patent_live/` |

SSH: user `root`, password auth. Lab password is provided out-of-band by the operator — export as `SSH_PASS` (never hardcode into git).

---

## 2. Local machine assumptions (Cursor / Windows)

Agents usually drive the cluster from **WSL** with:

```bash
export SSH_PASS='…'          # from operator; never commit
export SSH_USER=root
```

Helpers (repo root on Windows often under `/mnt/c/Users/.../codes/exps/router`):

| Script | Purpose |
|--------|---------|
| `scripts/patent_ssh.py` | `run` / `upload` / `upload-run` via PTY password SSH |
| `scripts/_tmp_scp_upload.py` | Fast password `scp` (preferred for files >~50KB) |
| `scripts/_wsl_long_run.sh` | Long remote commands (build / wait / ablation) with custom SSH timeout |

### Quoting pitfalls (read before scripting)

1. **PowerShell** mangles nested `\"` inside `wsl -e bash -lc "..."`. Prefer: put remote logic in a `.sh`/`.py` file → `scp` → `bash /tmp/...`.
2. Password characters like `[` need **bash single quotes**: `export SSH_PASS='…'`.
3. `patent_ssh.py upload` chunk-encodes via many SSH round-trips — fine for tiny scripts; for source tarballs use **`_tmp_scp_upload.py`**.
4. Do **not** pack a tarball into the same directory you are tarring (self-include / “file changed as we read it”). Write to `/tmp` or a sibling path outside the tree.

---

## 3. Differentiated topology (disclosure-aligned)

One-host PD layout used by `scripts/910b3/qwen06_pd_patent_1host.sh` (small model for fast verify):

| Role | Names | NPU | HTTP | KV port | `max_model_len` |
|------|-------|-----|------|---------|-----------------|
| Prefill **short** | `pd-pat-prefill-0/1` | 0–1 | 18001–02 | 30000–01 | **2048** |
| Prefill **open** | `pd-pat-prefill-2/3` | 2–3 | 18003–04 | 30002–03 | **8192** |
| Decode | `pd-pat-decode-0/1` | 4–5 | 18011–12 | 30100–01 | 8192 |
| Router | host process on `:8090` | — | 8090 | — | — |

- Short-pool workers labeled bucket `0`; soft/hard ceiling default `1024`.
- Prefill = `kv_producer`, Decode = `kv_consumer`, connector `MooncakeConnectorV1`.
- Served name: `qwen3-0.6b` (must match client `MODEL_NAME`).

---

## 4. Bootstrap router binary on host

Host is aarch64 — **build on the host** (or another aarch64 box). Do not scp an x86/Windows binary.

```bash
# On remote (after rustup + openssl-devel installed once):
source /root/.cargo/env
cd /root/gyh_dev/vllm_router
cargo build --release --bin vllm-router
cp -f target/release/vllm-router ./vllm-router
```

Sync source from the agent machine (example):

```bash
# From WSL
tar czf /tmp/router-patent-src.tgz -C /path/to/router \
  --exclude=target --exclude=.git --exclude='**/__pycache__' .
python3 scripts/_tmp_scp_upload.py /tmp/router-patent-src.tgz 39.156.164.110 /tmp/router-patent-src.tgz
# then on remote: tar xzf /tmp/router-patent-src.tgz -C /root/gyh_dev/vllm_router
```

First `cargo build --release` takes ~10–15+ minutes (downloads crates + LTO). Subsequent builds are much faster.

### Critical: run router on the **host**, not in the vLLM image

The release binary links host `libssl.so.1.1`. The Ascend vLLM container often lacks it → `error while loading shared libraries: libssl.so.1.1`.

`qwen06_pd_patent_1host.sh` already starts the router with `nohup` on the host. Do not wrap it in `nerdctl run` unless you mount matching OpenSSL libs.

---

## 5. Day-to-day commands (on host 110)

Script: `/root/gyh_dev/vllm_router/scripts/910b3/qwen06_pd_patent_1host.sh`

```bash
cd /root/gyh_dev/vllm_router

# Stop patent containers + host router
bash scripts/910b3/qwen06_pd_patent_1host.sh stop

# Start differentiated workers only
bash scripts/910b3/qwen06_pd_patent_1host.sh serve
bash scripts/910b3/qwen06_pd_patent_1host.sh wait 900
bash scripts/910b3/qwen06_pd_patent_1host.sh status

# Router arms: A0 | Hard | SoftFull | SoftPlusDecode
bash scripts/910b3/qwen06_pd_patent_1host.sh router SoftFull

# Full bring-up then SoftPlusDecode router
bash scripts/910b3/qwen06_pd_patent_1host.sh all

# Ablation (rewrites logs/patent_live/SUMMARY.md)
RATE=1.5 FINAL_N=40 bash scripts/910b3/qwen06_pd_patent_1host.sh ablation
```

From WSL (long job):

```bash
export SSH_PASS='…'
python3 scripts/_tmp_scp_upload.py scripts/_remote_cmd_....sh 39.156.164.110 /tmp/cmd.sh
bash scripts/_wsl_long_run.sh 39.156.164.110 'bash /tmp/cmd.sh' 3600
```

### Ablation arms (what the script configures)

| Arm | Prefill policy | Decode policy |
|-----|----------------|---------------|
| A0 | `round_robin` | `round_robin` |
| Hard | `prefill_bucket` + `--prefill-hard-partition` + spill | `round_robin` |
| SoftFull | `prefill_bucket` + `--prefill-preference-weight 2.0` | `round_robin` |
| SoftPlusDecode | same SoftFull Prefill | `decode_kv_cost` |

---

## 6. Client / load generator

- Binary entry: `real_pd_trajectory_client.py`
- Defaults are **wrong** for this small-model setup (`MODEL_NAME` defaults to `qwen3.8`). The serve script injects:

```text
MODEL_NAME=qwen3-0.6b
TOKENIZER_PATH=/home/weights/Qwen3-0.6B
ROUTER_URL=http://127.0.0.1:8090/v1/chat/completions
SHORT_PORTS=18001,18002
```

- Client runs **inside** the vLLM image with `--net=host` so it reaches `:8090` and workers.
- Smoke check:

```bash
curl -s http://127.0.0.1:8090/v1/models
curl -s http://127.0.0.1:8090/health
# chat completion with model=qwen3-0.6b should return 200
```

Wrong `MODEL_NAME` → HTTP 404. Long prompt routed to short `@2048` under RR → HTTP 400 (expected; Hard/Soft should avoid).

---

## 7. Artifacts & reporting

| Location | Contents |
|----------|----------|
| Remote `logs/patent_live/` | `SUMMARY.md`, `metrics.csv`, `{Arm}_r1.5.json`, router logs |
| Repo `docs/patent_ablation_results/live_110_qwen06/` | Copied SUMMARY / METHOD / JSON for git |
| Repo `docs/patent_ablation_results/` | Device-free sim results (do not mix wording with live) |

When writing disclosure text: live results are **supporting**; phrase carefully (“实测可验证…”) and keep sim tables labeled device-free.

---

## 8. Known failure modes (checklist)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| SSH `Permission denied` | Bad/truncated `SSH_PASS` | Re-confirm with operator; check quoting |
| SSH empty output | Password wrong or PowerShell ate the remote cmd | Use uploaded script files |
| `libssl.so.1.1` missing | Router run inside container | Host-native `nohup` (script default) |
| All requests 404 | `MODEL_NAME` ≠ served name | Export `MODEL_NAME=qwen3-0.6b` |
| Workers never healthy | First torch.compile / weight load | Wait up to ~5–15 min; `wait` polls `/health` |
| Disk 100% on a host | Logs / leftover images | Prefer `110` (had free space); clean before clone |
| `git clone` HTTP/2 fail | Lab network | Use tarball + scp |
| SoftPlusDecode ≈ SoftFull | Decode pool too symmetric / light load | Heavier Decode asymmetry or rely on sim Exp2 |
| Tar “file changed as we read it” | Writing archive into same tree | Output tarball outside repo |

---

## 9. Etiquette / safety

- Authorized lab access only; stop containers with `nerdctl` when reclaiming NPUs (`pd-pat-*` prefix is this experiment’s namespace).
- Prefer rotating lab passwords after shared-agent use; never paste secrets into commits, PR bodies, or tracked docs.
- Leave workers running only if the next agent needs them; otherwise `…/qwen06_pd_patent_1host.sh stop`.
- Multi-host (107–109) is optional expansion; start from validated `110` topology unless disk/NPU forces a move.

---

## 10. Minimal “cold agent” path

1. Confirm `SSH_PASS` from operator; probe `110` disk + `npu-smi` + `nerdctl`.
2. Sync current `feat/soft-length-aware` (or target branch) tree → `/root/gyh_dev/vllm_router`.
3. Ensure `/root/gyh_dev/vllm_router/vllm-router` exists (build if missing).
4. `bash scripts/910b3/qwen06_pd_patent_1host.sh all` then `RATE=1.5 FINAL_N=40 … ablation`.
5. Copy `logs/patent_live/` → `docs/patent_ablation_results/live_110_qwen06/` for the repo.
6. Do **not** claim live gains as the only patent evidence; cross-link sim SUMMARY.
