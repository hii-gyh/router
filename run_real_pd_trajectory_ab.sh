#!/bin/bash
# Real-PD trajectory A/B: round_robin+round_robin vs prefill_bucket+decode_kv_cost
# Do NOT use mock PD. Workers stay up; only the router restarts between arms.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
DATADIR=$ROOT/data
CLIENT=$ROOT/real_pd_trajectory_client.py
RESULT=$LOGDIR/real_pd_trajectory_ab.json
SUMMARY=$LOGDIR/real_pd_trajectory_ab.txt
TRAJ=$DATADIR/trajectory.json
mkdir -p "$LOGDIR" "$DATADIR"

# ---- Locked success criteria (written before any timed arm) ----
SUCCESS_JSON='{"at":"R_sat","require":["experiment.output_tok_s > control.output_tok_s","experiment.ttft_p90 <= control.ttft_p90"],"note":"Per-length-bucket regressions stay in the report. SYSTEM output tok/s + TTFT p90 are the headline metrics."}'
echo "$SUCCESS_JSON" > "$LOGDIR/real_pd_trajectory_success_criteria.json"
echo "SUCCESS CRITERIA LOCKED at $(date -Is)" | tee "$SUMMARY"
echo "$SUCCESS_JSON" | tee -a "$SUMMARY"

disk_ok() {
  local avail
  avail=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
  echo "disk_free_G=$avail"
  if [[ "${avail:-0}" -lt 20 ]]; then
    echo "STOP: disk free ${avail}G < 20G" | tee -a "$SUMMARY"
    exit 2
  fi
}

workers_ok() {
  local p ok=0
  for p in 18001 18002 18003 18004 18011 18012 18013 18014; do
    if curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
      ok=$((ok+1))
    fi
  done
  echo "workers_healthy=${ok}/8"
  if [[ "$ok" -ne 8 ]]; then
    echo "workers down; attempting relaunch via qwen38_pd_real_1host8.sh" | tee -a "$SUMMARY"
    bash "$ROOT/qwen38_pd_real_1host8.sh" status || true
    # Bring workers back without touching vllm-ascend-* 
    # Only restart missing pd-real workers via full serve is heavy; prefer status report.
    bash "$ROOT/qwen38_pd_real_1host8.sh" all || true
    sleep 5
    ok=0
    for p in 18001 18002 18003 18004 18011 18012 18013 18014; do
      curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && ok=$((ok+1)) || true
    done
    echo "workers_healthy_after=${ok}/8"
    [[ "$ok" -eq 8 ]] || { echo "FATAL: workers not healthy"; exit 3; }
  fi
}

restart_router() {
  local arm="$1"  # rr | bucket
  nerdctl rm -f pd-real-router >/dev/null 2>&1 || true
  local -a args=(
    "$ROOT/vllm-router"
    --host 0.0.0.0 --port 8090
    --vllm-pd-disaggregation --kv-connector nixl
    --prefill http://127.0.0.1:18001 30000
    --prefill http://127.0.0.1:18002 30001
    --prefill http://127.0.0.1:18003 30002
    --prefill http://127.0.0.1:18004 30003
    --decode http://127.0.0.1:18011
    --decode http://127.0.0.1:18012
    --decode http://127.0.0.1:18013
    --decode http://127.0.0.1:18014
    --disable-retries --disable-circuit-breaker
  )
  if [[ "$arm" == "rr" ]]; then
    args+=(--prefill-policy round_robin --decode-policy round_robin)
  else
    # Spill earlier under load: defaults (abs=8, rel=1.5) never fire with
    # max-num-seqs=8 while short pool is hot and open idle — that drove the
    # system/short TTFT p90 regression. abs=2, rel=1.0 spills as soon as the
    # lightest short worker is strictly busier than the lightest open worker.
    args+=(
      --prefill-policy prefill_bucket
      --decode-policy decode_kv_cost
      --prefill-bucket-max-tokens 2048
      --prefill-spill-abs 2
      --prefill-spill-rel 1.0
      --prefill-bucket-worker http://127.0.0.1:18001=0
      --prefill-bucket-worker http://127.0.0.1:18002=0
    )
  fi
  nerdctl run -d --name pd-real-router --net=host \
    -v "$ROOT:$ROOT" \
    "$IMG" \
    "${args[@]}" >/dev/null
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready arm=$arm (${i}s)"
      return 0
    fi
    sleep 1
  done
  echo "router failed arm=$arm" >&2
  nerdctl logs --tail 60 pd-real-router 2>&1 | tail -60 >&2
  return 1
}

run_client() {
  # Run inside ascend image for transformers + tokenizer weights.
  nerdctl run --rm --net=host \
    -v "$ROOT:$ROOT" \
    -v /mnt/weights:/home/weights \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e TOKENIZER_PATH=/home/weights/Qwen3.8-27B-w8a8/ \
    "$IMG" \
    python3 "$CLIENT" "$@"
}

host_client_idle() {
  # Idle wait does not need tokenizer; run on host for speed / nerdctl logs access.
  python3 - <<'PY'
import json, urllib.request, time
DECODE=[18011,18012,18013,18014]
def snap():
  out={}
  for p in DECODE:
    try:
      t=urllib.request.urlopen(f'http://127.0.0.1:{p}/metrics',timeout=3).read().decode()
    except Exception as e:
      out[p]={'running':99,'waiting':99}; continue
    run=wait=0.0
    for line in t.splitlines():
      if line.startswith('vllm:num_requests_running{'): run=float(line.split()[-1])
      elif line.startswith('vllm:num_requests_waiting{') and 'reason' not in line: wait=float(line.split()[-1])
    out[p]={'running':run,'waiting':wait}
  return out
t0=time.time()
while time.time()-t0 < 240:
  m=snap(); r=sum(v['running'] for v in m.values()); w=sum(v['waiting'] for v in m.values())
  if r<=0.01 and w<=0.01:
    print(json.dumps({'idle':True,'metrics':m})); raise SystemExit(0)
  time.sleep(2)
print(json.dumps({'idle':False,'metrics':snap()})); raise SystemExit(1)
PY
}

try_download_sharegpt() {
  local dest="$DATADIR/sharegpt.json"
  if [[ -f "$dest" ]]; then
    echo "sharegpt already present: $dest"
    return 0
  fi
  echo "attempting ShareGPT download..."
  # Modest subset mirrors; any failure falls back to synthetic.
  if command -v curl >/dev/null; then
    curl -L --max-time 90 -o "$dest.partial" \
      "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
      && mv "$dest.partial" "$dest" && echo "downloaded ShareGPT" && return 0 || true
    rm -f "$dest.partial"
  fi
  echo "ShareGPT download blocked/failed; will synthesize"
  return 1
}

disk_ok
workers_ok

# Prepare trajectory (download or synthetic)
try_download_sharegpt || true
# prepare uses transformers if sharegpt present; synthetic path does not need it on host
if [[ -f "$DATADIR/sharegpt.json" ]]; then
  run_client --mode prepare --traj "$TRAJ" --sharegpt "$DATADIR/sharegpt.json" | tee -a "$SUMMARY"
else
  # host python enough for synthetic prepare
  python3 "$CLIENT" --mode prepare --traj "$TRAJ" --sharegpt "$DATADIR/sharegpt.json" | tee -a "$SUMMARY"
fi

PROBE_N="${PROBE_N:-40}"
FINAL_N="${FINAL_N:-60}"
WARM_N="${WARM_N:-20}"

echo "=== R_sat search with round_robin $(date -Is) ===" | tee -a "$SUMMARY"
restart_router rr
host_client_idle || true
run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
host_client_idle || true

# Probe rates (req/s). Find where TTFT p90 climbs sharply.
# Prior 0.4-2.0 sweep stayed flat (~0.5-0.85s); push higher to find the knee.
RATES_PROBE=(1.0 2.0 3.0 4.0 5.0 6.5 8.0)
PROBE_DIR=$LOGDIR/real_pd_probe
mkdir -p "$PROBE_DIR"
# Reuse prior low-rate probes if present; only run missing/higher ones.
offset=0
# Reserve offsets: keep deterministic windows. Start after any prior 5*40=200 if old probes exist.
if [[ -f "$PROBE_DIR/rr_rate_2.0.json" ]]; then
  offset=200
fi
for r in "${RATES_PROBE[@]}"; do
  out="$PROBE_DIR/rr_rate_${r}.json"
  if [[ -f "$out" ]]; then
    echo "--- reuse probe rate=$r ---" | tee -a "$SUMMARY"
    continue
  fi
  workers_ok
  disk_ok
  echo "--- probe rate=$r offset=$offset ---" | tee -a "$SUMMARY"
  run_client --mode run --traj "$TRAJ" --rate "$r" --n "$PROBE_N" --offset "$offset" \
    --label "probe_rr_${r}" --out "$out" | tee "$PROBE_DIR/rr_rate_${r}.log" | tail -5
  offset=$((offset + PROBE_N))
  host_client_idle || true
done

# Choose R_sat: first rate where ttft_p90 climbs sharply vs low-load baseline.
# Print ONLY the numeric R_sat on stdout (debug on stderr).
R_SAT=$(python3 - <<'PY'
import json, pathlib, sys
rates=[]
for p in sorted(pathlib.Path('/root/gyh_dev/vllm_router/logs/real_pd_probe').glob('rr_rate_*.json')):
    d=json.loads(p.read_text())
    rates.append((float(d['rate_rps']), d.get('ttft_p90'), d.get('output_tok_s'), d.get('error_rate') or 0.0))
rates.sort()
print('PROBE_ROWS', rates, file=sys.stderr)
if not rates:
    print(2.0)
    raise SystemExit
# baseline = median of the lowest 2 healthy p90s
healthy=[(r,p90,ots,err) for r,p90,ots,err in rates if p90 is not None and err < 0.2]
if not healthy:
    print(rates[-1][0]); raise SystemExit
base=sorted(p90 for _,p90,_,_ in healthy[: max(2, min(3,len(healthy))) ])[0]
rsat=None
for r,p90,ots,err in healthy:
    if p90 >= max(base*1.6, base+1.0) or p90 >= 5.0 or err >= 0.1:
        rsat=r
        break
if rsat is None:
    # No TTFT knee: use throughput peak as saturation proxy and note it.
    peak=max(healthy, key=lambda x: (x[2] or 0))
    # pick first rate at/after peak where ots drops >10% or last
    peak_ots=peak[2] or 0
    rsat=peak[0]
    for r,p90,ots,err in healthy:
        if r < peak[0]:
            continue
        if ots is not None and peak_ots and ots < 0.9*peak_ots:
            rsat=r
            break
    else:
        rsat=healthy[-1][0]
    print(f'NO_TTFT_KNEE baseline={base} peak_ots_rate={peak[0]} -> R_sat={rsat}', file=sys.stderr)
else:
    print(f'TTFT_KNEE baseline={base} -> R_sat={rsat}', file=sys.stderr)
print(rsat)
PY
)
echo "R_sat=$R_SAT" | tee -a "$SUMMARY"
if ! [[ "$R_SAT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "FATAL: invalid R_sat='$R_SAT'" | tee -a "$SUMMARY"
  exit 4
fi

# Final A/B rates
mapfile -t FINAL_RATES < <(python3 - <<PY
rsat=float("$R_SAT")
for m in (0.5, 0.8, 1.0):
    print(round(rsat * m, 4))
PY
)

echo "FINAL_RATES=${FINAL_RATES[*]}" | tee -a "$SUMMARY"
if [[ ${#FINAL_RATES[@]} -lt 3 ]]; then
  echo "FATAL: FINAL_RATES empty" | tee -a "$SUMMARY"
  exit 4
fi

AB_DIR=$LOGDIR/real_pd_ab
mkdir -p "$AB_DIR"
# Fixed request windows so both arms see identical order/content.
# Use disjoint offsets per rate, shared across arms.
declare -A RATE_OFFSET
off=$offset
for r in "${FINAL_RATES[@]}"; do
  RATE_OFFSET[$r]=$off
  off=$((off + FINAL_N))
done

run_arm() {
  local arm="$1"  # rr | bucket
  local policy_name="$2"
  restart_router "$arm"
  host_client_idle || true
  run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
  host_client_idle || true
  for r in "${FINAL_RATES[@]}"; do
    workers_ok
    disk_ok
    local o=${RATE_OFFSET[$r]}
    echo "=== arm=$policy_name rate=$r offset=$o $(date -Is) ===" | tee -a "$SUMMARY"
    local out="$AB_DIR/${arm}_rate_${r}.json"
    run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset "$o" \
      --label "${policy_name}_${r}" --out "$out" | tee "$AB_DIR/${arm}_rate_${r}.log" | tail -8
    host_client_idle || true
  done
}

echo "=== CONTROL round_robin $(date -Is) ===" | tee -a "$SUMMARY"
run_arm rr round_robin

echo "=== EXPERIMENT prefill_bucket+decode_kv_cost $(date -Is) ===" | tee -a "$SUMMARY"
run_arm bucket bucket_kv

# Aggregate
python3 - <<'PY'
import json, pathlib, statistics
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root / 'real_pd_ab'
probe = root / 'real_pd_probe'
success = json.loads((root/'real_pd_trajectory_success_criteria.json').read_text())

def load_arm(prefix):
    rows={}
    for p in sorted(ab.glob(f'{prefix}_rate_*.json')):
        d=json.loads(p.read_text())
        rows[str(d['rate_rps'])]=d
    return rows

rr=load_arm('rr')
bucket=load_arm('bucket')
# recover R_sat from summary rates
rates=sorted({float(k) for k in list(rr)+list(bucket)})
# R_sat is the 1.0x rate = max of final rates
rsat = max(rates) if rates else None

def pass_at(rate):
    a=rr.get(str(rate)) or rr.get(str(float(rate)))
    # key might be 0.8 vs 0.8000
    if a is None:
        for k,v in rr.items():
            if abs(float(k)-float(rate))<1e-6: a=v; break
    b=None
    for k,v in bucket.items():
        if abs(float(k)-float(rate))<1e-6: b=v; break
    if not a or not b:
        return None
    ots_ok = (b.get('output_tok_s') or 0) > (a.get('output_tok_s') or 0)
    p90a, p90b = a.get('ttft_p90'), b.get('ttft_p90')
    ttft_ok = (p90a is not None and p90b is not None and p90b <= p90a)
    return {
        'rate': rate,
        'control_output_tok_s': a.get('output_tok_s'),
        'experiment_output_tok_s': b.get('output_tok_s'),
        'control_ttft_p90': p90a,
        'experiment_ttft_p90': p90b,
        'output_tok_s_win': ots_ok,
        'ttft_p90_not_worse': ttft_ok,
        'pass': bool(ots_ok and ttft_ok),
        'control_buckets': a.get('buckets'),
        'experiment_buckets': b.get('buckets'),
        'control_prefill_prompt_delta': a.get('prefill_prompt_delta'),
        'experiment_prefill_prompt_delta': b.get('prefill_prompt_delta'),
        'control_decode_kv': a.get('decode_kv'),
        'experiment_decode_kv': b.get('decode_kv'),
        'control_error_rate': a.get('error_rate'),
        'experiment_error_rate': b.get('error_rate'),
        'control_tpot_p50': a.get('tpot_p50'),
        'control_tpot_p90': a.get('tpot_p90'),
        'experiment_tpot_p50': b.get('tpot_p50'),
        'experiment_tpot_p90': b.get('tpot_p90'),
        'control_n_ok': a.get('n_ok'),
        'experiment_n_ok': b.get('n_ok'),
        'kv_transfer_samples_experiment': (b.get('kv_transfer_samples') or [])[:12],
    }

table=[]
for r in rates:
    table.append(pass_at(r))

sat_row=None
for row in table:
    if row and rsat is not None and abs(row['rate']-rsat)<1e-6:
        sat_row=row

probe_rows=[]
for p in sorted(probe.glob('rr_rate_*.json')):
    d=json.loads(p.read_text())
    probe_rows.append({
        'rate': d.get('rate_rps'),
        'ttft_p50': d.get('ttft_p50'),
        'ttft_p90': d.get('ttft_p90'),
        'output_tok_s': d.get('output_tok_s'),
        'error_rate': d.get('error_rate'),
    })

out={
    'success_criteria': success,
    'R_sat': rsat,
    'probe_rr': probe_rows,
    'table': table,
    'passed_at_R_sat': bool(sat_row and sat_row['pass']),
    'topology': 'real_pd_mooncake_8worker',
    'mock_pd': False,
}
(root/'real_pd_trajectory_ab.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
lines=[]
lines.append(f"R_sat={rsat}")
lines.append(f"passed_at_R_sat={out['passed_at_R_sat']}")
lines.append('rate policy output_tok/s ttft_p50 ttft_p90 tpot_p90 err prefill_short prefill_open kv_peak_max')
for r in rates:
    for prefix,name in [('rr','round_robin'),('bucket','bucket_kv')]:
        # find file
        best=None
        for p in ab.glob(f'{prefix}_rate_*.json'):
            d=json.loads(p.read_text())
            if abs(float(d['rate_rps'])-float(r))<1e-6:
                best=d; break
        if not best: continue
        pd=best.get('prefill_prompt_delta') or {}
        kv=best.get('decode_kv') or {}
        lines.append(
            f"{r} {name} {best.get('output_tok_s')} {best.get('ttft_p50')} {best.get('ttft_p90')} "
            f"{best.get('tpot_p90')} {best.get('error_rate')} {pd.get('short_pool')} {pd.get('open_pool')} {kv.get('kv_peak_max')}"
        )
    row=pass_at(r)
    if row:
        lines.append(f"  cmp@{r}: ots_win={row['output_tok_s_win']} ttft_ok={row['ttft_p90_not_worse']} PASS={row['pass']}")
(root/'real_pd_trajectory_ab.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print('WROTE', root/'real_pd_trajectory_ab.json')
PY

echo "=== DONE $(date -Is) ===" | tee -a "$SUMMARY"
# restore default router to bucket policy
restart_router bucket || true
exit 0
