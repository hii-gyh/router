#!/bin/bash
# Remeasure soft-preference prefill_bucket (+ decode_kv_cost) vs round_robin
# on the live real-PD 2+2P/4D topology. Same trajectory seed/windows as prior A/B.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$ROOT/data/trajectory.json
AB_DIR=$LOGDIR/real_pd_soft_ab
RESULT=$LOGDIR/real_pd_soft_remeasure.json
SUMMARY=$LOGDIR/real_pd_soft_remeasure.txt
mkdir -p "$LOGDIR" "$AB_DIR"

SUCCESS_JSON='{"at":"R_sat","R_sat":8.0,"rates":[4.0,6.4,8.0],"require":["experiment.output_tok_s > control.output_tok_s","experiment.ttft_p90 <= control.ttft_p90"],"policy":"soft preference_weight=2.0 hard_partition=false","note":"Same trajectory seed 20251031; fixed rates from prior hard-bucket A/B. Soft preference is the intervention under test."}'
echo "$SUCCESS_JSON" > "$LOGDIR/real_pd_soft_success_criteria.json"
echo "SOFT REMEASURE $(date -Is)" | tee "$SUMMARY"
echo "$SUCCESS_JSON" | tee -a "$SUMMARY"

FINAL_N="${FINAL_N:-60}"
WARM_N="${WARM_N:-20}"
# Match prior remesaure offsets used after spill/E1 where possible.
OFFSETS=(440 500 560)
RATES=(4.0 6.4 8.0)

workers_ok() {
  local p ok=0
  for p in 18001 18002 18003 18004 18011 18012 18013 18014; do
    curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && ok=$((ok+1)) || true
  done
  echo "workers_healthy=${ok}/8"
  [[ "$ok" -eq 8 ]]
}

host_client_idle() {
  python3 - <<'PY'
import json, urllib.request, time
DECODE=[18011,18012,18013,18014]
def snap():
  out={}
  for p in DECODE:
    try:
      t=urllib.request.urlopen(f'http://127.0.0.1:{p}/metrics',timeout=3).read().decode()
    except Exception:
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
    print(json.dumps({'idle':True})); raise SystemExit(0)
  time.sleep(2)
print(json.dumps({'idle':False,'metrics':snap()})); raise SystemExit(1)
PY
}

run_client() {
  nerdctl run --rm --net=host \
    -v "$ROOT:$ROOT" \
    -v /mnt/weights:/home/weights \
    -e TOKENIZER_PATH=/home/weights/Qwen3.8-27B-w8a8/ \
    "$IMG" \
    python3 "$CLIENT" "$@"
}

restart_router() {
  local arm="$1"
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
    # Soft preference is the binary default (hard_partition=false).
    # Explicit weight=2.0; do NOT pass --prefill-hard-partition.
    # Bucket membership: 18001/18002 short tier 0; 18003/18004 open.
    args+=(
      --prefill-policy prefill_bucket
      --decode-policy decode_kv_cost
      --prefill-bucket-max-tokens 2048
      --prefill-preference-weight 2.0
      --prefill-bucket-worker http://127.0.0.1:18001=0
      --prefill-bucket-worker http://127.0.0.1:18002=0
    )
  fi
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready arm=$arm (${i}s)"
      # Confirm soft flags appear in process/logs when experiment arm.
      if [[ "$arm" == "bucket" ]]; then
        nerdctl logs pd-real-router 2>&1 | grep -E 'preference|hard_partition|PrefillBucket|prefill_bucket' | head -5 || true
      fi
      return 0
    fi
    sleep 1
  done
  echo "router failed arm=$arm" >&2
  nerdctl logs --tail 80 pd-real-router 2>&1 | tail -80 >&2
  return 1
}

ensure_binary_has_soft() {
  local hit
  hit=$(strings "$ROOT/vllm-router" 2>/dev/null | grep -E 'prefill_preference_weight|prefill-preference-weight' | head -1 || true)
  if [[ -z "$hit" ]]; then
    echo "FATAL: vllm-router binary lacks soft-preference flag; rebuild first" | tee -a "$SUMMARY"
    exit 5
  fi
  echo "binary soft-preference flag present: $hit" | tee -a "$SUMMARY"
  md5sum "$ROOT/vllm-router" | tee -a "$SUMMARY"
}

[[ -f "$TRAJ" ]] || { echo "missing $TRAJ"; exit 2; }
[[ -x "$ROOT/vllm-router" ]] || { echo "missing binary"; exit 2; }
ensure_binary_has_soft
workers_ok || { echo "FATAL workers"; exit 3; }

run_arm() {
  local arm="$1" name="$2"
  restart_router "$arm"
  host_client_idle || true
  run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
  host_client_idle || true
  local i
  for i in 0 1 2; do
    local r=${RATES[$i]} o=${OFFSETS[$i]}
    workers_ok || exit 3
    echo "=== arm=$name rate=$r offset=$o $(date -Is) ===" | tee -a "$SUMMARY"
    local out="$AB_DIR/${arm}_rate_${r}.json"
    run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset "$o" \
      --label "${name}_${r}" --out "$out" | tee "$AB_DIR/${arm}_rate_${r}.log" | tail -12
    host_client_idle || true
  done
}

echo "=== CONTROL round_robin $(date -Is) ===" | tee -a "$SUMMARY"
run_arm rr round_robin

echo "=== EXPERIMENT soft prefill_bucket+decode_kv_cost $(date -Is) ===" | tee -a "$SUMMARY"
run_arm soft soft_bucket_kv

python3 - <<'PY'
import json, pathlib
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root / 'real_pd_soft_ab'
success = json.loads((root/'real_pd_soft_success_criteria.json').read_text())
rates = [4.0, 6.4, 8.0]
rows = []
for r in rates:
    rr = json.loads((ab/f'rr_rate_{r}.json').read_text())
    ex = json.loads((ab/f'soft_rate_{r}.json').read_text())
    tok_win = (ex.get('output_tok_s') or 0) > (rr.get('output_tok_s') or 0)
    ttft_ok = (ex.get('ttft_p90') or 9e9) <= (rr.get('ttft_p90') or 0)
    rows.append({
        'rate': r,
        'rr_tok_s': rr.get('output_tok_s'),
        'ex_tok_s': ex.get('output_tok_s'),
        'rr_ttft_p90': rr.get('ttft_p90'),
        'ex_ttft_p90': ex.get('ttft_p90'),
        'rr_ttft_p50': rr.get('ttft_p50'),
        'ex_ttft_p50': ex.get('ttft_p50'),
        'rr_short_p90': (rr.get('buckets') or {}).get('short_le_2048', {}).get('ttft_p90'),
        'ex_short_p90': (ex.get('buckets') or {}).get('short_le_2048', {}).get('ttft_p90'),
        'rr_long_p90': (rr.get('buckets') or {}).get('long_gt_2048', {}).get('ttft_p90'),
        'ex_long_p90': (ex.get('buckets') or {}).get('long_gt_2048', {}).get('ttft_p90'),
        'ex_short_pool': (ex.get('prefill_prompt_delta') or {}).get('short_pool'),
        'ex_open_pool': (ex.get('prefill_prompt_delta') or {}).get('open_pool'),
        'tok_win': tok_win,
        'ttft_ok': ttft_ok,
        'pass': tok_win and ttft_ok,
        'rr_err': rr.get('error_rate'),
        'ex_err': ex.get('error_rate'),
    })
rsat = 8.0
rsat_row = next(x for x in rows if x['rate'] == rsat)
out = {
    'success_criteria': success,
    'intervention': 'soft preference_weight=2.0, hard_partition=false, 2 short@4096 + 2 open@16384',
    'passed_at_R_sat': bool(rsat_row['pass']),
    'rows': rows,
}
(root/'real_pd_soft_remeasure.json').write_text(json.dumps(out, indent=2))
lines = [
    f"passed_at_R_sat={out['passed_at_R_sat']}",
    'rate | RR tok/s | soft tok/s | RR p90 | soft p90 | short/open | PASS',
]
for x in rows:
    lines.append(
        f"{x['rate']} | {x['rr_tok_s']:.1f} | {x['ex_tok_s']:.1f} | {x['rr_ttft_p90']:.3f} | {x['ex_ttft_p90']:.3f} | "
        f"{x['ex_short_pool']}/{x['ex_open_pool']} | {x['pass']}"
    )
text = '\n'.join(lines) + '\n'
print(text)
(root/'real_pd_soft_remeasure.txt').write_text(
    (root/'real_pd_soft_remeasure.txt').read_text() + '\n' + text
)
PY

echo "=== soft remesaure done $(date -Is) ===" | tee -a "$SUMMARY"
cat "$RESULT" | tee -a "$SUMMARY"
exit 0
