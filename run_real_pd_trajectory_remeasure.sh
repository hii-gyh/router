#!/bin/bash
# Remeasure-only: same trajectory seed/offsets, rates 4.0 / 6.4 / 8.0, N=60.
# Intervention B: earlier short→open spill (abs=2, rel=1.0). No mock PD.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
DATADIR=$ROOT/data
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$DATADIR/trajectory.json
RESULT=$LOGDIR/real_pd_trajectory_ab_remeasure.json
SUMMARY=$LOGDIR/real_pd_trajectory_ab_remeasure.txt
DIAG=$LOGDIR/real_pd_trajectory_ab_diagnosis.md
AB_DIR=$LOGDIR/real_pd_ab_remeasure
mkdir -p "$LOGDIR" "$AB_DIR"

FINAL_N="${FINAL_N:-60}"
WARM_N="${WARM_N:-20}"
# Same windows as the failed locked run (seed 20251031).
declare -A RATE_OFFSET=(["4.0"]=440 ["6.4"]=500 ["8.0"]=560)
FINAL_RATES=(4.0 6.4 8.0)
R_SAT=8.0

SUCCESS_JSON='{"at":"R_sat","require":["experiment.output_tok_s > control.output_tok_s","experiment.ttft_p90 <= control.ttft_p90"],"note":"Per-length-bucket regressions stay in the report. SYSTEM output tok/s + TTFT p90 are the headline metrics."}'
echo "$SUCCESS_JSON" > "$LOGDIR/real_pd_trajectory_success_criteria.json"

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
    echo "FATAL: workers not healthy ${ok}/8 (will not relaunch full serve)" | tee -a "$SUMMARY"
    exit 3
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
  nerdctl run --rm --net=host \
    -v "$ROOT:$ROOT" \
    -v /mnt/weights:/home/weights \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e TOKENIZER_PATH=/home/weights/Qwen3.8-27B-w8a8/ \
    "$IMG" \
    python3 "$CLIENT" "$@"
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
    print(json.dumps({'idle':True,'metrics':m})); raise SystemExit(0)
  time.sleep(2)
print(json.dumps({'idle':False,'metrics':snap()})); raise SystemExit(1)
PY
}

: > "$SUMMARY"
echo "REMEASURE spill_abs=2 spill_rel=1.0 $(date -Is)" | tee -a "$SUMMARY"
echo "$SUCCESS_JSON" | tee -a "$SUMMARY"

disk_ok
workers_ok
[[ -f "$TRAJ" ]] || { echo "FATAL: missing $TRAJ"; exit 4; }

run_arm() {
  local arm="$1"
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

echo "=== EXPERIMENT prefill_bucket+decode_kv_cost spill2/1.0 $(date -Is) ===" | tee -a "$SUMMARY"
run_arm bucket bucket_kv_spill2

python3 - <<'PY'
import json, pathlib
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root / 'real_pd_ab_remeasure'
success = json.loads((root/'real_pd_trajectory_success_criteria.json').read_text())
rsat = 8.0
rates = [4.0, 6.4, 8.0]

def load_arm(prefix):
    rows = {}
    for p in sorted(ab.glob(f'{prefix}_rate_*.json')):
        d = json.loads(p.read_text())
        rows[float(d['rate_rps'])] = d
    return rows

rr = load_arm('rr')
bucket = load_arm('bucket')

def pass_at(rate):
    a = rr.get(rate)
    b = bucket.get(rate)
    if not a or not b:
        return None
    ots_ok = (b.get('output_tok_s') or 0) > (a.get('output_tok_s') or 0)
    p90a, p90b = a.get('ttft_p90'), b.get('ttft_p90')
    ttft_ok = p90a is not None and p90b is not None and p90b <= p90a
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
        'control_ttft_p50': a.get('ttft_p50'),
        'experiment_ttft_p50': b.get('ttft_p50'),
    }

table = [pass_at(r) for r in rates]
sat_row = next((row for row in table if row and abs(row['rate'] - rsat) < 1e-6), None)
out = {
    'success_criteria': success,
    'R_sat': rsat,
    'intervention': {
        'id': 'B',
        'name': 'earlier_short_to_open_spill',
        'prefill_spill_abs': 2,
        'prefill_spill_rel': 1.0,
        'topology_unchanged': '2 short(4096) + 2 open(16384) + 4 decode',
        'note': 'Defaults abs=8/rel=1.5 never spilled under max-num-seqs=8; short pool absorbed almost all short traffic while open idled.',
    },
    'table': table,
    'passed_at_R_sat': bool(sat_row and sat_row['pass']),
    'topology': 'real_pd_mooncake_8worker',
    'mock_pd': False,
    'seed': 20251031,
    'offsets': {'4.0': 440, '6.4': 500, '8.0': 560},
    'n_per_rate': 60,
    'arms': {
        **{f'rr_rate_{r}': rr[r] for r in rates if r in rr},
        **{f'bucket_rate_{r}': bucket[r] for r in rates if r in bucket},
    },
}
(root/'real_pd_trajectory_ab_remeasure.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
# Also refresh primary JSON pointer used by canvas consumers.
(root/'real_pd_trajectory_ab.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))

lines = [
    f"R_sat={rsat}",
    f"passed_at_R_sat={out['passed_at_R_sat']}",
    'intervention=B earlier_spill abs=2 rel=1.0',
    'rate policy output_tok/s ttft_p50 ttft_p90 tpot_p90 err short_ttft_p90 long_ttft_p90 prefill_short prefill_open kv_peak',
]
for r in rates:
    for prefix, name in [('rr', 'round_robin'), ('bucket', 'bucket_kv_spill2')]:
        d = (rr if prefix == 'rr' else bucket).get(r)
        if not d:
            continue
        b = d.get('buckets') or {}
        s = (b.get('short_le_2048') or {}).get('ttft_p90')
        lg = (b.get('long_gt_2048') or {}).get('ttft_p90')
        pd = d.get('prefill_prompt_delta') or {}
        kv = d.get('decode_kv') or {}
        lines.append(
            f"{r} {name} {d.get('output_tok_s')} {d.get('ttft_p50')} {d.get('ttft_p90')} "
            f"{d.get('tpot_p90')} {d.get('error_rate')} {s} {lg} "
            f"{pd.get('short_pool')} {pd.get('open_pool')} {kv.get('kv_peak_max')}"
        )
    row = pass_at(r)
    if row:
        lines.append(
            f"  cmp@{r}: ots_win={row['output_tok_s_win']} ttft_ok={row['ttft_p90_not_worse']} PASS={row['pass']}"
        )
(root/'real_pd_trajectory_ab_remeasure.txt').write_text('\n'.join(lines) + '\n')
(root/'real_pd_trajectory_ab.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('WROTE', root/'real_pd_trajectory_ab_remeasure.json')
PY

# Diagnosis note (also written before run start content is appended here after aggregate)
python3 - <<'PY'
from pathlib import Path
import json
root = Path('/root/gyh_dev/vllm_router/logs')
d = json.loads((root/'real_pd_trajectory_ab_remeasure.json').read_text())
sat = next(r for r in d['table'] if abs(r['rate']-8.0)<1e-9)
text = f"""# Real-PD trajectory A/B diagnosis + remesaure

## Root cause (failed locked run)
- Prefill-side regression, not decode. Decode KV peaks stayed ~9–13% both arms; waiting≈0; TPOT p90 matched. Remaining-output-load would NOT have fixed this failure.
- Bucket steered ≤2048 prompts onto 2 short workers (4096) while RR spread them across 4 prefill workers. At R_sat=8.0 experiment short_pool=33204 vs open=6739 (18003=0); RR was ~22686/17257 balanced.
- System TTFT p90 is short-dominated (58/60 shorts). Short p90 also regresses at 4.0/6.4/8.0; long cell is worst (2.162 vs 1.073) because rare longs are confined to 2 open workers (often one).
- Default spill (abs=8, rel=1.5) cannot fire usefully with max-num-seqs=8: short stays hot, open idles.
- R_sat=8.0 dual criteria is partly RR throughput collapse (416→294 from 6.4→8.0) while experiment held ~416. TTFT still fails at 4.0 and 6.4, so collapse is not the only issue.

## Intervention chosen: B (earlier spill)
- `--prefill-spill-abs 2 --prefill-spill-rel 1.0` (topology 2+2 unchanged).
- Rejected A (more open) as primary: open was under-utilized, not capacity-starved; short overload is the system-p90 driver.
- Rejected C (decode remaining-output-load): no decode queueing signal.

## Remeasure @ seed 20251031 offsets 440/500/560 N=60
- passed_at_R_sat={d['passed_at_R_sat']}
- R_sat=8.0: RR tok/s={sat['control_output_tok_s']} ttft_p90={sat['control_ttft_p90']} | exp tok/s={sat['experiment_output_tok_s']} ttft_p90={sat['experiment_ttft_p90']} | ots_win={sat['output_tok_s_win']} ttft_ok={sat['ttft_p90_not_worse']} PASS={sat['pass']}
"""
for row in d['table']:
    text += (
        f"- rate {row['rate']}: ots {row['control_output_tok_s']:.1f}->{row['experiment_output_tok_s']:.1f} "
        f"ttft_p90 {row['control_ttft_p90']:.3f}->{row['experiment_ttft_p90']:.3f} PASS={row['pass']}\n"
    )
(root/'real_pd_trajectory_ab_diagnosis.md').write_text(text)
print(text)
PY

echo "=== REMEASURE DONE $(date -Is) ===" | tee -a "$SUMMARY"
restart_router bucket || true
exit 0
