#!/bin/bash
# High-KV pressure remesaure for full-pool min-cost decode scheduling.
#   A0: RR+RR
#   A1: RR prefill + decode_kv_cost (full-pool lowest_cost)
#
# Prior ShareGPT-ish traj only hit ~10–12% KV; this arm uses a long-context
# resident + probe mixture with --ignore-eos.
#
# Env overrides:
#   FINAL_N WARM_N RATES OFFSETS
#   TRAJ_SEED (default 20251102)
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$ROOT/data/trajectory_high_kv.json
AB_DIR=$LOGDIR/real_pd_decode_high_kv
RESULT=$LOGDIR/real_pd_decode_high_kv.json
SUMMARY=$LOGDIR/real_pd_decode_high_kv.txt
PACK=$LOGDIR/decode_high_kv_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK" "$(dirname "$TRAJ")"

echo "DECODE HIGH-KV PRESSURE REMEASURE $(date -Is)" | tee "$SUMMARY"
FINAL_N="${FINAL_N:-60}"
WARM_N="${WARM_N:-12}"
TRAJ_SEED="${TRAJ_SEED:-20251102}"
# Lower rates than ShareGPT remesaure: each request holds KV much longer.
# Default window includes the preload resident burst at offset 0.
if [[ -z "${RATES:-}" ]]; then RATES=(1.0 1.6 2.4); else read -r -a RATES <<< "$RATES"; fi
if [[ -z "${OFFSETS:-}" ]]; then OFFSETS=(0 60 120); else read -r -a OFFSETS <<< "$OFFSETS"; fi
# R_sat for this pressure pack (third rate by default).
R_SAT="${R_SAT:-2.4}"

workers_ok() {
  local p ok=0
  for p in 18001 18002 18003 18004 18011 18012 18013 18014; do
    curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && ok=$((ok+1)) || true
  done
  echo "workers_healthy=${ok}/8"; [[ "$ok" -eq 8 ]]
}

host_client_idle() {
  python3 - <<'PY'
import json, urllib.request, time
DECODE=[18011,18012,18013,18014]
def snap():
  out={}
  for p in DECODE:
    try: t=urllib.request.urlopen(f'http://127.0.0.1:{p}/metrics',timeout=3).read().decode()
    except Exception: out[p]={'running':99,'waiting':99}; continue
    run=wait=0.0
    for line in t.splitlines():
      if line.startswith('vllm:num_requests_running{'): run=float(line.split()[-1])
      elif line.startswith('vllm:num_requests_waiting{') and 'reason' not in line: wait=float(line.split()[-1])
    out[p]={'running':run,'waiting':wait}
  return out
t0=time.time()
# Long residents can take minutes to drain.
while time.time()-t0 < 900:
  m=snap(); r=sum(v['running'] for v in m.values()); w=sum(v['waiting'] for v in m.values())
  if r<=0.01 and w<=0.01: print(json.dumps({'idle':True})); raise SystemExit(0)
  time.sleep(3)
print(json.dumps({'idle':False})); raise SystemExit(1)
PY
}

run_client() {
  nerdctl run --rm --net=host -v "$ROOT:$ROOT" -v /mnt/weights:/home/weights \
    -e TOKENIZER_PATH=/home/weights/Qwen3.8-27B-w8a8/ "$IMG" python3 "$CLIENT" "$@"
}

restart_router() {
  local arm="$1"
  nerdctl rm -f pd-real-router >/dev/null 2>&1 || true
  local -a args=(
    "$ROOT/vllm-router" --host 0.0.0.0 --port 8090
    --vllm-pd-disaggregation --kv-connector nixl
    --prefill http://127.0.0.1:18001 30000 --prefill http://127.0.0.1:18002 30001
    --prefill http://127.0.0.1:18003 30002 --prefill http://127.0.0.1:18004 30003
    --decode http://127.0.0.1:18011 --decode http://127.0.0.1:18012
    --decode http://127.0.0.1:18013 --decode http://127.0.0.1:18014
    --disable-retries --disable-circuit-breaker
  )
  if [[ "$arm" == "A0" ]]; then
    args+=(--prefill-policy round_robin --decode-policy round_robin)
  else
    args+=(--prefill-policy round_robin --decode-policy decode_kv_cost)
  fi
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && { echo "router ready arm=$arm (${i}s)"; return 0; }
    sleep 1
  done
  nerdctl logs --tail 60 pd-real-router 2>&1 | tail -60; return 1
}

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
    run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset "$o" \
      --ignore-eos \
      --label "${name}_${r}" --out "$AB_DIR/${arm}_rate_${r}.json" \
      | tee "$AB_DIR/${arm}_rate_${r}.log" | tail -20
    host_client_idle || true
  done
}

[[ -x "$ROOT/vllm-router" ]] || { echo "missing router binary"; exit 2; }
workers_ok || exit 3

echo "=== prepare high-kv traj seed=$TRAJ_SEED $(date -Is) ===" | tee -a "$SUMMARY"
run_client --mode prepare-high-kv --traj "$TRAJ" --seed "$TRAJ_SEED" | tee -a "$SUMMARY"
[[ -f "$TRAJ" ]] || exit 2

{
  echo "config FINAL_N=$FINAL_N WARM_N=$WARM_N RATES=${RATES[*]} OFFSETS=${OFFSETS[*]}"
  echo "config R_SAT=$R_SAT decode=full-pool min-cost"
  echo "traj=$TRAJ"
} | tee -a "$SUMMARY"

echo "=== A0 RR+RR $(date -Is) ===" | tee -a "$SUMMARY"; run_arm A0 A0_rr_rr
echo "=== A1 decode_kv_cost $(date -Is) ===" | tee -a "$SUMMARY"; run_arm A1 A1_rr_decode_kv_high_kv

python3 - <<PY
import json, pathlib, datetime, shutil, tarfile
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root/'real_pd_decode_high_kv'
rates = [float(x) for x in "${RATES[*]}".split()]
rsat = float("$R_SAT")
rows=[]
for r in rates:
  a0=json.loads((ab/f'A0_rate_{r}.json').read_text())
  a1=json.loads((ab/f'A1_rate_{r}.json').read_text())
  k0=a0.get('decode_kv') or {}
  k1=a1.get('decode_kv') or {}
  rows.append({
    'rate': r,
    'A0_tok_s': a0.get('output_tok_s'), 'A1_tok_s': a1.get('output_tok_s'),
    'A0_ttft_p90': a0.get('ttft_p90'), 'A1_ttft_p90': a1.get('ttft_p90'),
    'A0_kv_peak_max': k0.get('kv_peak_max'), 'A1_kv_peak_max': k1.get('kv_peak_max'),
    'A0_kv_mean_of_max': k0.get('kv_mean_of_max'), 'A1_kv_mean_of_max': k1.get('kv_mean_of_max'),
    'A0_kv_frac_ge_0_25': k0.get('kv_frac_max_ge_0_25'), 'A1_kv_frac_ge_0_25': k1.get('kv_frac_max_ge_0_25'),
    'A0_kv_frac_ge_0_40': k0.get('kv_frac_max_ge_0_40'), 'A1_kv_frac_ge_0_40': k1.get('kv_frac_max_ge_0_40'),
    'A0_inflight_frac_ge_4': k0.get('inflight_frac_max_ge_4'),
    'A1_inflight_frac_ge_4': k1.get('inflight_frac_max_ge_4'),
    'A0_kv_std': k0.get('kv_stddev_mean'), 'A1_kv_std': k1.get('kv_stddev_mean'),
    'pass': (a1.get('output_tok_s') or 0)>(a0.get('output_tok_s') or 0) and (a1.get('ttft_p90') or 9e9)<=(a0.get('ttft_p90') or 0),
  })
rsat_row=next((x for x in rows if abs(x['rate']-rsat)<1e-9), rows[-1])
# Pressure sanity: must leave the light KV band (inflight-only is NOT enough —
# prior light traj already engaged via inflight>=4 and still lost at R_sat).
pressure_ok = any(
  (x.get('A0_kv_mean_of_max') or 0) >= 0.35
  or (x.get('A0_kv_frac_ge_0_25') or 0) >= 0.40
  or (x.get('A0_kv_peak_max') or 0) >= 0.45
  for x in rows
)
out={
  'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
  'scheme': 'full-pool min-cost decode_kv_cost',
  'traj': 'synthetic_high_kv_decode_pressure',
  'R_sat': rsat,
  'ignore_eos': True,
  'pressure_target_hit': bool(pressure_ok),
  'A1_passed_at_R_sat': bool(rsat_row['pass']),
  'rows': rows,
}
(root/'real_pd_decode_high_kv.json').write_text(json.dumps(out, indent=2))
lines=[
  f"A1_passed_at_R_sat={out['A1_passed_at_R_sat']}",
  f"pressure_target_hit={out['pressure_target_hit']}",
  'rate | A0 tok | A1 tok | A0 p90 | A1 p90 | A0 kv_mean_max | A1 kv_mean_max | A0 frac>=0.25 | A0 kv_std | A1 kv_std | PASS',
]
for x in rows:
  lines.append(
    f"{x['rate']} | {x['A0_tok_s']:.1f} | {x['A1_tok_s']:.1f} | "
    f"{(x['A0_ttft_p90'] or 0):.3f} | {(x['A1_ttft_p90'] or 0):.3f} | "
    f"{x['A0_kv_mean_of_max']} | {x['A1_kv_mean_of_max']} | "
    f"{x['A0_kv_frac_ge_0_25']} | {x['A0_kv_std']} | {x['A1_kv_std']} | {x['pass']}"
  )
text='\n'.join(lines)+'\n'
print(text)
sum_path = root/'real_pd_decode_high_kv.txt'
prev = sum_path.read_text() if sum_path.exists() else ''
sum_path.write_text(prev + '\n' + text)
pack=root/'decode_high_kv_pack'; pack.mkdir(exist_ok=True)
(pack/'SUMMARY.md').write_text(
  '# Decode high-KV pressure remesaure\n\n'
  f"- scheme: `{out['scheme']}`\n"
  f"- traj: `{out['traj']}` ignore_eos=True\n"
  f"- R_sat={rsat}\n"
  f"- pressure_target_hit={out['pressure_target_hit']}\n"
  f"- A1_passed_at_R_sat={out['A1_passed_at_R_sat']}\n\n"
  + text
)
(pack/'real_pd_decode_high_kv.json').write_text(json.dumps(out, indent=2))
(pack/'DONE').write_text(
  f"status=OK\nA1_passed_at_R_sat={out['A1_passed_at_R_sat']}\n"
  f"pressure_target_hit={out['pressure_target_hit']}\nfinished_at={out['finished_at']}\n"
)
dst=pack/'runs'; dst.mkdir(exist_ok=True)
for p in ab.glob('*'): shutil.copy2(p, dst/p.name)
shutil.copy2(pathlib.Path('/root/gyh_dev/vllm_router/data/trajectory_high_kv.json'), pack/'trajectory_high_kv.json')
with tarfile.open(root/'decode_high_kv_pack.tgz','w:gz') as tar:
  tar.add(pack, arcname='decode_high_kv_pack')
print('PACKED', root/'decode_high_kv_pack.tgz')
PY
echo "=== decode high-kv remesaure done $(date -Is) ===" | tee -a "$SUMMARY"
exit 0
