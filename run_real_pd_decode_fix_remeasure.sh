#!/bin/bash
# Remeasure decode-only after full-pool min-cost decode_kv_cost:
#   A0: RR+RR
#   A1: RR prefill + decode_kv_cost (full-pool lowest_cost)
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$ROOT/data/trajectory.json
AB_DIR=$LOGDIR/real_pd_decode_fix
RESULT=$LOGDIR/real_pd_decode_fix.json
SUMMARY=$LOGDIR/real_pd_decode_fix.txt
PACK=$LOGDIR/decode_fix_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK"

echo "DECODE FIX REMEASURE $(date -Is)" | tee "$SUMMARY"
FINAL_N="${FINAL_N:-60}"; WARM_N="${WARM_N:-20}"
OFFSETS=(440 500 560); RATES=(4.0 6.4 8.0)

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
while time.time()-t0 < 240:
  m=snap(); r=sum(v['running'] for v in m.values()); w=sum(v['waiting'] for v in m.values())
  if r<=0.01 and w<=0.01: print(json.dumps({'idle':True})); raise SystemExit(0)
  time.sleep(2)
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
      --label "${name}_${r}" --out "$AB_DIR/${arm}_rate_${r}.json" \
      | tee "$AB_DIR/${arm}_rate_${r}.log" | tail -15
    host_client_idle || true
  done
}

[[ -f "$TRAJ" && -x "$ROOT/vllm-router" ]] || exit 2
workers_ok || exit 3
echo "=== A0 $(date -Is) ===" | tee -a "$SUMMARY"; run_arm A0 A0_rr_rr
echo "=== A1 $(date -Is) ===" | tee -a "$SUMMARY"; run_arm A1 A1_rr_decode_kv_cost

python3 - <<'PY'
import json, pathlib, datetime, shutil, tarfile
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root/'real_pd_decode_fix'
rows=[]
for r in [4.0,6.4,8.0]:
  a0=json.loads((ab/f'A0_rate_{r}.json').read_text())
  a1=json.loads((ab/f'A1_rate_{r}.json').read_text())
  rows.append({
    'rate': r,
    'A0_tok_s': a0.get('output_tok_s'), 'A1_tok_s': a1.get('output_tok_s'),
    'A0_ttft_p90': a0.get('ttft_p90'), 'A1_ttft_p90': a1.get('ttft_p90'),
    'A0_kv_std': (a0.get('decode_kv') or {}).get('kv_stddev_mean'),
    'A1_kv_std': (a1.get('decode_kv') or {}).get('kv_stddev_mean'),
    'pass': (a1.get('output_tok_s') or 0)>(a0.get('output_tok_s') or 0) and (a1.get('ttft_p90') or 9e9)<=(a0.get('ttft_p90') or 0),
  })
rsat=next(x for x in rows if x['rate']==8.0)
out={'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
     'scheme':'full-pool min-cost decode_kv_cost',
     'A1_passed_at_R_sat': bool(rsat['pass']), 'rows': rows}
(root/'real_pd_decode_fix.json').write_text(json.dumps(out, indent=2))
lines=[f"A1_passed_at_R_sat={out['A1_passed_at_R_sat']}",
       'rate | A0 tok | A1 tok | A0 p90 | A1 p90 | A0 kv_std | A1 kv_std | PASS']
for x in rows:
  lines.append(f"{x['rate']} | {x['A0_tok_s']:.1f} | {x['A1_tok_s']:.1f} | {x['A0_ttft_p90']:.3f} | {x['A1_ttft_p90']:.3f} | {x['A0_kv_std']} | {x['A1_kv_std']} | {x['pass']}")
text='\n'.join(lines)+'\n'
print(text)
sum_path = root/'real_pd_decode_fix.txt'
prev = sum_path.read_text() if sum_path.exists() else ''
sum_path.write_text(prev + '\n' + text)
pack=root/'decode_fix_pack'; pack.mkdir(exist_ok=True)
(pack/'SUMMARY.md').write_text('# Decode full-pool min-cost remesaure\n\n'+text)
(pack/'real_pd_decode_fix.json').write_text(json.dumps(out, indent=2))
(pack/'DONE').write_text(f"status=OK\nA1_passed_at_R_sat={out['A1_passed_at_R_sat']}\nfinished_at={out['finished_at']}\n")
dst=pack/'runs'; dst.mkdir(exist_ok=True)
for p in ab.glob('*'): shutil.copy2(p, dst/p.name)
with tarfile.open(root/'decode_fix_pack.tgz','w:gz') as tar: tar.add(pack, arcname='decode_fix_pack')
print('PACKED', root/'decode_fix_pack.tgz')
PY
echo "=== decode fix remesaure done $(date -Is) ===" | tee -a "$SUMMARY"
exit 0
