#!/bin/bash
# Ablation on live real-PD 2+2P/4D (same trajectory as soft remesaure):
#   A0 baseline:     prefill=round_robin, decode=round_robin
#   A1 decode-only:  prefill=round_robin, decode=decode_kv_cost
#   A2 soft-full:    prefill=prefill_bucket soft, decode=decode_kv_cost
# Success still judged at R_sat vs A0; also report incremental A1-A0 and A2-A1.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$ROOT/data/trajectory.json
AB_DIR=$LOGDIR/real_pd_ablation
RESULT=$LOGDIR/real_pd_ablation.json
SUMMARY=$LOGDIR/real_pd_ablation.txt
PACK=$LOGDIR/ablation_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK"

SUCCESS_JSON='{"at":"R_sat","R_sat":8.0,"rates":[4.0,6.4,8.0],"arms":["A0_rr_rr","A1_rr_decode_kv","A2_soft_prefill_decode_kv"],"require_vs_A0":["arm.output_tok_s > A0.output_tok_s","arm.ttft_p90 <= A0.ttft_p90"],"note":"Ablation: isolate decode_kv_cost then add soft prefill_bucket. Native prefill = round_robin."}'
echo "$SUCCESS_JSON" > "$LOGDIR/real_pd_ablation_success_criteria.json"
echo "ABLATION $(date -Is)" | tee "$SUMMARY"
echo "$SUCCESS_JSON" | tee -a "$SUMMARY"

FINAL_N="${FINAL_N:-60}"
WARM_N="${WARM_N:-20}"
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

ensure_binary_has_soft() {
  # Avoid `cmd | grep` under pipefail with a crashing --help; probe the binary bytes.
  local hit
  hit=$(strings "$ROOT/vllm-router" 2>/dev/null | grep -E 'prefill_preference_weight|prefill-preference-weight' | head -1 || true)
  if [[ -z "$hit" ]]; then
    echo "FATAL: missing soft preference in binary" | tee -a "$SUMMARY"
    exit 5
  fi
  echo "soft flag ok: $hit" | tee -a "$SUMMARY"
  md5sum "$ROOT/vllm-router" | tee -a "$SUMMARY"
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
  case "$arm" in
    A0)
      args+=(--prefill-policy round_robin --decode-policy round_robin)
      ;;
    A1)
      # Native prefill; decode intervention only.
      args+=(--prefill-policy round_robin --decode-policy decode_kv_cost)
      ;;
    A2)
      args+=(
        --prefill-policy prefill_bucket
        --decode-policy decode_kv_cost
        --prefill-bucket-max-tokens 2048
        --prefill-preference-weight 2.0
        --prefill-bucket-worker http://127.0.0.1:18001=0
        --prefill-bucket-worker http://127.0.0.1:18002=0
      )
      ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready arm=$arm (${i}s)" | tee -a "$SUMMARY"
      nerdctl logs pd-real-router 2>&1 | grep -E 'policy|PrefillBucket|decode_kv|round_robin|preference' | head -8 | tee -a "$SUMMARY" || true
      return 0
    fi
    sleep 1
  done
  echo "router failed arm=$arm" >&2
  nerdctl logs --tail 80 pd-real-router 2>&1 | tail -80 >&2
  return 1
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
    local out="$AB_DIR/${arm}_rate_${r}.json"
    run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset "$o" \
      --label "${name}_${r}" --out "$out" | tee "$AB_DIR/${arm}_rate_${r}.log" | tail -12
    host_client_idle || true
  done
}

[[ -f "$TRAJ" ]] || { echo "missing $TRAJ"; exit 2; }
[[ -x "$ROOT/vllm-router" ]] || { echo "missing binary"; exit 2; }
ensure_binary_has_soft
workers_ok || { echo "FATAL workers"; exit 3; }

echo "=== A0 baseline RR+RR $(date -Is) ===" | tee -a "$SUMMARY"
run_arm A0 A0_rr_rr

echo "=== A1 decode-only RR+decode_kv_cost $(date -Is) ===" | tee -a "$SUMMARY"
run_arm A1 A1_rr_decode_kv

echo "=== A2 soft-full soft_prefill+decode_kv_cost $(date -Is) ===" | tee -a "$SUMMARY"
run_arm A2 A2_soft_prefill_decode_kv

python3 - <<'PY'
import json, pathlib, datetime
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root / 'real_pd_ablation'
success = json.loads((root/'real_pd_ablation_success_criteria.json').read_text())
rates = [4.0, 6.4, 8.0]
arms = ['A0', 'A1', 'A2']
names = {
  'A0': 'RR+RR',
  'A1': 'RR+decode_kv_cost',
  'A2': 'soft_prefill+decode_kv_cost',
}

def load(arm, r):
  return json.loads((ab/f'{arm}_rate_{r}.json').read_text())

rows = []
for r in rates:
  by = {a: load(a, r) for a in arms}
  a0, a1, a2 = by['A0'], by['A1'], by['A2']
  def delta(ex, base, key):
    ev, bv = ex.get(key), base.get(key)
    if ev is None or bv is None: return None
    return ev - bv
  row = {
    'rate': r,
    'A0_tok_s': a0.get('output_tok_s'), 'A0_ttft_p90': a0.get('ttft_p90'),
    'A1_tok_s': a1.get('output_tok_s'), 'A1_ttft_p90': a1.get('ttft_p90'),
    'A2_tok_s': a2.get('output_tok_s'), 'A2_ttft_p90': a2.get('ttft_p90'),
    'A1_vs_A0_tok': delta(a1,a0,'output_tok_s'),
    'A1_vs_A0_ttft': delta(a1,a0,'ttft_p90'),
    'A2_vs_A0_tok': delta(a2,a0,'output_tok_s'),
    'A2_vs_A0_ttft': delta(a2,a0,'ttft_p90'),
    'A2_vs_A1_tok': delta(a2,a1,'output_tok_s'),
    'A2_vs_A1_ttft': delta(a2,a1,'ttft_p90'),
    'A1_pass_vs_A0': (a1.get('output_tok_s') or 0) > (a0.get('output_tok_s') or 0) and (a1.get('ttft_p90') or 9e9) <= (a0.get('ttft_p90') or 0),
    'A2_pass_vs_A0': (a2.get('output_tok_s') or 0) > (a0.get('output_tok_s') or 0) and (a2.get('ttft_p90') or 9e9) <= (a0.get('ttft_p90') or 0),
    'A0_err': a0.get('error_rate'), 'A1_err': a1.get('error_rate'), 'A2_err': a2.get('error_rate'),
  }
  rows.append(row)

rsat = next(x for x in rows if x['rate'] == 8.0)
out = {
  'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
  'success_criteria': success,
  'intervention': 'ablation A0 RR+RR / A1 RR+decode_kv / A2 soft_prefill+decode_kv',
  'A1_passed_at_R_sat': bool(rsat['A1_pass_vs_A0']),
  'A2_passed_at_R_sat': bool(rsat['A2_pass_vs_A0']),
  'rows': rows,
  'arm_names': names,
}
(root/'real_pd_ablation.json').write_text(json.dumps(out, indent=2))
lines = [
  f"A1_passed_at_R_sat={out['A1_passed_at_R_sat']}  (decode-only vs RR)",
  f"A2_passed_at_R_sat={out['A2_passed_at_R_sat']}  (soft+decode vs RR)",
  'rate | A0 tok | A1 tok | A2 tok | A0 p90 | A1 p90 | A2 p90 | A1pass | A2pass | dTok(A1-A0) | dTok(A2-A1)',
]
for x in rows:
  lines.append(
    f"{x['rate']} | {x['A0_tok_s']:.1f} | {x['A1_tok_s']:.1f} | {x['A2_tok_s']:.1f} | "
    f"{x['A0_ttft_p90']:.3f} | {x['A1_ttft_p90']:.3f} | {x['A2_ttft_p90']:.3f} | "
    f"{x['A1_pass_vs_A0']} | {x['A2_pass_vs_A0']} | "
    f"{x['A1_vs_A0_tok']:.1f} | {x['A2_vs_A1_tok']:.1f}"
  )
text = '\n'.join(lines) + '\n'
print(text)
sum_path = root/'real_pd_ablation.txt'
sum_path.write_text(sum_path.read_text() + '\n' + text)

pack = root/'ablation_pack'
pack.mkdir(exist_ok=True)
(pack/'SUMMARY.md').write_text(
  '# Real-PD ablation\n\n'
  f"- finished_at: {out['finished_at']}\n"
  f"- A1_passed_at_R_sat (decode-only): **{out['A1_passed_at_R_sat']}**\n"
  f"- A2_passed_at_R_sat (soft+decode): **{out['A2_passed_at_R_sat']}**\n\n"
  + text
)
(pack/'real_pd_ablation.json').write_text(json.dumps(out, indent=2))
(pack/'DONE').write_text(
  f"status=OK\nfinished_at={out['finished_at']}\n"
  f"A1_passed_at_R_sat={out['A1_passed_at_R_sat']}\n"
  f"A2_passed_at_R_sat={out['A2_passed_at_R_sat']}\n"
)
import shutil
ab_dst = pack/'real_pd_ablation'
ab_dst.mkdir(exist_ok=True)
for p in ab.glob('*'):
  shutil.copy2(p, ab_dst/p.name)
shutil.copy2(root/'real_pd_ablation.txt', pack/'real_pd_ablation.txt')
import tarfile
with tarfile.open(root/'ablation_pack.tgz', 'w:gz') as tar:
  tar.add(pack, arcname='ablation_pack')
print(f"PACKED {root/'ablation_pack.tgz'}")
PY

echo "=== ablation done $(date -Is) ===" | tee -a "$SUMMARY"
cat "$RESULT" | tee -a "$SUMMARY"
exit 0
