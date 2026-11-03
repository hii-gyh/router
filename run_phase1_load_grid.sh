#!/bin/bash
# Phase 1 coarse load-scenario grid on live Ascend PD.
# Current design under test (A1): soft prefill_bucket + decode_kv_cost (full-pool min-cost)
# Control (A0): RR+RR
#
# Dual success criteria (consistent across cells):
#   A1.output_tok_s > A0.output_tok_s  AND  A1.ttft_p90 <= A0.ttft_p90
#
# Coarse grid (fixed prompt/out where possible):
#   prompts: 512 / 2048 / 8192
#   outs:    256 / 2048
#   rates:   short-biased higher; long-biased lower
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
AB_DIR=$LOGDIR/phase1_load_grid
RESULT=$LOGDIR/phase1_load_grid.json
SUMMARY=$LOGDIR/phase1_load_grid.txt
PACK=$LOGDIR/phase1_load_grid_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK"

FINAL_N="${FINAL_N:-40}"
WARM_N="${WARM_N:-8}"
SEED="${SEED:-20251103}"

echo "PHASE1 LOAD GRID $(date -Is)" | tee "$SUMMARY"
echo "criteria: A1.tok_s>A0.tok_s AND A1.ttft_p90<=A0.ttft_p90" | tee -a "$SUMMARY"
echo "A1=soft prefill_bucket + decode_kv_cost (full-pool); A0=RR+RR" | tee -a "$SUMMARY"

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
while time.time()-t0 < 600:
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
    args+=(
      --prefill-policy prefill_bucket --decode-policy decode_kv_cost
      --prefill-bucket-max-tokens 2048 --prefill-preference-weight 2.0
      --prefill-bucket-worker http://127.0.0.1:18001=0
      --prefill-bucket-worker http://127.0.0.1:18002=0
    )
  fi
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && { echo "router ready arm=$arm (${i}s)"; return 0; }
    sleep 1
  done
  nerdctl logs --tail 40 pd-real-router 2>&1 | tail -40; return 1
}

# Confirm binary has no hybrid RR floor CLI.
if "$ROOT/vllm-router" --help 2>&1 | grep -q 'decode-rr-below'; then
  echo "FATAL: hybrid RR CLI still present" | tee -a "$SUMMARY"; exit 5
fi
echo "binary_no_hybrid_cli=ok" | tee -a "$SUMMARY"

[[ -x "$ROOT/vllm-router" ]] || { echo missing binary; exit 2; }
workers_ok || exit 3

# cell list: prompt out rates...
CELLS=(
  "512:256:4.0,8.0,12.0"
  "2048:256:2.0,4.0,6.0"
  "2048:2048:1.0,2.0,3.0"
  "8192:256:1.0,2.0,3.0"
  "8192:2048:0.8,1.6,2.4"
)

ROWS_JSON='[]'
echo "[]" > "$AB_DIR/rows.json"

for cell in "${CELLS[@]}"; do
  IFS=':' read -r P O RATES_CSV <<<"$cell"
  TRAJ=$ROOT/data/trajectory_fixed_p${P}_o${O}.json
  echo "=== CELL p=$P o=$O rates=$RATES_CSV $(date -Is) ===" | tee -a "$SUMMARY"
  run_client --mode prepare-fixed --traj "$TRAJ" --seed "$SEED" --n "$FINAL_N" \
    --prompt-tokens "$P" --max-tokens "$O" | tee -a "$SUMMARY"
  IFS=',' read -r -a RATES <<<"$RATES_CSV"

  for arm in A0 A1; do
    restart_router "$arm"
    host_client_idle || true
    run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
    host_client_idle || true
    for r in "${RATES[@]}"; do
      workers_ok || exit 3
      label="${arm}_p${P}_o${O}_r${r}"
      out="$AB_DIR/${label}.json"
      echo "=== run $label $(date -Is) ===" | tee -a "$SUMMARY"
      ignore=()
      if [[ "$O" -ge 512 ]]; then ignore=(--ignore-eos); fi
      run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset 0 \
        --label "$label" --out "$out" "${ignore[@]}" \
        | tee "$AB_DIR/${label}.log" | tail -15
      host_client_idle || true
    done
  done

  python3 - <<PY
import json, pathlib
ab = pathlib.Path("$AB_DIR")
p, o = int("$P"), int("$O")
rates = [float(x) for x in "$RATES_CSV".split(",")]
rows = json.loads((ab/"rows.json").read_text())
for r in rates:
  a0 = json.loads((ab/f"A0_p{p}_o{o}_r{r}.json").read_text())
  a1 = json.loads((ab/f"A1_p{p}_o{o}_r{r}.json").read_text())
  k0 = a0.get("decode_kv") or {}
  k1 = a1.get("decode_kv") or {}
  ots_win = (a1.get("output_tok_s") or 0) > (a0.get("output_tok_s") or 0)
  ttft_ok = (a1.get("ttft_p90") or 1e9) <= (a0.get("ttft_p90") or 0)
  row = {
    "prompt_tokens": p, "max_tokens": o, "rate": r,
    "A0_tok_s": a0.get("output_tok_s"), "A1_tok_s": a1.get("output_tok_s"),
    "A0_ttft_p90": a0.get("ttft_p90"), "A1_ttft_p90": a1.get("ttft_p90"),
    "A0_error_rate": a0.get("error_rate"), "A1_error_rate": a1.get("error_rate"),
    "A0_kv_mean_of_max": k0.get("kv_mean_of_max"), "A1_kv_mean_of_max": k1.get("kv_mean_of_max"),
    "A0_kv_peak_max": k0.get("kv_peak_max"), "A1_kv_peak_max": k1.get("kv_peak_max"),
    "ots_win": ots_win, "ttft_ok": ttft_ok, "pass": bool(ots_win and ttft_ok),
  }
  rows.append(row)
  print(json.dumps(row))
(ab/"rows.json").write_text(json.dumps(rows, indent=2))
PY
done

python3 - <<'PY'
import json, pathlib, datetime, shutil, tarfile
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root/'phase1_load_grid'
rows = json.loads((ab/'rows.json').read_text())
wins = [r for r in rows if r.get('pass')]
out = {
  'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
  'scheme': 'A1 soft prefill_bucket + decode_kv_cost(full-pool) vs A0 RR+RR',
  'criteria': ['A1.output_tok_s > A0.output_tok_s', 'A1.ttft_p90 <= A0.ttft_p90'],
  'n_cells': len(rows),
  'n_pass': len(wins),
  'wins': wins,
  'rows': rows,
}
(root/'phase1_load_grid.json').write_text(json.dumps(out, indent=2))
lines = [
  f"n_cells={len(rows)} n_pass={len(wins)}",
  "WINS:" if wins else "WINS: none",
]
for w in wins:
  lines.append(
    f"  p={w['prompt_tokens']} o={w['max_tokens']} r={w['rate']} "
    f"tok {w['A0_tok_s']}->{w['A1_tok_s']} ttft {w['A0_ttft_p90']:.3f}->{w['A1_ttft_p90']:.3f}"
  )
sum_path = root/'phase1_load_grid.txt'
sum_path.write_text(sum_path.read_text() + "\n" + "\n".join(lines) + "\n", encoding='utf-8')
pack = root/'phase1_load_grid_pack'
pack.mkdir(exist_ok=True)
for p in ab.glob('*.json'):
  shutil.copy2(p, pack/p.name)
(pack/'phase1_load_grid.json').write_text(json.dumps(out, indent=2))
shutil.copy2(sum_path, pack/'phase1_load_grid.txt')
with tarfile.open(root/'phase1_load_grid_pack.tgz', 'w:gz') as tar:
  tar.add(pack, arcname='phase1_load_grid_pack')
print('PACKED', root/'phase1_load_grid_pack.tgz')
print('PASS_COUNT', len(wins))
PY

echo "PHASE1 DONE $(date -Is)" | tee -a "$SUMMARY"
