#!/bin/bash
# Phase1 HIGH-KV single-factor ablation (small grid).
# Baseline A0: RR+RR
# Arms:
#   A_prefill: soft prefill_bucket + RR decode
#   A_decode:  RR prefill + decode_kv_cost (full-pool min-cost)
#   A_both:    soft prefill_bucket + decode_kv_cost (current design)
#
# Traj: prepare-high-kv (long residents + probes). Fair open-only 2P@16384
# for ALL arms (prompts overflow short@4096). No hybrid RR. p32k N/A.
#
# Dual criteria vs A0: tok/s↑ AND TTFT p90≤A0; also report tok_only.
# Discard/unfair if error_rate > ERR_MAX. Remeasure any dual-PASS 1–2 reps.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
TRAJ=$ROOT/data/trajectory_high_kv_ablation.json
AB_DIR=$LOGDIR/phase1_high_kv_ablation
RESULT=$LOGDIR/phase1_high_kv_ablation.json
SUMMARY=$LOGDIR/phase1_high_kv_ablation.txt
PACK=$LOGDIR/phase1_high_kv_ablation_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK" "$(dirname "$TRAJ")"

FINAL_N="${FINAL_N:-40}"
WARM_N="${WARM_N:-8}"
SEED="${SEED:-20251103}"
ERR_MAX="${ERR_MAX:-0.05}"
REMEAS_REPS="${REMEAS_REPS:-2}"
# Small grid: 1–2 rates near saturation for high-KV open-2P.
if [[ -z "${RATES:-}" ]]; then RATES=(1.2 1.8); else read -r -a RATES <<< "$RATES"; fi

echo "PHASE1 HIGH-KV ABLATION $(date -Is)" | tee "$SUMMARY"
echo "criteria: arm.tok_s>A0.tok_s AND arm.ttft_p90<=A0.ttft_p90; also tok_only" | tee -a "$SUMMARY"
echo "A0=RR+RR; A_prefill=soft_bucket+RR; A_decode=RR+decode_kv_cost; A_both=soft_bucket+decode_kv_cost" | tee -a "$SUMMARY"
echo "fair: open-only 2P@16384 all arms; discard err>$ERR_MAX; ignore_eos; seed=$SEED" | tee -a "$SUMMARY"
echo "rates=${RATES[*]} FINAL_N=$FINAL_N WARM_N=$WARM_N REMEAS_REPS=$REMEAS_REPS" | tee -a "$SUMMARY"

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

# Always open-only 2P (fair for high-KV long prompts).
restart_router() {
  local arm="$1"
  nerdctl rm -f pd-real-router >/dev/null 2>&1 || true
  local -a args=(
    "$ROOT/vllm-router" --host 0.0.0.0 --port 8090
    --vllm-pd-disaggregation --kv-connector nixl
    --prefill http://127.0.0.1:18003 30002 --prefill http://127.0.0.1:18004 30003
    --decode http://127.0.0.1:18011 --decode http://127.0.0.1:18012
    --decode http://127.0.0.1:18013 --decode http://127.0.0.1:18014
    --disable-retries --disable-circuit-breaker
  )
  case "$arm" in
    A0)
      args+=(--prefill-policy round_robin --decode-policy round_robin)
      ;;
    A_prefill)
      args+=(
        --prefill-policy prefill_bucket --decode-policy round_robin
        --prefill-bucket-max-tokens 2048 --prefill-preference-weight 2.0
      )
      ;;
    A_decode)
      args+=(--prefill-policy round_robin --decode-policy decode_kv_cost)
      ;;
    A_both)
      args+=(
        --prefill-policy prefill_bucket --decode-policy decode_kv_cost
        --prefill-bucket-max-tokens 2048 --prefill-preference-weight 2.0
      )
      ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && {
      echo "router ready arm=$arm (${i}s)"; return 0
    }
    sleep 1
  done
  nerdctl logs --tail 40 pd-real-router 2>&1 | tail -40; return 1
}

run_one() {
  local arm="$1" rate="$2" label="$3" out="$4"
  workers_ok || exit 3
  echo "=== run $label $(date -Is) ===" | tee -a "$SUMMARY"
  run_client --mode run --traj "$TRAJ" --rate "$rate" --n "$FINAL_N" --offset 0 \
    --label "$label" --out "$out" --ignore-eos \
    | tee "${out%.json}.log" | tail -20
  host_client_idle || true
}

if "$ROOT/vllm-router" --help 2>&1 | grep -q 'decode-rr-below'; then
  echo "FATAL: hybrid RR CLI still present" | tee -a "$SUMMARY"; exit 5
fi
echo "binary_no_hybrid_cli=ok" | tee -a "$SUMMARY"
[[ -x "$ROOT/vllm-router" ]] || { echo missing binary; exit 2; }
workers_ok || exit 3

echo "=== prepare high-kv traj seed=$SEED $(date -Is) ===" | tee -a "$SUMMARY"
run_client --mode prepare-high-kv --traj "$TRAJ" --seed "$SEED" | tee -a "$SUMMARY"
[[ -f "$TRAJ" ]] || exit 2

echo "[]" > "$AB_DIR/rows.json"
ARMS=(A0 A_prefill A_decode A_both)

for arm in "${ARMS[@]}"; do
  echo "=== ARM $arm $(date -Is) ===" | tee -a "$SUMMARY"
  restart_router "$arm"
  host_client_idle || true
  run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
  host_client_idle || true
  for r in "${RATES[@]}"; do
    label="${arm}_r${r}"
    run_one "$arm" "$r" "$label" "$AB_DIR/${label}.json"
  done
done

# Score vs A0; collect dual-PASS cells for remesaure.
python3 - <<PY
import json, pathlib
ab = pathlib.Path("$AB_DIR")
err_max = float("$ERR_MAX")
rates = [float(x) for x in "${RATES[*]}".split()]
arms = ["A0", "A_prefill", "A_decode", "A_both"]

def load(arm, r):
  return json.loads((ab / f"{arm}_r{r}.json").read_text())

rows = []
pass_cells = []
for r in rates:
  by = {a: load(a, r) for a in arms}
  a0 = by["A0"]
  e0 = float(a0.get("error_rate") or 0.0)
  k0 = a0.get("decode_kv") or {}
  row = {
    "rate": r, "phase": "primary",
    "A0_tok_s": a0.get("output_tok_s"), "A0_ttft_p90": a0.get("ttft_p90"),
    "A0_error_rate": e0,
    "A0_kv_mean_of_max": k0.get("kv_mean_of_max"),
    "A0_kv_peak_max": k0.get("kv_peak_max"),
  }
  for arm in ("A_prefill", "A_decode", "A_both"):
    d = by[arm]
    k = d.get("decode_kv") or {}
    e = float(d.get("error_rate") or 0.0)
    fair = (e0 <= err_max) and (e <= err_max)
    ots_win = (d.get("output_tok_s") or 0) > (a0.get("output_tok_s") or 0)
    ttft_ok = (d.get("ttft_p90") or 1e9) <= (a0.get("ttft_p90") or 0)
    dual = bool(fair and ots_win and ttft_ok)
    row[f"{arm}_tok_s"] = d.get("output_tok_s")
    row[f"{arm}_ttft_p90"] = d.get("ttft_p90")
    row[f"{arm}_error_rate"] = e
    row[f"{arm}_kv_mean_of_max"] = k.get("kv_mean_of_max")
    row[f"{arm}_kv_peak_max"] = k.get("kv_peak_max")
    row[f"{arm}_fair"] = fair
    row[f"{arm}_ots_win"] = ots_win
    row[f"{arm}_ttft_ok"] = ttft_ok
    row[f"{arm}_pass"] = dual
    row[f"{arm}_tok_only"] = bool(fair and ots_win and not ttft_ok)
    if dual:
      pass_cells.append({"arm": arm, "rate": r})
  rows.append(row)
  print(json.dumps(row))
(ab / "rows.json").write_text(json.dumps(rows, indent=2))
(ab / "pass_cells.json").write_text(json.dumps(pass_cells, indent=2))
print("PASS_CELLS", len(pass_cells))
PY

# Remeasure dual-PASS cells (arm+rate) for 1–2 reps vs fresh A0.
PASS_CELLS_JSON="$AB_DIR/pass_cells.json"
if [[ -f "$PASS_CELLS_JSON" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$PASS_CELLS_JSON')) else 1)"; then
  echo "=== REMEASURE dual-PASS cells reps=$REMEAS_REPS $(date -Is) ===" | tee -a "$SUMMARY"
  python3 - <<'PY' > "$AB_DIR/remeasure_plan.txt"
import json, pathlib
cells = json.loads(pathlib.Path("/root/gyh_dev/vllm_router/logs/phase1_high_kv_ablation/pass_cells.json").read_text())
# For each rate with a PASS: remesaure A0 then each winning arm, per rep.
from collections import defaultdict
by_rate = defaultdict(list)
for c in cells:
  by_rate[c["rate"]].append(c["arm"])
for r in sorted(by_rate):
  arms = ["A0"] + sorted(set(by_rate[r]))
  for arm in arms:
    print(f"{arm} {r}")
PY
  # Group plan by rate so each rep has matched A0 + arm runs.
  rates_todo=$(python3 -c "import json; print(' '.join(str(c['rate']) for c in json.load(open('$PASS_CELLS_JSON'))))")
  # Unique rates
  for rate in $(echo "$rates_todo" | tr ' ' '\n' | awk '!a[$0]++'); do
    arms_for_rate=$(awk -v r="$rate" '$2==r {print $1}' "$AB_DIR/remeasure_plan.txt")
    for rep in $(seq 1 "$REMEAS_REPS"); do
      for arm in $arms_for_rate; do
        restart_router "$arm"
        host_client_idle || true
        run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
        host_client_idle || true
        label="${arm}_r${rate}_remeas${rep}"
        run_one "$arm" "$rate" "$label" "$AB_DIR/${label}.json"
      done
    done
  done
else
  echo "=== no dual-PASS cells; skip remesaure $(date -Is) ===" | tee -a "$SUMMARY"
fi

python3 - <<'PY'
import json, pathlib, datetime, shutil, tarfile
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root / 'phase1_high_kv_ablation'
err_max = 0.05
rows = json.loads((ab / 'rows.json').read_text())
pass_cells = json.loads((ab / 'pass_cells.json').read_text()) if (ab / 'pass_cells.json').exists() else []

# Attach remesaure summaries if present.
remeas = []
for cell in pass_cells:
  arm, r = cell['arm'], cell['rate']
  reps = []
  for rep in (1, 2, 3, 4):
    p = ab / f'{arm}_r{r}_remeas{rep}.json'
    a0p = ab / f'A0_r{r}_remeas{rep}.json'
    if not p.exists() or not a0p.exists():
      continue
    d = json.loads(p.read_text()); a0 = json.loads(a0p.read_text())
    e0 = float(a0.get('error_rate') or 0); e = float(d.get('error_rate') or 0)
    fair = e0 <= err_max and e <= err_max
    ots = (d.get('output_tok_s') or 0) > (a0.get('output_tok_s') or 0)
    ttft = (d.get('ttft_p90') or 1e9) <= (a0.get('ttft_p90') or 0)
    reps.append({
      'rep': rep, 'fair': fair, 'pass': bool(fair and ots and ttft),
      'tok_only': bool(fair and ots and not ttft),
      'A0_tok_s': a0.get('output_tok_s'), 'arm_tok_s': d.get('output_tok_s'),
      'A0_ttft_p90': a0.get('ttft_p90'), 'arm_ttft_p90': d.get('ttft_p90'),
      'A0_error_rate': e0, 'arm_error_rate': e,
      'A0_kv_mean_of_max': (a0.get('decode_kv') or {}).get('kv_mean_of_max'),
      'arm_kv_mean_of_max': (d.get('decode_kv') or {}).get('kv_mean_of_max'),
    })
  if reps:
    remeas.append({'arm': arm, 'rate': r, 'reps': reps,
                   'n_pass': sum(1 for x in reps if x['pass']),
                   'n_reps': len(reps)})

def arm_summary(arm):
  out = []
  for row in rows:
    fair = row.get(f'{arm}_fair')
    dual = row.get(f'{arm}_pass')
    tok_only = row.get(f'{arm}_tok_only')
    out.append({
      'rate': row['rate'],
      'fair': fair,
      'pass': dual,
      'tok_only': tok_only,
      'A0_tok_s': row.get('A0_tok_s'),
      f'{arm}_tok_s': row.get(f'{arm}_tok_s'),
      'A0_ttft_p90': row.get('A0_ttft_p90'),
      f'{arm}_ttft_p90': row.get(f'{arm}_ttft_p90'),
      'A0_error_rate': row.get('A0_error_rate'),
      f'{arm}_error_rate': row.get(f'{arm}_error_rate'),
      'A0_kv_mean_of_max': row.get('A0_kv_mean_of_max'),
      f'{arm}_kv_mean_of_max': row.get(f'{arm}_kv_mean_of_max'),
      'A0_kv_peak_max': row.get('A0_kv_peak_max'),
      f'{arm}_kv_peak_max': row.get(f'{arm}_kv_peak_max'),
    })
  return out

kv_peaks = [row.get('A0_kv_peak_max') or 0 for row in rows]
kv_means = [row.get('A0_kv_mean_of_max') or 0 for row in rows]
pressure_ok = any(m >= 0.25 or p >= 0.35 for m, p in zip(kv_means, kv_peaks))

out = {
  'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
  'scheme': 'Phase1 HIGH-KV ablation: A_prefill / A_decode / A_both vs A0 RR+RR',
  'fairness': 'open-only 2P@16384 all arms; discard err>0.05; ignore_eos; prepare-high-kv',
  'criteria': ['arm.output_tok_s > A0.output_tok_s', 'arm.ttft_p90 <= A0.ttft_p90', 'fair error rates'],
  'arms': {
    'A0': 'RR+RR',
    'A_prefill': 'soft prefill_bucket + RR decode',
    'A_decode': 'RR prefill + decode_kv_cost',
    'A_both': 'soft prefill_bucket + decode_kv_cost',
  },
  'n_rates': len(rows),
  'pressure_target_hit': bool(pressure_ok),
  'A0_kv_mean_of_max_max': max(kv_means) if kv_means else None,
  'A0_kv_peak_max_max': max(kv_peaks) if kv_peaks else None,
  'primary_pass_by_arm': {
    arm: sum(1 for r in rows if r.get(f'{arm}_pass'))
    for arm in ('A_prefill', 'A_decode', 'A_both')
  },
  'primary_tok_only_by_arm': {
    arm: sum(1 for r in rows if r.get(f'{arm}_tok_only'))
    for arm in ('A_prefill', 'A_decode', 'A_both')
  },
  'pass_cells': pass_cells,
  'remeasure': remeas,
  'per_arm': {arm: arm_summary(arm) for arm in ('A_prefill', 'A_decode', 'A_both')},
  'rows': rows,
}
(root / 'phase1_high_kv_ablation.json').write_text(json.dumps(out, indent=2))

lines = [
  f"pressure_target_hit={out['pressure_target_hit']} A0_kv_mean_max={out['A0_kv_mean_of_max_max']} A0_kv_peak={out['A0_kv_peak_max_max']}",
  f"primary_pass={out['primary_pass_by_arm']} tok_only={out['primary_tok_only_by_arm']}",
  "rate | arm | result | A0 tok | arm tok | A0 p90 | arm p90 | err A0/arm | kv A0/arm",
]
for arm in ('A_prefill', 'A_decode', 'A_both'):
  for row in rows:
    if not row.get(f'{arm}_fair'):
      mark = 'UNFAIR'
    elif row.get(f'{arm}_pass'):
      mark = 'PASS'
    elif row.get(f'{arm}_tok_only'):
      mark = 'tok_only'
    elif row.get(f'{arm}_ttft_ok') and not row.get(f'{arm}_ots_win'):
      mark = 'ttft_only'
    else:
      mark = 'lose'
    lines.append(
      f"{row['rate']} | {arm} | {mark} | {row.get('A0_tok_s')} | {row.get(f'{arm}_tok_s')} | "
      f"{row.get('A0_ttft_p90')} | {row.get(f'{arm}_ttft_p90')} | "
      f"{row.get('A0_error_rate')}/{row.get(f'{arm}_error_rate')} | "
      f"{row.get('A0_kv_mean_of_max')}/{row.get(f'{arm}_kv_mean_of_max')}"
    )
for rm in remeas:
  lines.append(f"REMEAS {rm['arm']} r={rm['rate']} pass={rm['n_pass']}/{rm['n_reps']}")
  for x in rm['reps']:
    mark = 'PASS' if x['pass'] else ('tok_only' if x['tok_only'] else ('UNFAIR' if not x['fair'] else 'lose'))
    lines.append(
      f"  rep{x['rep']} {mark} tok {x['A0_tok_s']}->{x['arm_tok_s']} "
      f"ttft {x['A0_ttft_p90']}->{x['arm_ttft_p90']}"
    )
text = "\n".join(lines) + "\n"
print(text)
sum_path = root / 'phase1_high_kv_ablation.txt'
prev = sum_path.read_text() if sum_path.exists() else ''
sum_path.write_text(prev + "\n" + text, encoding='utf-8')

pack = root / 'phase1_high_kv_ablation_pack'
pack.mkdir(exist_ok=True)
for p in ab.glob('*.json'):
  shutil.copy2(p, pack / p.name)
(pack / 'phase1_high_kv_ablation.json').write_text(json.dumps(out, indent=2))
shutil.copy2(sum_path, pack / 'phase1_high_kv_ablation.txt')
(pack / 'DONE').write_text(
  f"status=OK\nfinished_at={out['finished_at']}\n"
  f"pressure_target_hit={out['pressure_target_hit']}\n"
  f"primary_pass={out['primary_pass_by_arm']}\n"
)
with tarfile.open(root / 'phase1_high_kv_ablation_pack.tgz', 'w:gz') as tar:
  tar.add(pack, arcname='phase1_high_kv_ablation_pack')
print('PACKED', root / 'phase1_high_kv_ablation_pack.tgz')
print('FINISHED', out['finished_at'], 'pass', out['primary_pass_by_arm'])
PY

echo "PHASE1_HIGH_KV_ABLATION_DONE $(date -Is)" | tee -a "$SUMMARY"
