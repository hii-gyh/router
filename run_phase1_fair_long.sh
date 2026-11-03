#!/bin/bash
# Phase1 FAIR long-prompt coarse grid.
# A1 = soft prefill_bucket + decode_kv_cost; A0 = RR+RR. No hybrid RR.
#
# Fairness fix for p that overflows short P@4096:
#   register ONLY open prefills 18003/18004 (@16384) for BOTH arms.
# Mid-long that still fits short (p+o <= 4096) uses full 4P.
# Discard cells with either-arm error_rate > ERR_MAX (default 0.05).
#
# Dual criteria: A1.tok_s > A0.tok_s AND A1.ttft_p90 <= A0.ttft_p90
# Caps: worker max_model_len=16384 → no p=32k cells on this topology.
set -eo pipefail

ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
CLIENT=$ROOT/real_pd_trajectory_client.py
AB_DIR=$LOGDIR/phase1_fair_long
RESULT=$LOGDIR/phase1_fair_long.json
SUMMARY=$LOGDIR/phase1_fair_long.txt
PACK=$LOGDIR/phase1_fair_long_pack
mkdir -p "$LOGDIR" "$AB_DIR" "$PACK"

FINAL_N="${FINAL_N:-40}"
WARM_N="${WARM_N:-8}"
SEED="${SEED:-20251103}"
ERR_MAX="${ERR_MAX:-0.05}"

echo "PHASE1 FAIR LONG GRID $(date -Is)" | tee "$SUMMARY"
echo "criteria: A1.tok_s>A0.tok_s AND A1.ttft_p90<=A0.ttft_p90" | tee -a "$SUMMARY"
echo "A1=soft prefill_bucket + decode_kv_cost; A0=RR+RR" | tee -a "$SUMMARY"
echo "fair: open-only 2P when prompt overflows short@4096; discard err>$ERR_MAX" | tee -a "$SUMMARY"
echo "always --ignore-eos; same traj/seed/tokenizer" | tee -a "$SUMMARY"

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

# pool=full → 4P (2 short@4096 + 2 open@16384); pool=open → 2 open only (fair long)
restart_router() {
  local arm="$1"
  local pool="$2"
  nerdctl rm -f pd-real-router >/dev/null 2>&1 || true
  local -a args=(
    "$ROOT/vllm-router" --host 0.0.0.0 --port 8090
    --vllm-pd-disaggregation --kv-connector nixl
  )
  if [[ "$pool" == "open" ]]; then
    args+=(
      --prefill http://127.0.0.1:18003 30002 --prefill http://127.0.0.1:18004 30003
    )
  else
    args+=(
      --prefill http://127.0.0.1:18001 30000 --prefill http://127.0.0.1:18002 30001
      --prefill http://127.0.0.1:18003 30002 --prefill http://127.0.0.1:18004 30003
    )
  fi
  args+=(
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
    )
    if [[ "$pool" == "full" ]]; then
      args+=(
        --prefill-bucket-worker http://127.0.0.1:18001=0
        --prefill-bucket-worker http://127.0.0.1:18002=0
      )
    fi
    # open-only pool: no short bucket workers; soft policy still active over open tier
  fi
  nerdctl run -d --name pd-real-router --net=host -v "$ROOT:$ROOT" "$IMG" "${args[@]}" >/dev/null
  for i in $(seq 1 90); do
    curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && {
      echo "router ready arm=$arm pool=$pool (${i}s)"; return 0
    }
    sleep 1
  done
  nerdctl logs --tail 40 pd-real-router 2>&1 | tail -40; return 1
}

if "$ROOT/vllm-router" --help 2>&1 | grep -q 'decode-rr-below'; then
  echo "FATAL: hybrid RR CLI still present" | tee -a "$SUMMARY"; exit 5
fi
echo "binary_no_hybrid_cli=ok" | tee -a "$SUMMARY"
[[ -x "$ROOT/vllm-router" ]] || { echo missing binary; exit 2; }
workers_ok || exit 3

# cell: prompt:out:rates:pool
# pool=full if p+out fits short@4096; else open (fair long, both arms)
CELLS=(
  # optional mid-long on full 4P (fits short maxlen)
  "3072:256:2.5,4.0:full"
  # long / higher-KV on open-only 2P@16384 (fair vs prior A0 50% err)
  "4096:256:2.0,3.5:open"
  "4096:1024:1.0,2.0:open"
  "8192:256:1.0,2.0:open"
  "8192:1024:0.8,1.4:open"
  "12288:256:0.6,1.0:open"
)

echo "[]" > "$AB_DIR/rows.json"

for cell in "${CELLS[@]}"; do
  IFS=':' read -r P O RATES_CSV POOL <<<"$cell"
  TRAJ=$ROOT/data/trajectory_fixed_p${P}_o${O}.json
  echo "=== CELL p=$P o=$O rates=$RATES_CSV pool=$POOL $(date -Is) ===" | tee -a "$SUMMARY"
  # capacity sanity: prompt+out must fit open@16384
  if [[ $((P + O)) -gt 16384 ]]; then
    echo "SKIP overflow p+o=$((P+O)) > 16384" | tee -a "$SUMMARY"
    continue
  fi
  run_client --mode prepare-fixed --traj "$TRAJ" --seed "$SEED" --n "$FINAL_N" \
    --prompt-tokens "$P" --max-tokens "$O" | tee -a "$SUMMARY"
  IFS=',' read -r -a RATES <<<"$RATES_CSV"

  for arm in A0 A1; do
    restart_router "$arm" "$POOL"
    host_client_idle || true
    run_client --mode warmup --warmup-n "$WARM_N" | tee -a "$SUMMARY"
    host_client_idle || true
    for r in "${RATES[@]}"; do
      workers_ok || exit 3
      label="${arm}_p${P}_o${O}_r${r}"
      out="$AB_DIR/${label}.json"
      echo "=== run $label pool=$POOL $(date -Is) ===" | tee -a "$SUMMARY"
      # always ignore_eos for fair long/KV hold (same traj + max_tokens)
      run_client --mode run --traj "$TRAJ" --rate "$r" --n "$FINAL_N" --offset 0 \
        --label "$label" --out "$out" --ignore-eos \
        | tee "$AB_DIR/${label}.log" | tail -20
      host_client_idle || true
    done
  done

  python3 - <<PY
import json, pathlib
ab = pathlib.Path("$AB_DIR")
p, o = int("$P"), int("$O")
pool = "$POOL"
err_max = float("$ERR_MAX")
rates = [float(x) for x in "$RATES_CSV".split(",")]
rows = json.loads((ab/"rows.json").read_text())
for r in rates:
  a0 = json.loads((ab/f"A0_p{p}_o{o}_r{r}.json").read_text())
  a1 = json.loads((ab/f"A1_p{p}_o{o}_r{r}.json").read_text())
  k0 = a0.get("decode_kv") or {}
  k1 = a1.get("decode_kv") or {}
  e0 = float(a0.get("error_rate") or 0.0)
  e1 = float(a1.get("error_rate") or 0.0)
  fair = (e0 <= err_max) and (e1 <= err_max)
  ots_win = (a1.get("output_tok_s") or 0) > (a0.get("output_tok_s") or 0)
  ttft_ok = (a1.get("ttft_p90") or 1e9) <= (a0.get("ttft_p90") or 0)
  dual = bool(ots_win and ttft_ok)
  row = {
    "prompt_tokens": p, "max_tokens": o, "rate": r, "pool": pool,
    "A0_tok_s": a0.get("output_tok_s"), "A1_tok_s": a1.get("output_tok_s"),
    "A0_ttft_p90": a0.get("ttft_p90"), "A1_ttft_p90": a1.get("ttft_p90"),
    "A0_error_rate": e0, "A1_error_rate": e1,
    "A0_kv_mean_of_max": k0.get("kv_mean_of_max"), "A1_kv_mean_of_max": k1.get("kv_mean_of_max"),
    "A0_kv_peak_max": k0.get("kv_peak_max"), "A1_kv_peak_max": k1.get("kv_peak_max"),
    "fair": fair, "ots_win": ots_win, "ttft_ok": ttft_ok,
    "pass": bool(fair and dual),
    "discard_reason": None if fair else f"unfair_err e0={e0} e1={e1} > {err_max}",
  }
  rows.append(row)
  print(json.dumps(row))
(ab/"rows.json").write_text(json.dumps(rows, indent=2))
PY
done

python3 - <<'PY'
import json, pathlib, datetime, shutil, tarfile
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
ab = root/'phase1_fair_long'
rows = json.loads((ab/'rows.json').read_text())
fair_rows = [r for r in rows if r.get('fair')]
unfair = [r for r in rows if not r.get('fair')]
wins = [r for r in fair_rows if r.get('pass')]
tok_only = [r for r in fair_rows if r.get('ots_win') and not r.get('ttft_ok')]
ttft_only = [r for r in fair_rows if r.get('ttft_ok') and not r.get('ots_win')]
out = {
  'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
  'scheme': 'Phase1 FAIR long: A1 soft prefill_bucket + decode_kv_cost vs A0 RR+RR',
  'fairness': 'open-only 2P@16384 when p overflows short@4096; discard err>0.05; always ignore_eos',
  'criteria': ['A1.output_tok_s > A0.output_tok_s', 'A1.ttft_p90 <= A0.ttft_p90', 'fair error rates'],
  'n_cells': len(rows),
  'n_fair': len(fair_rows),
  'n_unfair_discarded': len(unfair),
  'n_pass': len(wins),
  'n_tok_only': len(tok_only),
  'n_ttft_only': len(ttft_only),
  'wins': wins,
  'tok_only': tok_only,
  'ttft_only': ttft_only,
  'unfair': unfair,
  'rows': rows,
}
(root/'phase1_fair_long.json').write_text(json.dumps(out, indent=2))
lines = [
  f"n_cells={len(rows)} n_fair={len(fair_rows)} n_unfair={len(unfair)} n_pass={len(wins)}",
  "WINS (fair dual-PASS):" if wins else "WINS (fair dual-PASS): none",
]
for w in wins:
  lines.append(
    f"  PASS p={w['prompt_tokens']} o={w['max_tokens']} r={w['rate']} pool={w['pool']} "
    f"tok {w['A0_tok_s']}->{w['A1_tok_s']} ttft {w['A0_ttft_p90']:.3f}->{w['A1_ttft_p90']:.3f} "
    f"kv {w.get('A0_kv_mean_of_max')}->{w.get('A1_kv_mean_of_max')}"
  )
lines.append("TOK_ONLY (fair):" if tok_only else "TOK_ONLY (fair): none")
for w in tok_only:
  lines.append(
    f"  tok_only p={w['prompt_tokens']} o={w['max_tokens']} r={w['rate']} "
    f"tok {w['A0_tok_s']}->{w['A1_tok_s']} ttft {w['A0_ttft_p90']:.3f}->{w['A1_ttft_p90']:.3f}"
  )
lines.append("UNFAIR discarded:" if unfair else "UNFAIR discarded: none")
for w in unfair:
  lines.append(
    f"  discard p={w['prompt_tokens']} o={w['max_tokens']} r={w['rate']} "
    f"err {w['A0_error_rate']}/{w['A1_error_rate']}"
  )
sum_path = root/'phase1_fair_long.txt'
sum_path.write_text(sum_path.read_text() + "\n" + "\n".join(lines) + "\n", encoding='utf-8')
pack = root/'phase1_fair_long_pack'
pack.mkdir(exist_ok=True)
for p in ab.glob('*.json'):
  shutil.copy2(p, pack/p.name)
(pack/'phase1_fair_long.json').write_text(json.dumps(out, indent=2))
shutil.copy2(sum_path, pack/'phase1_fair_long.txt')
with tarfile.open(root/'phase1_fair_long_pack.tgz', 'w:gz') as tar:
  tar.add(pack, arcname='phase1_fair_long_pack')
print('PACKED', root/'phase1_fair_long_pack.tgz')
print('PASS_COUNT', len(wins), 'FAIR', len(fair_rows), 'UNFAIR', len(unfair))
PY

echo "PHASE1_FAIR_LONG_DONE $(date -Is)" | tee -a "$SUMMARY"
