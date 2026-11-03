#!/bin/bash
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
RESULT=$LOGDIR/compare_result.json

snap_metrics() {
  python3 - <<'PY'
import json, urllib.request
out = {}
for port in range(18001, 18009):
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    row = {"prompt_tokens": 0.0, "generation_tokens": 0.0, "running": 0.0, "waiting": 0.0, "kv": 0.0}
    for line in text.splitlines():
        if line.startswith("vllm:prompt_tokens_total{"):
            row["prompt_tokens"] = float(line.split()[-1])
        elif line.startswith("vllm:generation_tokens_total{"):
            row["generation_tokens"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_running{"):
            row["running"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_waiting{") and "reason" not in line:
            row["waiting"] = float(line.split()[-1])
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            row["kv"] = float(line.split()[-1])
    out[str(port)] = row
print(json.dumps(out))
PY
}

restart_router() {
  local policy="$1"
  nerdctl rm -f pd-exp-router >/dev/null 2>&1 || true
  local args=(
    /src/vllm-router
    --host 0.0.0.0 --port 8090
    --worker-urls
    http://127.0.0.1:18001 http://127.0.0.1:18002
    http://127.0.0.1:18003 http://127.0.0.1:18004
    http://127.0.0.1:18005 http://127.0.0.1:18006
    http://127.0.0.1:18007 http://127.0.0.1:18008
    --disable-retries --disable-circuit-breaker
  )
  if [[ "$policy" == "round_robin" ]]; then
    args+=(--policy round_robin)
  else
    args+=(
      --policy prefill_bucket
      --prefill-bucket-max-tokens 2048
      --prefill-bucket-worker http://127.0.0.1:18001=0
      --prefill-bucket-worker http://127.0.0.1:18002=0
      --prefill-bucket-worker http://127.0.0.1:18003=0
      --prefill-bucket-worker http://127.0.0.1:18004=0
    )
  fi
  nerdctl run -d --name pd-exp-router --net=host \
    -v "$ROOT:/src" \
    "$IMG" \
    "${args[@]}" >/dev/null
  # wait healthy
  for i in $(seq 1 60); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready policy=$policy after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "router failed to become healthy"
  nerdctl logs --tail 80 pd-exp-router || true
  return 1
}

run_client() {
  local out="$1"
  # run inside ascend image so transformers + weights path exist
  nerdctl run --rm --net=host \
    -v "$ROOT:/workspace/vllm_router" \
    -v /mnt/weights:/home/weights \
    "$IMG" \
    python3 /workspace/vllm_router/compare_routing.py > "$out" 2>&1
}

delta_json() {
  python3 - "$1" "$2" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
d={}
for k in a:
    d[k]={m: b[k][m]-a[k][m] for m in a[k]}
print(json.dumps(d))
PY
}

parse_client() {
  python3 - "$1" <<'PY'
import json,sys,re
text=open(sys.argv[1],encoding='utf-8',errors='replace').read().splitlines()
rows={}
for line in text:
    m=re.match(r'(short|long)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)$', line.strip())
    if m:
        rows[m.group(1)]={"n":int(m.group(2)),"ok":int(m.group(3)),"ttft_p50":None if m.group(4)=='-' else float(m.group(4)),"ttft_p90":None if m.group(5)=='-' else float(m.group(5))}
print(json.dumps(rows))
PY
}

echo "=== A/B compare start $(date -Is) ==="
declare -A BEFORE AFTER CLIENT POLICY_JSON
FINAL_PARTS=()

for policy in round_robin length; do
  echo "=== policy=$policy ==="
  restart_router "$policy"
  sleep 2
  snap_metrics > "$LOGDIR/before_${policy}.json"
  run_client "$LOGDIR/${policy}_run.txt"
  snap_metrics > "$LOGDIR/after_${policy}.json"
  delta_json "$LOGDIR/before_${policy}.json" "$LOGDIR/after_${policy}.json" > "$LOGDIR/delta_${policy}.json"
  parse_client "$LOGDIR/${policy}_run.txt" > "$LOGDIR/ttft_${policy}.json"
  echo "--- ttft ---"; cat "$LOGDIR/ttft_${policy}.json"; echo
  echo "--- delta prompt_tokens ---"
  python3 - "$LOGDIR/delta_${policy}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
short=sum(d[str(p)]['prompt_tokens'] for p in range(18001,18005))
long=sum(d[str(p)]['prompt_tokens'] for p in range(18005,18009))
print('short_pool_prompt_tokens', short)
print('long_pool_prompt_tokens', long)
for p in range(18001,18009):
    print(p, int(d[str(p)]['prompt_tokens']))
PY
done

python3 - <<PY
import json, pathlib
root = pathlib.Path("$LOGDIR")
out = {
  "host": "39.156.164.118",
  "model": "qwen3.8 / Qwen3.8-27B-w8a8",
  "workers": 8,
  "short_bucket_ports": [18001,18002,18003,18004],
  "open_bucket_ports": [18005,18006,18007,18008],
  "bucket_max_tokens": 2048,
  "workload": "4 waves x (8 long@8192 then 8 short@256), max_tokens=16, concurrent",
  "note": "Regular mode (Mooncake PD unavailable). Compares round_robin vs prefill_bucket on identical 8 plain vLLM workers.",
  "policies": {}
}
for policy in ("round_robin", "length"):
    out["policies"][policy] = {
        "ttft": json.loads((root / f"ttft_{policy}.json").read_text()),
        "prompt_token_delta": json.loads((root / f"delta_{policy}.json").read_text()),
    }
    d = out["policies"][policy]["prompt_token_delta"]
    out["policies"][policy]["short_pool_prompt_tokens"] = sum(d[str(p)]["prompt_tokens"] for p in range(18001,18005))
    out["policies"][policy]["long_pool_prompt_tokens"] = sum(d[str(p)]["prompt_tokens"] for p in range(18005,18009))
rr = out["policies"]["round_robin"]["ttft"]
ln = out["policies"]["length"]["ttft"]
gains = {}
for kind in ("short", "long"):
    gains[kind] = {}
    for metric in ("ttft_p50", "ttft_p90"):
        a = rr[kind][metric]; b = ln[kind][metric]
        gains[kind][metric] = {
            "round_robin": a,
            "prefill_bucket": b,
            "delta_s": None if a is None or b is None else round(a - b, 3),
            "speedup": None if not a or not b else round(a / b, 2),
            "reduction_pct": None if not a or not b else round((a - b) / a * 100, 1),
        }
out["gains"] = gains
pathlib.Path("$RESULT").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("WROTE", "$RESULT")
print(json.dumps(gains, indent=2))
PY
echo "=== A/B compare done $(date -Is) ==="
