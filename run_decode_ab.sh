#!/bin/bash
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"

start_router() {
  local dpolicy="$1"
  nerdctl rm -f pd-exp-router >/dev/null 2>&1 || true
  nerdctl run -d --name pd-exp-router --net=host -v "$ROOT:/src" "$IMG" \
    /src/vllm-router --host 0.0.0.0 --port 8090 \
    --vllm-pd-disaggregation --kv-connector nixl \
    --prefill http://127.0.0.1:18001 --prefill http://127.0.0.1:18002 \
    --prefill http://127.0.0.1:18003 --prefill http://127.0.0.1:18004 \
    --decode http://127.0.0.1:18005 --decode http://127.0.0.1:18006 \
    --decode http://127.0.0.1:18007 --decode http://127.0.0.1:18008 \
    --prefill-policy round_robin --decode-policy "$dpolicy" \
    --disable-retries --disable-circuit-breaker >/dev/null
  for i in $(seq 1 90); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready decode_policy=$dpolicy (${i}s)"
      return 0
    fi
    sleep 1
  done
  echo "router not healthy for $dpolicy"
  nerdctl logs --tail 40 pd-exp-router || true
  return 1
}

echo "=== decode A/B start $(date -Is) ==="
for dp in round_robin decode_kv_cost; do
  echo "=== decode_policy=$dp ==="
  start_router "$dp"
  sleep 3
  nerdctl run --rm --net=host \
    -v "$ROOT:/workspace/vllm_router" -v /mnt/weights:/home/weights \
    "$IMG" python3 /workspace/vllm_router/decode_compare.py \
    > "$LOGDIR/decode_${dp}.txt" 2>&1 || echo "client exited nonzero for $dp"
  grep -o 'RESULT_JSON .*' "$LOGDIR/decode_${dp}.txt" | sed 's/^RESULT_JSON //' > "$LOGDIR/decode_${dp}.json" || true
  echo "--- $dp ---"
  cat "$LOGDIR/decode_${dp}.json" 2>/dev/null || tail -20 "$LOGDIR/decode_${dp}.txt"
  echo
done

python3 - <<'PY'
import json, pathlib
root = pathlib.Path('/root/gyh_dev/vllm_router/logs')
try:
    rr = json.loads((root / 'decode_round_robin.json').read_text())
    kc = json.loads((root / 'decode_decode_kv_cost.json').read_text())
except Exception as exc:
    print('cannot summarize:', exc)
    raise SystemExit(0)
summary = {'round_robin': rr, 'decode_kv_cost': kc, 'comparison': {}}
def cmp(name, a, b, lower_better=True):
    if a is None or b is None:
        return
    summary['comparison'][name] = {
        'round_robin': a, 'decode_kv_cost': b,
        'delta': round(b - a, 4),
        'pct': round((a - b) / a * 100, 1) if a else None,
        'better': 'decode_kv_cost' if ((b < a) == lower_better and b != a) else ('tie' if a == b else 'round_robin'),
    }
cmp('kv_spread_mean', rr['kv_spread_mean'], kc['kv_spread_mean'])
cmp('kv_spread_p90', rr['kv_spread_p90'], kc['kv_spread_p90'])
cmp('kv_stddev_mean', rr['kv_stddev_mean'], kc['kv_stddev_mean'])
cmp('kv_peak_max', rr['kv_peak_max'], kc['kv_peak_max'])
cmp('max_waiting', rr['max_waiting_any_worker'], kc['max_waiting_any_worker'])
cmp('waiting_sample_sum', rr['waiting_sample_sum'], kc['waiting_sample_sum'])
for kind in sorted(set(rr['ttft']) & set(kc['ttft'])):
    cmp(f'ttft_{kind}_p50', rr['ttft'][kind]['p50'], kc['ttft'][kind]['p50'])
    cmp(f'ttft_{kind}_p90', rr['ttft'][kind]['p90'], kc['ttft'][kind]['p90'])
    cmp(f'e2e_{kind}_p90', rr['e2e'][kind]['p90'], kc['e2e'][kind]['p90'])
(root / 'decode_compare_result.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary['comparison'], indent=2, ensure_ascii=False))
PY
echo "=== decode A/B done $(date -Is) ==="
