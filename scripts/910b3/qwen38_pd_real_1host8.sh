#!/bin/bash
# Real Mooncake PD on one 8-NPU host (TP=1). Do NOT use mock PD.
#
# Topology:
#   pd-real-prefill-0/1  short pool  NPU0-1  HTTP 18001-02  kv 30000-01  max-len 4096
#   pd-real-prefill-2/3  open pool   NPU2-3  HTTP 18003-04  kv 30002-03  max-len 16384
#   pd-real-decode-0..3  decode      NPU4-7  HTTP 18011-14  kv 30100-03  max-len 16384
#   pd-real-router       port 8090   --kv-connector nixl (passthrough); workers use MooncakeConnectorV1
#
# Usage on the NPU host:
#   bash qwen38_pd_real_1host8.sh stop
#   bash qwen38_pd_real_1host8.sh serve
#   bash qwen38_pd_real_1host8.sh wait
#   bash qwen38_pd_real_1host8.sh router
#   bash qwen38_pd_real_1host8.sh all          # stop + serve + wait + router
#   bash qwen38_pd_real_1host8.sh status
#
# Env overrides: ROOT, IMG, IFNAME, HOST_IP, MAX_NUM_SEQS, GPU_MEM_UTIL

set -euo pipefail

ROOT="${ROOT:-/root/gyh_dev/vllm_router}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.30.0rc2}"
IFNAME="${IFNAME:-data0.3002}"
HOST_IP="${HOST_IP:-192.168.151.129}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
ROUTER_BIN="${ROUTER_BIN:-$ROOT/vllm-router}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

mkdir -p "$LOG_DIR"
for i in 0 1 2 3; do
  mkdir -p "/tmp/pd-real-cache-p${i}" "/tmp/pd-real-cache-d${i}"
done

write_serve_script() {
  cat > "$ROOT/serve_pd_real.sh" << 'EOF'
#!/bin/bash
set -eo pipefail
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh ]]; then
  source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
fi
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HOME=/cache XDG_CACHE_HOME=/cache TORCH_HOME=/cache/torch
export ACL_OP_COMPILER_CACHE_DIR=/cache/acl
ROLE="$1"
PORT="$2"
KV_PORT="$3"
MAX_LEN="$4"
MAX_NUM_SEQS="${5:-8}"
GPU_MEM_UTIL="${6:-0.85}"
exec vllm serve /home/weights/Qwen3.8-27B-w8a8/ \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --quantization ascend \
  --served-model-name qwen3.8 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_LEN" \
  --max-num-batched-tokens "$MAX_LEN" \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"${ROLE}\",\"kv_port\":\"${KV_PORT}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":1},\"decode\":{\"dp_size\":1,\"tp_size\":1}}}"
EOF
  chmod +x "$ROOT/serve_pd_real.sh"
}

mount_args() {
  local -a m=(
    -v /usr/local/dcmi:/usr/local/dcmi
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
    -v /etc/ascend_install.info:/etc/ascend_install.info
    -v /etc/hccn.conf:/etc/hccn.conf
    -v /mnt/weights:/home/weights
    -v "$ROOT:$ROOT"
  )
  if [[ -d /usr/local/Ascend/ascend-toolkit ]]; then
    m+=(-v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit)
  fi
  if [[ -d /usr/local/Ascend/nnal ]]; then
    m+=(-v /usr/local/Ascend/nnal:/usr/local/Ascend/nnal)
  fi
  printf '%s\n' "${m[@]}"
}

stop_pd_real() {
  local names
  names="$(nerdctl ps -a --format '{{.Names}}' | grep '^pd-real-' || true)"
  if [[ -z "$names" ]]; then
    echo "no pd-real-* containers"
    return 0
  fi
  # shellcheck disable=SC2086
  echo "$names" | xargs -r nerdctl rm -f
  echo "removed pd-real-* containers"
}

start_one() {
  local name="$1" npu="$2" role="$3" port="$4" kv="$5" max_len="$6" cache="$7"
  local -a mounts=()
  mapfile -t mounts < <(mount_args)
  nerdctl rm -f "$name" >/dev/null 2>&1 || true
  nerdctl run -d \
    --name "$name" \
    --shm-size=16g \
    --net=host \
    --privileged \
    --device "/dev/davinci${npu}" \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    "${mounts[@]}" \
    -v "${cache}:/cache" \
    -e ASCEND_RT_VISIBLE_DEVICES="$npu" \
    -e HCCL_IF_IP="$HOST_IP" \
    -e GLOO_SOCKET_IFNAME="$IFNAME" \
    -e TP_SOCKET_IFNAME="$IFNAME" \
    -e HCCL_SOCKET_IFNAME="$IFNAME" \
    "$IMG" \
    bash "$ROOT/serve_pd_real.sh" "$role" "$port" "$kv" "$max_len" "$MAX_NUM_SEQS" "$GPU_MEM_UTIL"
  echo "started $name npu=$npu role=$role http=$port kv=$kv max_len=$max_len"
}

serve_workers() {
  write_serve_script
  # short prefill pool
  start_one pd-real-prefill-0 0 kv_producer 18001 30000 4096  /tmp/pd-real-cache-p0
  sleep 2
  start_one pd-real-prefill-1 1 kv_producer 18002 30001 4096  /tmp/pd-real-cache-p1
  sleep 2
  # open/long prefill pool
  start_one pd-real-prefill-2 2 kv_producer 18003 30002 16384 /tmp/pd-real-cache-p2
  sleep 2
  start_one pd-real-prefill-3 3 kv_producer 18004 30003 16384 /tmp/pd-real-cache-p3
  sleep 2
  # decode pool
  start_one pd-real-decode-0 4 kv_consumer 18011 30100 16384 /tmp/pd-real-cache-d0
  sleep 2
  start_one pd-real-decode-1 5 kv_consumer 18012 30101 16384 /tmp/pd-real-cache-d1
  sleep 2
  start_one pd-real-decode-2 6 kv_consumer 18013 30102 16384 /tmp/pd-real-cache-d2
  sleep 2
  start_one pd-real-decode-3 7 kv_consumer 18014 30103 16384 /tmp/pd-real-cache-d3
}

worker_names() {
  echo pd-real-prefill-0 pd-real-prefill-1 pd-real-prefill-2 pd-real-prefill-3 \
       pd-real-decode-0 pd-real-decode-1 pd-real-decode-2 pd-real-decode-3
}

worker_ports() {
  echo 18001 18002 18003 18004 18011 18012 18013 18014
}

check_hcomm_fail() {
  local c="$1"
  if nerdctl logs "$c" 2>&1 | grep -Eqi 'HcommMemReg|HCCL.+failed|RuntimeError|Traceback \(most recent'; then
    echo "FAIL: $c shows Hcomm/error in logs" >&2
    nerdctl logs --tail 80 "$c" 2>&1 | tail -80 >&2
    return 1
  fi
  return 0
}

wait_health() {
  local timeout_s="${1:-900}"
  local start now p c
  start="$(date +%s)"
  echo "waiting up to ${timeout_s}s for 8 /health endpoints..."
  while true; do
    now="$(date +%s)"
    if (( now - start > timeout_s )); then
      echo "TIMEOUT waiting for health" >&2
      status_cmd
      for c in $(worker_names); do
        echo "===== $c ====="
        nerdctl logs --tail 40 "$c" 2>&1 | tail -40
      done
      return 1
    fi
    for c in $(worker_names); do
      if ! nerdctl inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true; then
        echo "container $c not running"
        nerdctl logs --tail 60 "$c" 2>&1 | tail -60
        return 1
      fi
      if ! check_hcomm_fail "$c"; then
        return 1
      fi
    done
    local ok=0
    for p in $(worker_ports); do
      if curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
        ok=$((ok + 1))
      fi
    done
    echo "$(date -Is) healthy=${ok}/8"
    if [[ "$ok" -eq 8 ]]; then
      echo "all 8 workers healthy"
      return 0
    fi
    sleep 10
  done
}

start_router() {
  if [[ ! -x "$ROUTER_BIN" ]]; then
    echo "missing router binary: $ROUTER_BIN" >&2
    return 1
  fi
  nerdctl rm -f pd-real-router >/dev/null 2>&1 || true
  # nixl mode skips /query bootstrap; bootstrap args use kv_port (= handshake_port).
  nerdctl run -d --name pd-real-router --net=host \
    -v "$ROOT:$ROOT" \
    "$IMG" \
    "$ROUTER_BIN" --host 0.0.0.0 --port 8090 \
    --vllm-pd-disaggregation --kv-connector nixl \
    --prefill http://127.0.0.1:18001 30000 \
    --prefill http://127.0.0.1:18002 30001 \
    --prefill http://127.0.0.1:18003 30002 \
    --prefill http://127.0.0.1:18004 30003 \
    --decode http://127.0.0.1:18011 \
    --decode http://127.0.0.1:18012 \
    --decode http://127.0.0.1:18013 \
    --decode http://127.0.0.1:18014 \
    --prefill-policy prefill_bucket --decode-policy decode_kv_cost \
    --prefill-bucket-max-tokens 2048 \
    --prefill-bucket-worker http://127.0.0.1:18001=0 \
    --prefill-bucket-worker http://127.0.0.1:18002=0 \
    --disable-retries --disable-circuit-breaker
  local i
  for i in $(seq 1 60); do
    if curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1; then
      echo "router ready ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "router failed to become healthy" >&2
  nerdctl logs --tail 80 pd-real-router 2>&1 | tail -80 >&2
  return 1
}

status_cmd() {
  echo "=== $(date -Is) ==="
  nerdctl ps -a --format '{{.Names}} {{.Status}}' | grep -E 'pd-real|vllm-ascend' || true
  echo "=== health ==="
  local p
  for p in $(worker_ports) 8090; do
    if curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
      echo "$p healthy"
    else
      echo "$p DOWN"
    fi
  done
}

case "${1:-}" in
  stop) stop_pd_real ;;
  serve) serve_workers ;;
  wait) wait_health "${2:-900}" ;;
  router) start_router ;;
  status) status_cmd ;;
  all)
    stop_pd_real
    serve_workers
    wait_health 900
    start_router
    status_cmd
    ;;
  *)
    echo "usage: $0 stop|serve|wait|router|status|all" >&2
    exit 1
    ;;
esac
