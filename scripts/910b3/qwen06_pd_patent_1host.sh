#!/bin/bash
# Patent live retest: differentiated Prefill length pools + Decode KV cost.
# Small model Qwen3-0.6B for fast bring-up on one 910B3 host (TP=1).
#
# Topology (matches disclosure short/open Prefill + Decode pool):
#   pd-pat-prefill-0/1  short  NPU0-1  :18001-02  kv 30000-01  max-len 2048
#   pd-pat-prefill-2/3  open   NPU2-3  :18003-04  kv 30002-03  max-len 8192
#   pd-pat-decode-0/1   decode NPU4-5  :18011-12  kv 30100-01  max-len 8192
#   pd-pat-router       :8090
#
# Usage:
#   bash qwen06_pd_patent_1host.sh stop|serve|wait|router|status|all|ablation

set -euo pipefail

ROOT="${ROOT:-/root/gyh_dev/vllm_router}"
IMG="${IMG:-quay.io/ascend/vllm-ascend:v0.30.0rc2}"
IFNAME="${IFNAME:-data0.3001}"
HOST_IP="${HOST_IP:-192.168.151.55}"
MODEL="${MODEL:-/home/weights/Qwen3-0.6B}"
SERVED_NAME="${SERVED_NAME:-qwen3-0.6b}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
ROUTER_BIN="${ROUTER_BIN:-$ROOT/vllm-router}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/patent_live}"
CLIENT="${CLIENT:-$ROOT/real_pd_trajectory_client.py}"
SHORT_LEN="${SHORT_LEN:-2048}"
OPEN_LEN="${OPEN_LEN:-8192}"
BUCKET_CEILING="${BUCKET_CEILING:-1024}"

mkdir -p "$LOG_DIR" "$ROOT/data"
for i in 0 1 2 3; do mkdir -p "/tmp/pd-pat-cache-p${i}"; done
for i in 0 1; do mkdir -p "/tmp/pd-pat-cache-d${i}"; done

write_serve_script() {
  cat > "$ROOT/serve_pd_patent.sh" << EOF
#!/bin/bash
set -eo pipefail
[[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] && source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 || true
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HOME=/cache XDG_CACHE_HOME=/cache TORCH_HOME=/cache/torch
export ACL_OP_COMPILER_CACHE_DIR=/cache/acl
ROLE="\$1"; PORT="\$2"; KV_PORT="\$3"; MAX_LEN="\$4"
MAX_NUM_SEQS="\${5:-16}"; GPU_MEM_UTIL="\${6:-0.85}"
exec vllm serve ${MODEL} \\
  --host 0.0.0.0 --port "\$PORT" \\
  --tensor-parallel-size 1 \\
  --served-model-name ${SERVED_NAME} \\
  --max-num-seqs "\$MAX_NUM_SEQS" \\
  --max-model-len "\$MAX_LEN" \\
  --max-num-batched-tokens "\$MAX_LEN" \\
  --trust-remote-code \\
  --dtype bfloat16 \\
  --gpu-memory-utilization "\$GPU_MEM_UTIL" \\
  --no-enable-prefix-caching \\
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"\${ROLE}\",\"kv_port\":\"\${KV_PORT}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":1},\"decode\":{\"dp_size\":1,\"tp_size\":1}}}"
EOF
  chmod +x "$ROOT/serve_pd_patent.sh"
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
  [[ -d /usr/local/Ascend/ascend-toolkit ]] && m+=(-v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit)
  [[ -d /usr/local/Ascend/nnal ]] && m+=(-v /usr/local/Ascend/nnal:/usr/local/Ascend/nnal)
  printf '%s\n' "${m[@]}"
}

stop_all() {
  stop_router
  local names
  names="$(nerdctl ps -a --format '{{.Names}}' | grep -E '^pd-pat-' || true)"
  if [[ -n "$names" ]]; then
    # shellcheck disable=SC2086
    echo "$names" | xargs -r nerdctl rm -f
  fi
  echo "stopped pd-pat-*"
}

start_one() {
  local name="$1" npu="$2" role="$3" port="$4" kv="$5" max_len="$6" cache="$7"
  local -a mounts=()
  mapfile -t mounts < <(mount_args)
  nerdctl rm -f "$name" >/dev/null 2>&1 || true
  nerdctl run -d \
    --name "$name" --shm-size=8g --net=host --privileged \
    --device "/dev/davinci${npu}" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    "${mounts[@]}" -v "${cache}:/cache" \
    -e ASCEND_RT_VISIBLE_DEVICES="$npu" \
    -e HCCL_IF_IP="$HOST_IP" \
    -e GLOO_SOCKET_IFNAME="$IFNAME" \
    -e TP_SOCKET_IFNAME="$IFNAME" \
    -e HCCL_SOCKET_IFNAME="$IFNAME" \
    "$IMG" bash "$ROOT/serve_pd_patent.sh" "$role" "$port" "$kv" "$max_len" "$MAX_NUM_SEQS" "$GPU_MEM_UTIL"
  echo "started $name npu=$npu role=$role http=$port kv=$kv max_len=$max_len"
}

serve_workers() {
  write_serve_script
  start_one pd-pat-prefill-0 0 kv_producer 18001 30000 "$SHORT_LEN" /tmp/pd-pat-cache-p0
  sleep 2
  start_one pd-pat-prefill-1 1 kv_producer 18002 30001 "$SHORT_LEN" /tmp/pd-pat-cache-p1
  sleep 2
  start_one pd-pat-prefill-2 2 kv_producer 18003 30002 "$OPEN_LEN" /tmp/pd-pat-cache-p2
  sleep 2
  start_one pd-pat-prefill-3 3 kv_producer 18004 30003 "$OPEN_LEN" /tmp/pd-pat-cache-p3
  sleep 2
  start_one pd-pat-decode-0 4 kv_consumer 18011 30100 "$OPEN_LEN" /tmp/pd-pat-cache-d0
  sleep 2
  start_one pd-pat-decode-1 5 kv_consumer 18012 30101 "$OPEN_LEN" /tmp/pd-pat-cache-d1
}

worker_names() { echo pd-pat-prefill-0 pd-pat-prefill-1 pd-pat-prefill-2 pd-pat-prefill-3 pd-pat-decode-0 pd-pat-decode-1; }
worker_ports() { echo 18001 18002 18003 18004 18011 18012; }

wait_health() {
  local timeout_s="${1:-900}" start now ok
  start="$(date +%s)"
  echo "waiting up to ${timeout_s}s for 6 /health ..."
  while true; do
    now="$(date +%s)"
    (( now - start > timeout_s )) && { echo TIMEOUT; status_cmd; return 1; }
    for c in $(worker_names); do
      nerdctl inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true || {
        echo "$c not running"; nerdctl logs --tail 50 "$c" 2>&1 | tail -50; return 1; }
    done
    ok=0
    for p in $(worker_ports); do
      curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1 && ok=$((ok+1)) || true
    done
    echo "$(date -Is) healthy=${ok}/6"
    [[ "$ok" -eq 6 ]] && { echo all_healthy; return 0; }
    sleep 8
  done
}

router_args_for_arm() {
  local arm="$1"
  local -a args=(
    "$ROUTER_BIN" --host 0.0.0.0 --port 8090
    --vllm-pd-disaggregation --kv-connector nixl
    --prefill http://127.0.0.1:18001 30000
    --prefill http://127.0.0.1:18002 30001
    --prefill http://127.0.0.1:18003 30002
    --prefill http://127.0.0.1:18004 30003
    --decode http://127.0.0.1:18011
    --decode http://127.0.0.1:18012
    --disable-retries --disable-circuit-breaker
  )
  case "$arm" in
    A0)
      args+=(--prefill-policy round_robin --decode-policy round_robin)
      ;;
    Hard)
      args+=(
        --prefill-policy prefill_bucket --decode-policy round_robin
        --prefill-bucket-max-tokens "$BUCKET_CEILING"
        --prefill-hard-partition
        --prefill-spill-abs 6 --prefill-spill-rel 1.5
        --prefill-bucket-worker http://127.0.0.1:18001=0
        --prefill-bucket-worker http://127.0.0.1:18002=0
      )
      ;;
    SoftFull)
      args+=(
        --prefill-policy prefill_bucket --decode-policy round_robin
        --prefill-bucket-max-tokens "$BUCKET_CEILING"
        --prefill-preference-weight 2.0
        --prefill-bucket-worker http://127.0.0.1:18001=0
        --prefill-bucket-worker http://127.0.0.1:18002=0
      )
      ;;
    SoftPlusDecode)
      args+=(
        --prefill-policy prefill_bucket --decode-policy decode_kv_cost
        --prefill-bucket-max-tokens "$BUCKET_CEILING"
        --prefill-preference-weight 2.0
        --prefill-bucket-worker http://127.0.0.1:18001=0
        --prefill-bucket-worker http://127.0.0.1:18002=0
      )
      ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  printf '%s\n' "${args[@]}"
}

stop_router() {
  # Host-native router (binary links host OpenSSL 1.1; container image lacks it).
  if [[ -f "$LOG_DIR/router.pid" ]]; then
    local pid
    pid="$(cat "$LOG_DIR/router.pid" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$LOG_DIR/router.pid"
  fi
  pkill -f '/vllm-router --host 0.0.0.0 --port 8090' 2>/dev/null || true
  nerdctl rm -f pd-pat-router >/dev/null 2>&1 || true
  fuser -k 8090/tcp >/dev/null 2>&1 || true
}

start_router_arm() {
  local arm="${1:-SoftPlusDecode}"
  [[ -x "$ROUTER_BIN" ]] || { echo "missing $ROUTER_BIN"; return 1; }
  mkdir -p "$LOG_DIR"
  stop_router
  local -a args=()
  mapfile -t args < <(router_args_for_arm "$arm")
  nohup "${args[@]}" >"$LOG_DIR/router_${arm}.log" 2>&1 &
  echo $! >"$LOG_DIR/router.pid"
  local i
  for i in $(seq 1 90); do
    curl -sf -m 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && {
      echo "router ready arm=$arm ${i}s pid=$(cat "$LOG_DIR/router.pid")"; return 0; }
    sleep 1
  done
  echo "router failed arm=$arm"; tail -80 "$LOG_DIR/router_${arm}.log" || true
  return 1
}

status_cmd() {
  echo "=== $(date -Is) ==="
  nerdctl ps -a --format '{{.Names}} {{.Status}}' | grep '^pd-pat-' || true
  if [[ -f "$LOG_DIR/router.pid" ]] && kill -0 "$(cat "$LOG_DIR/router.pid")" 2>/dev/null; then
    echo "router pid=$(cat "$LOG_DIR/router.pid") alive"
  else
    echo "router host-process absent"
  fi
  for p in $(worker_ports) 8090; do
    if curl -sf -m 2 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then echo "$p healthy"
    else echo "$p DOWN"; fi
  done
}

client_env() {
  # MODEL_NAME must match --served-model-name; default in client is qwen3.8.
  printf '%s\n' \
    -e "TOKENIZER_PATH=$MODEL" \
    -e "MODEL_NAME=$SERVED_NAME" \
    -e "ROUTER_URL=http://127.0.0.1:8090/v1/chat/completions" \
    -e "SHORT_PORTS=18001,18002" \
    -e "CTX_HEADROOM=128"
}

prepare_traj() {
  local traj="$ROOT/data/trajectory_patent_live.json"
  local -a env_args=()
  mapfile -t env_args < <(client_env)
  nerdctl run --rm --net=host -v "$ROOT:$ROOT" -v /mnt/weights:/home/weights \
    "${env_args[@]}" "$IMG" \
    python3 "$CLIENT" --mode prepare-mixed --traj "$traj" --seed 20251113 \
      --short-prompt 512 --mid-prompt 1536 --long-prompt 4096 --open-prompt 6144 \
      --n 120 2>/dev/null \
    || nerdctl run --rm --net=host -v "$ROOT:$ROOT" -v /mnt/weights:/home/weights \
         "${env_args[@]}" "$IMG" \
         python3 "$CLIENT" --mode prepare --traj "$traj" --seed 20251113
  echo "traj=$traj"
}

run_arm_once() {
  local arm="$1" rate="$2" n="$3" out="$4"
  local -a env_args=()
  mapfile -t env_args < <(client_env)
  start_router_arm "$arm"
  nerdctl run --rm --net=host -v "$ROOT:$ROOT" -v /mnt/weights:/home/weights \
    "${env_args[@]}" "$IMG" \
    python3 "$CLIENT" --mode run --traj "$ROOT/data/trajectory_patent_live.json" \
      --rate "$rate" --n "$n" --label "$arm" --out "$out" --ignore-eos \
    | tee "${out%.json}.log" | tail -30
}

ablation() {
  local rate="${RATE:-1.5}" n="${FINAL_N:-60}"
  local summary="$LOG_DIR/SUMMARY.md" rows="$LOG_DIR/rows.json"
  mkdir -p "$LOG_DIR"
  echo "# Patent live ablation $(date -Is)" > "$summary"
  echo "- host=$(hostname) model=Qwen3-0.6B short=$SHORT_LEN open=$OPEN_LEN ceiling=$BUCKET_CEILING" >> "$summary"
  echo "- rate=$rate n=$n arms=A0,Hard,SoftFull,SoftPlusDecode" >> "$summary"
  echo "[]" > "$rows"
  prepare_traj || true
  for arm in A0 Hard SoftFull SoftPlusDecode; do
    local out="$LOG_DIR/${arm}_r${rate}.json"
    echo "=== ARM $arm ===" | tee -a "$summary"
    run_arm_once "$arm" "$rate" "$n" "$out" | tee -a "$summary"
  done
  echo "results in $LOG_DIR"
}

case "${1:-}" in
  stop) stop_all ;;
  serve) serve_workers ;;
  wait) wait_health "${2:-900}" ;;
  router) start_router_arm "${2:-SoftPlusDecode}" ;;
  status) status_cmd ;;
  all) stop_all; serve_workers; wait_health 900; start_router_arm SoftPlusDecode; status_cmd ;;
  ablation) ablation ;;
  *) echo "usage: $0 stop|serve|wait|router|status|all|ablation"; exit 1 ;;
esac
