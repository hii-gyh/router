#!/bin/bash
# Qwen3.8-27B-w8a8 PD length-policy experiment, 16 TP=2 processes.
#
# TP=2 uses two cards per process. Sixteen processes need 32 cards, which is
# four 8-card machines:
#
#   machine     role            local indexes  HTTP           kv_port    bucket
#   SHORT_HOST  prefill short   0..3           18001-18004    30000-30003  0
#   LONG_HOST   prefill long    0..3           18001-18004    30100-30103  open
#   DECODE_A    decode          0..3           18011-18014    30200-30203  -
#   DECODE_B    decode          0..3           18011-18014    30300-30303  -
#
# Devices on every machine: 0,1 / 2,3 / 4,5 / 6,7.
# Short max-model-len is 4096. Long prefill and every decode use 16384.
# HTTP ports are reused across machines; Mooncake kv_port is unique per process.
#
# Two 8-card machines are 16 cards, which is 8 TP=2 processes, not 16.
# Sixteen processes on two machines requires TP=1.
#
# On each NPU machine:
#   export HOST_IP=<this host>
#   export IFNAME=<HCCL nic>
#   bash qwen38_pd_length.sh prefill-short all
#   bash qwen38_pd_length.sh prefill-long all
#   bash qwen38_pd_length.sh decode-a all
#   bash qwen38_pd_length.sh decode-b all
#
# Router, from a host that can reach all four:
#   export SHORT_HOST=10.0.0.1 LONG_HOST=10.0.0.2
#   export DECODE_A_HOST=10.0.0.3 DECODE_B_HOST=10.0.0.4
#   bash qwen38_pd_length.sh router round_robin
#   bash qwen38_pd_length.sh router length

set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3.8-27B-w8a8/}"
export VLLM_USE_MODELSCOPE=True
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

SERVED_NAME=qwen3.8
BOOTSTRAP_PORT=8998
TP=2
DP=1
LOG_DIR="${LOG_DIR:-./pd-logs}"

worker_env() {
    export HCCL_IF_IP="${HOST_IP:?set HOST_IP to this host}"
    export GLOO_SOCKET_IFNAME="${IFNAME:?set IFNAME to the HCCL nic}"
    export TP_SOCKET_IFNAME="${IFNAME}"
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
}

devices_for() {
    case "$1" in
        0) echo 0,1 ;;
        1) echo 2,3 ;;
        2) echo 4,5 ;;
        3) echo 6,7 ;;
        *) echo "local index must be 0..3 or all" >&2; return 1 ;;
    esac
}

kv_config() {
    local role="$1"
    local kv_port="$2"
    printf '{"kv_connector":"MooncakeConnectorV1","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":%s,"tp_size":%s},"decode":{"dp_size":%s,"tp_size":%s}}}' \
        "$role" "$kv_port" "$DP" "$TP" "$DP" "$TP"
}

serve_common() {
    local port="$1"
    local max_len="$2"
    echo \
        --host 0.0.0.0 \
        --port "${port}" \
        --data-parallel-size "${DP}" \
        --tensor-parallel-size "${TP}" \
        --quantization ascend \
        --served-model-name "${SERVED_NAME}" \
        --max-num-seqs 32 \
        --max-model-len "${max_len}" \
        --max-num-batched-tokens 16384 \
        --trust-remote-code \
        --allowed-local-media-path / \
        --dtype bfloat16 \
        --reasoning-parser qwen3 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.85 \
        --additional-config '{"enable_cpu_binding":true}'
}

launch() {
    local name="$1"
    shift
    mkdir -p "${LOG_DIR}"
    if [[ "${FOREGROUND:-0}" == 1 ]]; then
        "$@"
        return
    fi
    nohup "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
    echo $! >"${LOG_DIR}/${name}.pid"
    echo "${name} pid $(cat "${LOG_DIR}/${name}.pid") log ${LOG_DIR}/${name}.log"
}

start_prefill_one() {
    local kind="$1"
    local index="$2"
    local devices http_port kv_base max_len
    devices="$(devices_for "${index}")"
    case "${kind}" in
        short) http_port=$((18001 + index)); kv_base=30000; max_len=4096 ;;
        long) http_port=$((18001 + index)); kv_base=30100; max_len=16384 ;;
        *) echo "prefill kind must be short or long" >&2; exit 1 ;;
    esac
    export ASCEND_RT_VISIBLE_DEVICES="${devices}"
    launch "prefill-${kind}-${index}" \
        vllm serve "${MODEL_PATH}" \
        $(serve_common "${http_port}" "${max_len}") \
        --kv-transfer-config "$(kv_config kv_producer $((kv_base + index)))"
}

start_decode_one() {
    local group="$1"
    local index="$2"
    local devices kv_base
    devices="$(devices_for "${index}")"
    case "${group}" in
        a) kv_base=30200 ;;
        b) kv_base=30300 ;;
        *) echo "decode group must be a or b" >&2; exit 1 ;;
    esac
    export ASCEND_RT_VISIBLE_DEVICES="${devices}"
    launch "decode-${group}-${index}" \
        vllm serve "${MODEL_PATH}" \
        $(serve_common $((18011 + index)) 16384) \
        --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 5}' \
        --compilation-config '{"cudagraph_capture_sizes": [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --kv-transfer-config "$(kv_config kv_consumer $((kv_base + index)))"
}

start_group() {
    local starter="$1"
    local which="$2"
    local index="${3:-all}"
    worker_env
    if [[ "${index}" == all ]]; then
        local i
        for i in 0 1 2 3; do
            "${starter}" "${which}" "${i}"
        done
        return
    fi
    FOREGROUND=1 "${starter}" "${which}" "${index}"
}

start_router() {
    local policy="$1"
    local bin="${ROUTER_BIN:-./target/release/vllm-router}"
    : "${SHORT_HOST:?set SHORT_HOST}"
    : "${LONG_HOST:?set LONG_HOST}"
    : "${DECODE_A_HOST:?set DECODE_A_HOST}"
    : "${DECODE_B_HOST:?set DECODE_B_HOST}"
    local -a args=(
        --host 0.0.0.0 --port 8090
        --vllm-pd-disaggregation
        --kv-connector mooncake
        --prometheus-host 0.0.0.0 --prometheus-port 29000
    )
    local port host
    for port in 18001 18002 18003 18004; do
        args+=(--prefill "http://${SHORT_HOST}:${port}" "${BOOTSTRAP_PORT}")
        args+=(--prefill "http://${LONG_HOST}:${port}" "${BOOTSTRAP_PORT}")
    done
    for host in "${DECODE_A_HOST}" "${DECODE_B_HOST}"; do
        for port in 18011 18012 18013 18014; do
            args+=(--decode "http://${host}:${port}")
        done
    done
    case "${policy}" in
        round_robin)
            args+=(--prefill-policy round_robin --decode-policy round_robin)
            ;;
        length)
            args+=(--prefill-policy prefill_bucket --decode-policy decode_kv_cost --prefill-bucket-max-tokens 2048)
            for port in 18001 18002 18003 18004; do
                args+=(--prefill-bucket-worker "http://${SHORT_HOST}:${port}=0")
            done
            ;;
        *) echo "router policy must be round_robin or length" >&2; exit 1 ;;
    esac
    exec "${bin}" "${args[@]}"
}

case "${1:-}" in
    prefill-short) start_group start_prefill_one short "${2:-all}" ;;
    prefill-long) start_group start_prefill_one long "${2:-all}" ;;
    decode-a) start_group start_decode_one a "${2:-all}" ;;
    decode-b) start_group start_decode_one b "${2:-all}" ;;
    router) start_router "${2:?round_robin or length}" ;;
    *)
        echo "usage: $0 prefill-short|prefill-long|decode-a|decode-b [0|1|2|3|all]" >&2
        echo "       $0 router round_robin|length" >&2
        exit 1
        ;;
esac
