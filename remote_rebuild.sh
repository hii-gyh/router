#!/bin/bash
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
LOG=$ROOT/logs/rebuild.log
echo "=== rebuild start $(date -Is) ===" | tee "$LOG"
nerdctl run --rm --net=host \
  -v "$ROOT:/src" \
  -e CARGO_HOME=/tmp/cargo-home \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  "$IMG" bash -lc '
    set -eo pipefail
    export DEBIAN_FRONTEND=noninteractive
    (pkg-config --version >/dev/null 2>&1 || apt-get update -qq && apt-get install -y -qq pkg-config libssl-dev) >/dev/null 2>&1 || true
    pkg-config --version || { echo "pkg-config still missing"; exit 1; }
    cd /src
    ulimit -n 4096 || true
    cargo build --release -j2
    cp -f /tmp/cargo-target/release/vllm-router /src/vllm-router.new
    echo BUILD_OK
    cargo test --release --test pd_length_policy_sim -j2 2>&1 | tail -30
  ' 2>&1 | tee -a "$LOG"
echo "=== rebuild done $(date -Is) ===" | tee -a "$LOG"
ls -lh "$ROOT/vllm-router.new" 2>/dev/null || echo 'no new binary'
