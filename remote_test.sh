#!/bin/bash
set -eo pipefail
ROOT=/root/gyh_dev/vllm_router
IMG=quay.io/ascend/vllm-ascend:v0.30.0rc2
echo "=== test run start $(date -Is) ==="
# CARGO_HOME and the target dir stay inside the container: cargo's package
# cache lock hangs when CARGO_HOME sits on a bind mount.
nerdctl run --rm --net=host --name pd-unit-test \
  -v "$ROOT:/src" \
  -e CARGO_HOME=/tmp/cargo-home \
  -e CARGO_TARGET_DIR=/tmp/cargo-target \
  "$IMG" bash -lc '
    set -eo pipefail
    export DEBIAN_FRONTEND=noninteractive
    pkg-config --version >/dev/null 2>&1 || {
      apt-get update -qq >/dev/null 2>&1
      apt-get install -y -qq pkg-config libssl-dev >/dev/null 2>&1
    }
    pkg-config --version >/dev/null || { echo "pkg-config missing"; exit 1; }
    cd /src
    ulimit -n 4096 || true
    # Unbuffered so progress is visible in the log while it runs.
    stdbuf -oL -eL cargo test --test pd_length_policy_sim -j2 2>&1
  '
echo "=== test run done $(date -Is) ==="
