#!/usr/bin/env bash
set -euo pipefail
export OPENSSL_DIR="${HOME}/openssl-local"
export OPENSSL_STATIC=1
export PKG_CONFIG_PATH="${OPENSSL_DIR}/lib64/pkgconfig:${OPENSSL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="${HOME}/.cargo/bin:${PATH}"
cd /mnt/c/Users/Lenovo/codes/exps/router
echo "OPENSSL_DIR=${OPENSSL_DIR}"
ls "${OPENSSL_DIR}/include/openssl/ssl.h"
ls "${OPENSSL_DIR}/lib64/libssl.a" "${OPENSSL_DIR}/lib/libssl.a" 2>/dev/null || true
cargo test --features cpu,ready soft_length_aware --lib -- --nocapture
cargo test --features cpu,ready --test soft_length_aware_sim -- --nocapture
cargo test --features cpu,ready --bin vllm-router parses_soft_length_aware_prefill_policy -- --nocapture
