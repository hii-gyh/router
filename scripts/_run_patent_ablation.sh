#!/usr/bin/env bash
set -euo pipefail
export OPENSSL_DIR="${HOME}/openssl-local"
export OPENSSL_STATIC=1
export PKG_CONFIG_PATH="${OPENSSL_DIR}/lib64/pkgconfig:${OPENSSL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="${HOME}/.cargo/bin:${PATH}"
cd /mnt/c/Users/Lenovo/codes/exps/router
cargo test --features cpu,ready --test patent_ablation_sim -- --nocapture
