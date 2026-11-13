#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/gyh_dev/vllm_router
mkdir -p "$ROOT/logs"
cd /root/gyh_dev
if [[ -d vllm_router/.git ]]; then
  cd vllm_router
  git fetch origin feat/soft-length-aware || git fetch --all
  git checkout feat/soft-length-aware
  git pull --ff-only origin feat/soft-length-aware || true
else
  rm -rf vllm_router.new
  git clone --branch feat/soft-length-aware --single-branch https://github.com/hii-gyh/router.git vllm_router.new
  # keep data/logs if any
  mkdir -p vllm_router
  rsync -a --exclude target vllm_router.new/ vllm_router/
fi
cd "$ROOT"
ls -la | head
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
rustc -vV | head -3
# system openssl headers?
if ! pkg-config --exists openssl 2>/dev/null; then
  echo "openssl pc missing; trying apt without sudo or local build"
  if [[ ! -d $HOME/openssl-local ]]; then
    echo "NEED_OPENSSL"
  fi
fi
echo BOOTSTRAP_SRC_OK
