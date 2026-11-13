#!/usr/bin/env bash
# Build aarch64 vllm-router on the NPU host (or fetch rust + build).
set -euo pipefail
ROOT=/root/gyh_dev/vllm_router
cd "$ROOT"
if [[ ! -f Cargo.toml ]]; then
  echo "missing source tree at $ROOT"; exit 2
fi
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  echo "installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
rustc -vV | head -5
# Prefer vendored openssl if system headers missing.
if [[ -d "$HOME/openssl-local" ]]; then
  export OPENSSL_DIR="$HOME/openssl-local" OPENSSL_STATIC=1
fi
echo "building release binary..."
cargo build --release --bin vllm-router 2>&1 | tee "$ROOT/logs/cargo_build_router.log" | tail -40
cp -f target/release/vllm-router "$ROOT/vllm-router"
chmod +x "$ROOT/vllm-router"
"$ROOT/vllm-router" --help 2>&1 | head -5
"$ROOT/vllm-router" --help 2>&1 | grep -E 'prefill.bucket|soft.length|decode.kv' || true
echo BUILD_OK
