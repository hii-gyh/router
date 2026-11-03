#!/bin/bash
# Detached on the remote host. Waits for the current unit test, then bakes a
# reusable image so the next cargo test/build does not recompile dependencies.
# Survives SSH hangup. Does not touch containers whose names start with vllm-ascend.
set -u
ROOT=/root/gyh_dev/vllm_router
LOG=$ROOT/logs/pack_image.log
DONE=$ROOT/logs/pack_image.done
BUILD=/root/gyh_dev/cargo-build
BASE=quay.io/ascend/vllm-ascend:v0.30.0rc2
DEPS=vllm-router-verify:deps
FINAL=vllm-router-verify:20261030
PACK=vllm-router-verify-pack
mkdir -p "$ROOT/logs"
exec >>"$LOG" 2>&1
echo "=== pack start $(date -Is) pid=$$ ==="
echo $$ > "$ROOT/logs/pack_image.pid"
trap '' HUP

exec 9>"$ROOT/logs/pack_image.lock"
if ! flock -n 9; then
  echo "another pack_verify_image.sh holds the lock; exiting"
  exit 0
fi

echo "disk: $(df -h / | awk 'NR==2 {print $4" free"}')"
echo "cargo-build now: $(du -sh "$BUILD" 2>/dev/null | awk '{print $1}')"

cargo_pid() {
  pgrep -n -f 'cargo test --test pd_length_policy_sim' || true
}

container_id_of() {
  local pid="$1" id=""
  [[ -z "$pid" || ! -r "/proc/$pid/cgroup" ]] && return 0
  echo "cargo pid=$pid" >&2
  echo "cgroup=$(tr '\n' ' ' < /proc/$pid/cgroup)" >&2
  id=$(grep -oE '[0-9a-f]{64}' "/proc/$pid/cgroup" | head -1 || true)
  if [[ -z "$id" && -r "/proc/$pid/mountinfo" ]]; then
    id=$(grep -oE '[0-9a-f]{64}' "/proc/$pid/mountinfo" | head -1 || true)
  fi
  if [[ -z "$id" && -r "/proc/$pid/environ" ]]; then
    id=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^HOSTNAME=//p' | head -1 || true)
  fi
  printf '%s\n' "$id"
}

build_is_mounted() {
  local pid="$1"
  [[ -z "$pid" || ! -r "/proc/$pid/mountinfo" ]] && return 1
  grep -q ' /build ' "/proc/$pid/mountinfo"
}

echo "=== nerdctl ps ==="
nerdctl ps -a --format '{{.ID}} {{.Names}} {{.Status}}' || true

CPID=$(cargo_pid)
CID=$(container_id_of "$CPID")
echo "resolved container=$CID"
if [[ -n "$CPID" ]]; then
  if build_is_mounted "$CPID"; then
    echo "MOUNT: /build is a host mount (artifacts stay in $BUILD)"
  else
    echo "MOUNT: /build is NOT a host mount (artifacts live in the container layer)"
  fi
fi

if [[ -n "$CID" ]]; then
  echo "early commit $CID -> $DEPS (captures pkg-config/libssl; pauses the container briefly)"
  if nerdctl commit "$CID" "$DEPS"; then
    echo "early commit ok"
  else
    echo "early commit failed"
  fi
else
  echo "no test container id yet; will rely on the host build dir or a later commit"
fi

echo "waiting for cargo test to exit"
for _ in $(seq 1 240); do
  CPID=$(cargo_pid)
  if [[ -z "$CPID" ]]; then
    echo "cargo gone $(date -Is)"
    break
  fi
  if (( _ % 6 == 1 )); then
    echo "$(date -Is) waiting du=$(du -sh "$BUILD" 2>/dev/null | awk '{print $1}') cargo_pid=$CPID"
  fi
  sleep 15
done

CPID=$(cargo_pid)
CID=$(container_id_of "$CPID")
if [[ -z "$CID" ]]; then
  # cargo just died; pid 1 may still be tail/bash. Grab any non-user container.
  CID=$(nerdctl ps --format '{{.ID}} {{.Names}}' | awk '$2 !~ /^vllm-ascend/ && $2 != "'"$PACK"'" {print $1; exit}')
fi
if [[ -n "$CID" ]]; then
  echo "commit-on-exit $CID -> $DEPS"
  nerdctl commit "$CID" "$DEPS" || echo "commit-on-exit failed"
fi

for _ in $(seq 1 30); do
  if grep -q 'test run done' "$ROOT/logs/unit_test.log" 2>/dev/null; then
    break
  fi
  if ! pgrep -f 'bash /root/gyh_dev/vllm_router/remote_test.sh' >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "=== unit test log tail ==="
tail -80 "$ROOT/logs/unit_test.log" 2>/dev/null || echo 'no unit_test.log'
if grep -q 'test result: ok' "$ROOT/logs/unit_test.log" 2>/dev/null; then
  RESULT=ok
elif grep -q 'test result: FAILED' "$ROOT/logs/unit_test.log" 2>/dev/null; then
  RESULT=FAILED
else
  RESULT=unknown
fi
echo "TEST_RESULT=$RESULT"

build_bytes=$(du -sb "$BUILD" 2>/dev/null | awk '{print $1}')
build_bytes=${build_bytes:-0}
echo "cargo-build bytes=$build_bytes"

free_kb=$(df -k / | awk 'NR==2 {print $4}')
need_kb=$(( build_bytes / 1024 * 13 / 10 + 2000000 ))
echo "disk free_kb=$free_kb need_kb=$need_kb"

SRC_IMAGE=$BASE
if nerdctl image inspect "$DEPS" >/dev/null 2>&1; then
  SRC_IMAGE=$DEPS
fi

if (( build_bytes < 50000000 )); then
  echo "host build dir is too small to be a compile cache"
  if nerdctl image inspect "$DEPS" >/dev/null 2>&1; then
    nerdctl tag "$DEPS" "$FINAL" || true
    echo "tagged $DEPS as $FINAL (container layer only; check /build inside it)"
  else
    echo "NO_IMAGE: compile cache missing and deps commit missing"
    echo "TEST_RESULT=$RESULT" > "$DONE"
    echo "no image" >> "$DONE"
    echo "=== pack done $(date -Is) ==="
    exit 1
  fi
else
  if (( free_kb < need_kb )); then
    echo "not enough disk to copy the cache into an image"
    echo "TEST_RESULT=$RESULT" > "$DONE"
    echo "disk short" >> "$DONE"
    exit 1
  fi
  nerdctl rm -f "$PACK" >/dev/null 2>&1 || true
  echo "starting pack container from $SRC_IMAGE"
  nerdctl run -d --name "$PACK" --net=host "$SRC_IMAGE" bash -lc 'sleep infinity'
  nerdctl exec "$PACK" bash -lc '
    if ! pkg-config --version >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y pkg-config libssl-dev
    fi
    mkdir -p /build
    grep -q CARGO_TARGET_DIR /etc/profile || cat >> /etc/profile << "EOF"
export CARGO_HOME=/build/home
export CARGO_TARGET_DIR=/build/target
EOF
  '
  echo "copying $BUILD -> container /build"
  tar -C "$BUILD" -cf - . | nerdctl exec -i "$PACK" tar -C /build -xf -
  if [[ ${PIPESTATUS[0]} -ne 0 || ${PIPESTATUS[1]} -ne 0 ]]; then
    echo "copy into image failed"
    nerdctl rm -f "$PACK" >/dev/null 2>&1 || true
    echo "TEST_RESULT=$RESULT" > "$DONE"
    echo "copy failed" >> "$DONE"
    exit 1
  fi
  nerdctl exec "$PACK" bash -lc 'du -sh /build /build/target /build/home; pkg-config --version'
  echo "committing $FINAL"
  if nerdctl commit --help 2>&1 | grep -q -- '--change'; then
    nerdctl commit \
      --change 'ENV CARGO_HOME=/build/home' \
      --change 'ENV CARGO_TARGET_DIR=/build/target' \
      --change 'WORKDIR /src' \
      "$PACK" "$FINAL"
  else
    nerdctl commit "$PACK" "$FINAL"
  fi
  nerdctl rm -f "$PACK" >/dev/null 2>&1 || true
fi

echo "=== verify ==="
nerdctl images | grep vllm-router-verify || true
nerdctl run --rm --net=host "$FINAL" bash -lc 'pkg-config --version; if [ -d /build/target ]; then echo TARGET_OK; du -sh /build/target; else echo TARGET_MISSING; fi'
echo "disk after: $(df -h / | awk 'NR==2 {print $4}')"
cat > "$DONE" << EOF
image=$FINAL
test_result=$RESULT
finished=$(date -Is)
reuse=nerdctl run --rm --net=host -v /root/gyh_dev/vllm_router:/src -e CARGO_HOME=/build/home -e CARGO_TARGET_DIR=/build/target $FINAL bash -lc 'cd /src && cargo test --test pd_length_policy_sim -j2'
EOF
echo "=== pack done $(date -Is) ==="
cat "$DONE"
