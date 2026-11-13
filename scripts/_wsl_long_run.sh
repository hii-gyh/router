#!/bin/bash
# Usage: _wsl_long_run.sh HOST REMOTE_CMD [TIMEOUT]
set -euo pipefail
: "${SSH_PASS:?}"
HOST="$1"
CMD="$2"
TIMEOUT="${3:-1200}"
python3 - "$HOST" "$CMD" "$TIMEOUT" <<'PY'
import os, sys
sys.path.insert(0, "/mnt/c/Users/Lenovo/codes/exps/router/scripts")
import patent_ssh
host, cmd, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(patent_ssh._ssh_run(host, cmd, timeout=timeout))
PY
