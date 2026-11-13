#!/usr/bin/env python3
"""Password SCP via PTY. Usage: _tmp_scp_upload.py LOCAL REMOTE_HOST REMOTE_PATH"""
from __future__ import annotations

import os
import pty
import select
import sys
import time


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: _tmp_scp_upload.py LOCAL HOST REMOTE_PATH")
    local, host, remote = sys.argv[1], sys.argv[2], sys.argv[3]
    password = os.environ.get("SSH_PASS")
    if not password:
        raise SystemExit("SSH_PASS required")
    user = os.environ.get("SSH_USER", "root")
    argv = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        local,
        f"{user}@{host}:{remote}",
    ]
    print("scp", local, "->", f"{user}@{host}:{remote}", flush=True)
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("scp", argv)
    deadline = time.time() + 600
    buf = b""
    sent = False
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            print("timeout", flush=True)
            break
        r, _, _ = select.select([fd], [], [], min(1.0, rem))
        if fd in r:
            try:
                data = os.read(fd, 8192)
            except OSError:
                break
            if not data:
                break
            buf += data
            if (not sent) and b"assword" in buf:
                os.write(fd, (password + "\n").encode())
                sent = True
                print("authed", flush=True)
            # show progress dots from scp
            if b"%" in data or b"ETA" in data:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
        else:
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    print("exit", status >> 8, flush=True)
                    sys.exit(status >> 8)
            except ChildProcessError:
                break
    try:
        wpid, status = os.waitpid(pid, 0)
        code = status >> 8
        print("exit", code, flush=True)
        sys.exit(code)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
