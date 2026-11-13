#!/usr/bin/env python3
"""Upload/run remote commands via password SSH. Password from SSH_PASS only."""
from __future__ import annotations

import base64
import os
import pty
import select
import sys
import time


def _password() -> str:
    pw = os.environ.get("SSH_PASS")
    if not pw:
        raise SystemExit("SSH_PASS env required")
    return pw


def _ssh_run(host: str, remote_cmd: str, timeout: int = 300) -> str:
    user = os.environ.get("SSH_USER", "root")
    argv = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "NumberOfPasswordPrompts=1",
        f"{user}@{host}",
        remote_cmd,
    ]
    password = _password()
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("ssh", argv)
    deadline = time.time() + timeout
    buf = b""
    sent = False
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            sys.stderr.write("timeout\n")
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
            if (not sent) and b"password" in buf.lower():
                os.write(fd, (password + "\n").encode())
                sent = True
        else:
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
                if wpid != 0:
                    break
            except ChildProcessError:
                break
    try:
        os.waitpid(pid, 0)
    except Exception:
        pass
    out = buf.decode("utf-8", "replace")
    lines = []
    for line in out.splitlines():
        low = line.lower()
        if "password" in low or "warning: permanently" in low:
            continue
        if "post-quantum" in low or "store now, decrypt" in low:
            continue
        if "the server may need" in low or "see https://openssh.com/pq" in low:
            continue
        lines.append(line)
    return "\n".join(lines)


def cmd_run(host: str, command: str) -> None:
    print(_ssh_run(host, command))


def cmd_upload_run(host: str, local_path: str, remote_path: str) -> None:
    data = open(local_path, "rb").read()
    b64 = base64.b64encode(data).decode("ascii")
    # Chunk to keep argv reasonable.
    chunks = [b64[i : i + 4000] for i in range(0, len(b64), 4000)]
    parts = [f"open('{remote_path}','wb').write(__import__('base64').b64decode(''"]
    # Build remote python that concatenates chunks.
    py = [
        "import base64",
        f"chunks={chunks!r}",
        f"open({remote_path!r},'wb').write(base64.b64decode(''.join(chunks)))",
    ]
    remote = "python3 -c " + repr("; ".join(py)) + f" && chmod +x {remote_path} && bash {remote_path}"
    print(_ssh_run(host, remote, timeout=600))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage:\n"
            "  patent_ssh.py run HOST 'cmd'\n"
            "  patent_ssh.py upload-run HOST LOCAL_SCRIPT [REMOTE_PATH]"
        )
    mode = sys.argv[1]
    host = sys.argv[2]
    if mode == "run":
        cmd_run(host, sys.argv[3])
    elif mode == "upload-run":
        local = sys.argv[3]
        remote = sys.argv[4] if len(sys.argv) > 4 else "/tmp/patent_cmd.sh"
        cmd_upload_run(host, local, remote)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
