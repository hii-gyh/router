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


def _upload_bytes(host: str, data: bytes, remote_path: str, timeout: int = 600) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    chunk_size = 12000
    chunks = [b64[i : i + chunk_size] for i in range(0, len(b64), chunk_size)]
    remote_b64 = remote_path + ".b64"
    logs = [_ssh_run(host, f"rm -f {remote_b64} {remote_path}; touch {remote_b64}", timeout=60)]
    for idx, chunk in enumerate(chunks):
        # Append one chunk per SSH call to avoid ARG_MAX.
        cmd = f"printf '%s' '{chunk}' >> {remote_b64}"
        logs.append(_ssh_run(host, cmd, timeout=120))
        if idx % 20 == 0 or idx == len(chunks) - 1:
            logs.append(f"progress {idx + 1}/{len(chunks)}")
    decode = (
        f"python3 -c \"import base64; open({remote_path!r},'wb').write("
        f"base64.b64decode(open({remote_b64!r},'rb').read())); "
        f"import os; print('wrote', {remote_path!r}, os.path.getsize({remote_path!r}))\""
    )
    logs.append(_ssh_run(host, decode, timeout=timeout))
    return "\n".join(x for x in logs if x)


def cmd_upload_run(host: str, local_path: str, remote_path: str) -> None:
    data = open(local_path, "rb").read()
    print(_upload_bytes(host, data, remote_path))
    print(_ssh_run(host, f"chmod +x {remote_path} && bash {remote_path}", timeout=600))


def cmd_upload(host: str, local_path: str, remote_path: str) -> None:
    data = open(local_path, "rb").read()
    print(_upload_bytes(host, data, remote_path, timeout=900))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage:\n"
            "  patent_ssh.py run HOST 'cmd'\n"
            "  patent_ssh.py upload-run HOST LOCAL_SCRIPT [REMOTE_PATH]\n"
            "  patent_ssh.py upload HOST LOCAL_FILE REMOTE_PATH"
        )
    mode = sys.argv[1]
    host = sys.argv[2]
    if mode == "run":
        cmd_run(host, sys.argv[3])
    elif mode == "upload-run":
        local = sys.argv[3]
        remote = sys.argv[4] if len(sys.argv) > 4 else "/tmp/patent_cmd.sh"
        cmd_upload_run(host, local, remote)
    elif mode == "upload":
        cmd_upload(host, sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
