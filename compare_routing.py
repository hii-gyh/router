#!/usr/bin/env python3
"""Compare router TTFT under concurrent short/long waves.

Each request gets a unique nonce so automatic prefix cache cannot hide
prefill cost across waves or policies.
"""
import json, secrets, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "qwen3.8"


def one(prompt, ntok, kind):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "skip_special_tokens": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "x-prompt-tokens": str(ntok),
    })
    t0 = time.perf_counter()
    ttft = None
    status = 0
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            status = resp.status
            while True:
                line = resp.readline()
                if not line:
                    break
                if ttft is None and line.startswith(b"data:") and b"[DONE]" not in line:
                    ttft = time.perf_counter() - t0
    except Exception as exc:
        err = str(exc)[:180]
    return kind, ntok, status, ttft, err


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
    return xs[i]


def wave(make_short, make_long, n=8):
    out = []
    with ThreadPoolExecutor(max_workers=n * 2) as pool:
        futures = [pool.submit(one, make_long(), 8192, "long") for _ in range(n)]
        time.sleep(0.4)
        futures += [pool.submit(one, make_short(), 256, "short") for _ in range(n)]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/weights/Qwen3.8-27B-w8a8/", trust_remote_code=True)

    def build(seed, n):
        # Unique head defeats automatic prefix caching between requests.
        nonce = secrets.token_hex(16)
        text = f"[{nonce}]" + seed
        while len(tok.encode(text)) < n:
            text += seed
        ids = tok.encode(text)[:n]
        return tok.decode(ids)

    make_short = lambda: build("请用一句话回答。", 256)
    make_long = lambda: build("请阅读以下材料并在最后用一句话总结。", 8192)
    rows = []
    for i in range(4):
        print(f"wave {i} start", flush=True)
        part = wave(make_short, make_long)
        rows.extend(part)
        ok = sum(1 for r in part if r[3] is not None)
        print(f"wave {i} done ok={ok}/{len(part)}", flush=True)
        for r in part:
            print(f"  {r[0]} status={r[2]} ttft={'-' if r[3] is None else f'{r[3]:.3f}'} {r[4]}", flush=True)
    print("kind n ok ttft_p50 ttft_p90")
    for kind in ("short", "long"):
        group = [r for r in rows if r[0] == kind]
        ts = [r[3] for r in group if r[3] is not None]
        p50 = pct(ts, 0.50)
        p90 = pct(ts, 0.90)
        def fmt(v):
            return "-" if v is None else f"{v:.3f}"
        print(f"{kind} {len(group)} {len(ts)} {fmt(p50)} {fmt(p90)}")
    fails = [r for r in rows if r[3] is None]
    for r in fails[:8]:
        print("FAIL", r[0], r[2], r[4])


if __name__ == "__main__":
    main()
