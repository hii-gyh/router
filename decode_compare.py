#!/usr/bin/env python3
"""Compare decode-worker selection under real KV pressure.

Phase 1 fills the decode pool with long-running generations of uneven
output length, so some decode workers stay resident far longer than
others. Phase 2 trickles in short requests; their latency shows whether
the policy steers away from loaded workers.

Router runs PD mode over plain vLLM workers (no KV transfer), so decode
re-does prefill. Both arms pay that identically; only decode selection
differs.
"""
import json, secrets, statistics, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROUTER = "http://127.0.0.1:8090/v1/chat/completions"
MODEL = "qwen3.8"
DECODE_PORTS = [18005, 18006, 18007, 18008]

_stop = threading.Event()
_samples = []


def sampler():
    while not _stop.is_set():
        kv, wait, run = {}, {}, {}
        for port in DECODE_PORTS:
            try:
                text = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/metrics", timeout=3
                ).read().decode()
            except Exception:
                continue
            for line in text.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    kv[port] = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_waiting{") and "reason" not in line:
                    wait[port] = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_running{"):
                    run[port] = float(line.split()[-1])
        if kv:
            _samples.append((time.time(), kv, wait, run))
        _stop.wait(0.5)


def one(prompt, ntok, kind, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        # Without this the summarization prompt hits EOS in ~150 tokens and
        # never builds the resident KV load this test is measuring.
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(ROUTER, data=body, headers={
        "Content-Type": "application/json",
        "x-prompt-tokens": str(ntok),
    })
    t0 = time.perf_counter()
    ttft = None
    status = 0
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            status = resp.status
            while True:
                line = resp.readline()
                if not line:
                    break
                if ttft is None and line.startswith(b"data:") and b"[DONE]" not in line:
                    ttft = time.perf_counter() - t0
    except Exception as exc:
        err = f"{type(exc).__name__}:{str(exc)[:120]}"
    return kind, status, ttft, time.perf_counter() - t0, err


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/weights/Qwen3.8-27B-w8a8/", trust_remote_code=True)

    seed = "请阅读以下材料并在最后用一句话详细总结其要点。"
    seed_len = max(1, len(tok.encode(seed)))

    def build_base(n):
        # One tokenize pass. Growing the text inside a re-encoding loop costs
        # O(n^2) and would serialize request submission in the main thread.
        text = seed * (n // seed_len + 2)
        return tok.decode(tok.encode(text)[:n])

    base = {n: build_base(n) for n in (12288, 512)}

    def build(n):
        return f"[{secrets.token_hex(16)}]" + base[n]

    threading.Thread(target=sampler, daemon=True).start()
    rows = []
    lock = threading.Lock()

    def record(fut_result):
        with lock:
            rows.append(fut_result)

    pool = ThreadPoolExecutor(max_workers=72)
    futures = []

    # Phase 1: 40 resident generations (10 per decode worker, above the
    # max-num-seqs 8 batch limit) with uneven output length, so workers that
    # draw the long jobs stay loaded far longer than the others.
    heavy_outputs = [640, 640, 80, 80]
    heavy_prompts = [build(12288) for _ in range(40)]
    light_prompts = [build(512) for _ in range(24)]
    t_start = time.time()
    for i in range(40):
        mt = heavy_outputs[i % len(heavy_outputs)]
        futures.append(pool.submit(one, heavy_prompts[i], 12288, f"heavy{mt}", mt))
        time.sleep(0.05)

    # Phase 2: short requests arriving while phase 1 is still generating.
    time.sleep(25)
    print(f"phase2 start t={time.time() - t_start:.1f}s", flush=True)
    for i in range(24):
        futures.append(pool.submit(one, light_prompts[i], 512, "light", 32))
        time.sleep(0.8)

    for fut in futures:
        rows.append(fut.result())
    pool.shutdown(wait=True)
    _stop.set()
    time.sleep(0.8)

    busy = [s for s in _samples if max(s[1].values(), default=0) > 0.02]
    spreads = [max(s[1].values()) - min(s[1].values()) for s in busy]
    stddevs = [statistics.pstdev(list(s[1].values())) for s in busy if len(s[1]) > 1]
    peaks = {p: max((s[1].get(p, 0) for s in _samples), default=0) for p in DECODE_PORTS}
    max_wait = max((max(s[2].values(), default=0) for s in _samples), default=0)
    total_wait = sum(sum(s[2].values()) for s in _samples)
    max_run = max((max(s[3].values(), default=0) for s in _samples), default=0)

    out = {
        "n_requests": len(rows),
        "ok": sum(1 for r in rows if r[2] is not None),
        "kv_samples_busy": len(busy),
        "kv_spread_mean": round(statistics.fmean(spreads), 4) if spreads else None,
        "kv_spread_p90": round(pct(spreads, 0.9), 4) if spreads else None,
        "kv_stddev_mean": round(statistics.fmean(stddevs), 4) if stddevs else None,
        "kv_peak_per_worker": {str(p): round(v, 4) for p, v in peaks.items()},
        "kv_peak_max": round(max(peaks.values()), 4) if peaks else None,
        "max_waiting_any_worker": max_wait,
        "waiting_sample_sum": total_wait,
        "max_running_any_worker": max_run,
        "ttft": {},
        "e2e": {},
    }
    kinds = sorted({r[0] for r in rows})
    for kind in kinds:
        g = [r for r in rows if r[0] == kind]
        ts = [r[2] for r in g if r[2] is not None]
        es = [r[3] for r in g if r[2] is not None]
        out["ttft"][kind] = {
            "n": len(g), "ok": len(ts),
            "p50": round(pct(ts, 0.5), 3) if ts else None,
            "p90": round(pct(ts, 0.9), 3) if ts else None,
            "max": round(max(ts), 3) if ts else None,
        }
        out["e2e"][kind] = {
            "p50": round(pct(es, 0.5), 3) if es else None,
            "p90": round(pct(es, 0.9), 3) if es else None,
        }
    fails = [r for r in rows if r[2] is None]
    out["failures"] = [{"kind": r[0], "status": r[1], "err": r[4]} for r in fails[:5]]
    print("RESULT_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
