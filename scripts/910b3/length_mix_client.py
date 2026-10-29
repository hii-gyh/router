#!/usr/bin/env python3
"""Send a fixed short/long mix to a PD router and record client TTFT.

No NPU is required on the client. Token counts should come from the same
tokenizer the servers use. Without --tokenizer, the script sends the
x-prompt-tokens value you pass and repeats a padding sentence to approximate
that length. That approximation is only for transport checks. For the
910B3 measurement run, pass --tokenizer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


SHORT_SENTENCE = "请用一句话回答：今天天气怎么样？"
LONG_PAD = "请阅读以下重复材料并在最后用一句话总结。"


@dataclass
class Sample:
    kind: str
    prompt_tokens: int
    status: int
    ttft_s: float | None
    total_s: float
    error: str


def build_prompt(tokenizer, target_tokens: int, kind: str) -> str:
    if tokenizer is None:
        unit = SHORT_SENTENCE if kind == "short" else LONG_PAD
        # About four characters per token is only a stand-in. The header
        # still carries the intended bucket length.
        repeats = max(1, target_tokens // 8)
        return (unit + "\n") * repeats
    text = SHORT_SENTENCE if kind == "short" else LONG_PAD
    while len(tokenizer.encode(text)) < target_tokens:
        text += LONG_PAD
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def send_one(url: str, model: str, prompt: str, prompt_tokens: int, max_tokens: int,
             session_id: str | None, timeout: float) -> Sample:
    kind = "short" if prompt_tokens <= 2048 else "long"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    raw = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-prompt-tokens": str(prompt_tokens),
    }
    if session_id:
        headers["x-session-id"] = session_id
    request = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    started = time.perf_counter()
    ttft = None
    status = 0
    error = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            while True:
                line = response.readline()
                if not line:
                    break
                if ttft is None and line.startswith(b"data:"):
                    ttft = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")[:300]
    except Exception as exc:  # noqa: BLE001 - report transport failures in the summary
        error = str(exc)
    return Sample(kind, prompt_tokens, status, ttft, time.perf_counter() - started, error)


def summarize(samples: list[Sample]) -> None:
    print(f"{'kind':<8} {'n':>4} {'ok':>4} {'ttft_p50_s':>12} {'ttft_p90_s':>12}")
    for kind in ("short", "long"):
        group = [sample for sample in samples if sample.kind == kind]
        ok = [sample.ttft_s for sample in group if sample.status == 200 and sample.ttft_s is not None]
        if not ok:
            print(f"{kind:<8} {len(group):>4} {0:>4} {'-':>12} {'-':>12}")
            continue
        ok.sort()
        p50 = ok[len(ok) // 2]
        p90 = ok[min(len(ok) - 1, int(len(ok) * 0.9))]
        print(f"{kind:<8} {len(group):>4} {len(ok):>4} {p50:12.3f} {p90:12.3f}")
    failed = [sample for sample in samples if sample.status != 200 or sample.ttft_s is None]
    for sample in failed[:10]:
        print(f"FAIL {sample.kind} tokens={sample.prompt_tokens} status={sample.status} {sample.error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--short-tokens", type=int, default=256)
    parser.add_argument("--long-tokens", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=30, help="short,long pairs after warmup")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--session-prefix", default="")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    plan = [("short", args.short_tokens), ("long", args.long_tokens)]
    samples: list[Sample] = []
    total = args.warmup + args.rounds
    for index in range(total):
        for kind, tokens in plan:
            prompt = build_prompt(tokenizer, tokens, kind)
            if tokenizer is not None:
                tokens = len(tokenizer.encode(prompt))
            session = f"{args.session_prefix}-{kind}" if args.session_prefix else None
            sample = send_one(args.url, args.model, prompt, tokens, args.max_tokens, session, args.timeout)
            if index >= args.warmup:
                samples.append(sample)
            print(
                f"{'warm' if index < args.warmup else 'run '} {sample.kind:<5} "
                f"tokens={sample.prompt_tokens:<5} status={sample.status:<3} "
                f"ttft={sample.ttft_s if sample.ttft_s is not None else float('nan'):.3f}s"
            )
    summarize(samples)
    if samples:
        ok = [sample.ttft_s for sample in samples if sample.ttft_s is not None]
        if len(ok) >= 2:
            print(f"all_ok_mean_s {statistics.fmean(ok):.3f}")


if __name__ == "__main__":
    main()
