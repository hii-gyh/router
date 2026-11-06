#!/usr/bin/env python3
"""Real-PD trajectory load client through the live PD router.

Prompt token counts are measured with the model tokenizer and sent as
x-prompt-tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import secrets
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROUTER = os.environ.get("ROUTER_URL", "http://127.0.0.1:8090/v1/chat/completions")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8")
TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH", "/home/weights/Qwen3.8-27B-w8a8/"
)
PREFILL_PORTS = [18001, 18002, 18003, 18004]
DECODE_PORTS = [18011, 18012, 18013, 18014]
# Comma-separated override for short-pool prompt accounting (E1: 18001 only).
_SHORT_ENV = os.environ.get("SHORT_PORTS", "18001,18002").strip()
SHORT_PORTS = {int(x) for x in _SHORT_ENV.split(",") if x.strip()}
MAX_CTX = 16384
# Chat-template / special-token headroom so prompt+max_tokens stays inside
# the worker max_model_len after the server wraps the user message.
CTX_HEADROOM = int(os.environ.get("CTX_HEADROOM", "256"))
SAFE_OPEN_CTX = MAX_CTX - CTX_HEADROOM  # 2P@16384 + 4D@16384
SAFE_SHORT_CTX = 4096 - 64  # 2P@4096 short pool

# Locked success criteria (do not change after the run starts).
SUCCESS_CRITERIA = {
    "at": "R_sat",
    "require": [
        "experiment.output_tok_s > control.output_tok_s",
        "experiment.ttft_p90 <= control.ttft_p90",
    ],
    "note": (
        "Per-length-bucket regressions stay in the report. "
        "SYSTEM output tok/s + TTFT p90 are the headline metrics."
    ),
}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))]


def snap_metrics():
    out = {}
    for port in PREFILL_PORTS + DECODE_PORTS:
        row = {
            "prompt_tokens": 0.0,
            "generation_tokens": 0.0,
            "running": 0.0,
            "waiting": 0.0,
            "kv": 0.0,
            "external_kv_tokens": 0.0,
        }
        try:
            text = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=5
            ).read().decode()
        except Exception as exc:
            row["error"] = str(exc)[:120]
            out[str(port)] = row
            continue
        for line in text.splitlines():
            if line.startswith("vllm:prompt_tokens_total{"):
                row["prompt_tokens"] = float(line.split()[-1])
            elif line.startswith("vllm:generation_tokens_total{"):
                row["generation_tokens"] = float(line.split()[-1])
            elif line.startswith("vllm:num_requests_running{"):
                row["running"] = float(line.split()[-1])
            elif (
                line.startswith("vllm:num_requests_waiting{")
                and "reason" not in line
            ):
                row["waiting"] = float(line.split()[-1])
            elif line.startswith("vllm:kv_cache_usage_perc{"):
                row["kv"] = float(line.split()[-1])
            elif (
                'source="external_kv_transfer"' in line
                and line.startswith("vllm:prompt_tokens_by_source_total{")
            ):
                row["external_kv_tokens"] = float(line.split()[-1])
        out[str(port)] = row
    return out


def delta_metrics(a, b):
    d = {}
    for k, av in a.items():
        bv = b.get(k, {})
        d[k] = {
            m: float(bv.get(m, 0.0)) - float(av.get(m, 0.0))
            for m in (
                "prompt_tokens",
                "generation_tokens",
                "running",
                "waiting",
                "kv",
                "external_kv_tokens",
            )
        }
    return d


def wait_decode_idle(timeout_s=180, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        m = snap_metrics()
        running = sum(m[str(p)]["running"] for p in DECODE_PORTS)
        waiting = sum(m[str(p)]["waiting"] for p in DECODE_PORTS)
        if running <= 0.01 and waiting <= 0.01:
            return True, m
        time.sleep(poll)
    return False, snap_metrics()


class KvSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.samples = []
        self._thr = None

    def start(self):
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            kv = {}
            run = {}
            wait = {}
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
                    elif line.startswith("vllm:num_requests_running{"):
                        run[port] = float(line.split()[-1])
                    elif (
                        line.startswith("vllm:num_requests_waiting{")
                        and "reason" not in line
                    ):
                        wait[port] = float(line.split()[-1])
            if kv:
                self.samples.append(
                    {"t": time.time(), "kv": kv, "running": run, "waiting": wait}
                )
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples:
            return {
                "n_samples": 0,
                "kv_peak": {},
                "kv_peak_max": None,
                "kv_mean_of_max": None,
                "kv_p50_of_max": None,
                "kv_p90_of_max": None,
                "kv_frac_max_ge_0_25": None,
                "kv_frac_max_ge_0_40": None,
                "inflight_max_peak": None,
                "inflight_mean_of_max": None,
                "inflight_frac_max_ge_4": None,
                "kv_stddev_mean": None,
            }
        peaks = {p: 0.0 for p in DECODE_PORTS}
        stds = []
        max_kvs = []
        max_inflights = []
        for s in self.samples:
            vals = []
            runs = []
            waits = []
            for p in DECODE_PORTS:
                v = float(s["kv"].get(p, 0.0))
                peaks[p] = max(peaks[p], v)
                vals.append(v)
                runs.append(float(s.get("running", {}).get(p, 0.0)))
                waits.append(float(s.get("waiting", {}).get(p, 0.0)))
            if vals:
                max_kvs.append(max(vals))
            if runs or waits:
                max_inflights.append(max(r + w for r, w in zip(runs, waits)))
            if len(vals) >= 2:
                stds.append(statistics.pstdev(vals))

        def frac_ge(xs, thr):
            if not xs:
                return None
            return round(sum(1 for x in xs if x >= thr) / len(xs), 4)

        return {
            "n_samples": len(self.samples),
            "kv_peak": {str(k): round(v, 4) for k, v in peaks.items()},
            "kv_peak_max": round(max(peaks.values()), 4) if peaks else None,
            "kv_mean_of_max": round(statistics.mean(max_kvs), 4) if max_kvs else None,
            "kv_p50_of_max": round(pct(max_kvs, 0.50), 4) if max_kvs else None,
            "kv_p90_of_max": round(pct(max_kvs, 0.90), 4) if max_kvs else None,
            "kv_frac_max_ge_0_25": frac_ge(max_kvs, 0.25),
            "kv_frac_max_ge_0_40": frac_ge(max_kvs, 0.40),
            "inflight_max_peak": round(max(max_inflights), 2) if max_inflights else None,
            "inflight_mean_of_max": (
                round(statistics.mean(max_inflights), 2) if max_inflights else None
            ),
            "inflight_frac_max_ge_4": frac_ge(max_inflights, 4.0),
            "kv_stddev_mean": round(statistics.mean(stds), 4) if stds else None,
        }

def synthesize_trajectory(n=800, seed=20251031):
    """Fixed-seed conversation-like length histogram (ShareGPT-ish).

    Input ~100-4k (heavy tail), output ~50-1k. Not bimodal 256/8192 spikes
    and not 640/640/80/80 cycling.
    """
    rng = random.Random(seed)
    rows = []
    t = 0.0
    for i in range(n):
        # Mixture: short chat, medium, long paste-then-ask.
        u = rng.random()
        if u < 0.55:
            prompt_tok = int(rng.lognormvariate(math.log(220), 0.55))
        elif u < 0.88:
            prompt_tok = int(rng.lognormvariate(math.log(900), 0.45))
        else:
            prompt_tok = int(rng.lognormvariate(math.log(2500), 0.35))
        prompt_tok = max(64, min(4096, prompt_tok))

        v = rng.random()
        if v < 0.60:
            out_tok = int(rng.lognormvariate(math.log(120), 0.50))
        elif v < 0.90:
            out_tok = int(rng.lognormvariate(math.log(280), 0.40))
        else:
            out_tok = int(rng.lognormvariate(math.log(650), 0.35))
        out_tok = max(32, min(1024, out_tok))

        if prompt_tok + out_tok > MAX_CTX:
            out_tok = max(16, MAX_CTX - prompt_tok)
        if prompt_tok + out_tok > MAX_CTX:
            continue

        # Synthetic inter-arrival base (seconds at rate=1); scaled later.
        gap = rng.expovariate(1.0)
        t += gap
        rows.append(
            {
                "id": f"syn-{i}",
                "prompt_tokens_target": prompt_tok,
                "max_tokens": out_tok,
                "arrival_offset_s": t,
                "source": "synthetic_seed_20251031",
            }
        )
    return {
        "generator": "synthetic_sharegptish_lognormal_mixture",
        "seed": seed,
        "n_raw": n,
        "n_kept": len(rows),
        "requests": rows,
        "notes": (
            "Mimics conversation length histograms: prompt~100-4k, "
            "output~50-1k heavy tail. Dropped prompt+max_tokens>16384."
        ),
    }


def _clamp_prompt_out(prompt_tok: int, out_tok: int, budget: int, min_out: int = 16):
    """Ensure prompt+max_tokens fits a worker max_model_len budget."""
    if prompt_tok >= budget - min_out:
        prompt_tok = max(16, budget - min_out)
    if prompt_tok + out_tok > budget:
        out_tok = max(min_out, budget - prompt_tok)
    if prompt_tok + out_tok > budget:
        return None
    return prompt_tok, out_tok


def synthesize_high_kv_trajectory(n=800, seed=20251102):
    """Decode-KV pressure trajectory for full-pool decode_kv_cost best-case.

    Prior ShareGPT-ish traj peaked ~10–12% kv_usage on 4D@16384, so cost
    placement had little room to beat RR. This mix deliberately:
      - lands large prompts on D (open-pool length, 4k–10k)
      - holds KV with long uneven outputs (resident vs short probe)
      - front-loads a resident preload burst so probes hit a hot pool
      - targets mean-of-max kv_usage in ~0.40–0.70 under ~80–150k D capacity

    All rows satisfy prompt+max_tokens ≤ SAFE_OPEN_CTX (16384-CTX_HEADROOM).
    Pair with open-pool-only prefills (or length-aware prefill) so requests
    never land on short P@4096. Use --ignore-eos so residents hold max_tokens.
    """
    rng = random.Random(seed)
    rows = []
    t = 0.0
    budget = SAFE_OPEN_CTX
    # Front-load ~1 full D batch of long residents before the Poisson mix so
    # RR already has a hotspot when probes arrive (max_num_seqs=8 → 32 pool).
    preload_n = min(24, max(16, n // 30))
    for i in range(preload_n):
        prompt_tok = int(rng.lognormvariate(math.log(7200), 0.22))
        prompt_tok = max(4096, min(10240, prompt_tok))
        out_choices = (1536, 2048, 2048, 3072, 4096, 5120)
        out_tok = out_choices[i % len(out_choices)]
        clamped = _clamp_prompt_out(prompt_tok, out_tok, budget, min_out=256)
        if clamped is None:
            continue
        prompt_tok, out_tok = clamped
        # Tight stagger so they land almost together (~0.15s apart).
        t = i * 0.15
        rows.append(
            {
                "id": f"hkv-preload-{i}",
                "prompt_tokens_target": prompt_tok,
                "max_tokens": out_tok,
                "arrival_offset_s": t,
                "kind": "resident",
                "source": f"synthetic_high_kv_seed_{seed}",
            }
        )
    t = preload_n * 0.15 + 1.0
    for i in range(n):
        u = rng.random()
        if u < 0.50:
            # Long-context residents: dominate KV footprint + residence time.
            kind = "resident"
            prompt_tok = int(rng.lognormvariate(math.log(6400), 0.25))
            prompt_tok = max(4096, min(10240, prompt_tok))
            # Uneven hold lengths so RR hotspots one D while others free.
            out_choices = (1024, 1536, 2048, 3072, 4096, 5120)
            out_tok = out_choices[rng.randrange(len(out_choices))]
            min_out = 256
        elif u < 0.75:
            kind = "medium"
            prompt_tok = int(rng.lognormvariate(math.log(3200), 0.28))
            prompt_tok = max(2048, min(7168, prompt_tok))
            out_tok = int(rng.lognormvariate(math.log(1100), 0.32))
            out_tok = max(512, min(3072, out_tok))
            min_out = 128
        else:
            # Short probes: latency/throughput signal while residents hold KV.
            kind = "probe"
            prompt_tok = int(rng.lognormvariate(math.log(420), 0.40))
            prompt_tok = max(128, min(1024, prompt_tok))
            out_tok = int(rng.lognormvariate(math.log(64), 0.35))
            out_tok = max(32, min(160, out_tok))
            min_out = 16

        clamped = _clamp_prompt_out(prompt_tok, out_tok, budget, min_out=min_out)
        if clamped is None:
            continue
        prompt_tok, out_tok = clamped

        # Mild burstiness: probes cluster after residents so they hit a hot pool.
        if kind == "probe":
            gap = rng.expovariate(2.5)
        elif kind == "resident":
            gap = rng.expovariate(0.70)
        else:
            gap = rng.expovariate(1.1)
        t += gap
        rows.append(
            {
                "id": f"hkv-{i}",
                "prompt_tokens_target": prompt_tok,
                "max_tokens": out_tok,
                "arrival_offset_s": t,
                "kind": kind,
                "source": f"synthetic_high_kv_seed_{seed}",
            }
        )
    return {
        "generator": "synthetic_high_kv_decode_pressure_clean",
        "seed": seed,
        "n_raw": n + preload_n,
        "n_kept": len(rows),
        "preload_n": preload_n,
        "safe_open_ctx": budget,
        "ctx_headroom": CTX_HEADROOM,
        "requests": rows,
        "recommend_ignore_eos": True,
        "recommend_rates": [1.0, 1.6, 2.4],
        "target_kv_usage_band": [0.40, 0.70],
        "notes": (
            "Clean high-KV decode pressure: 16–24 preload residents + ~50% resident "
            f"(prompt~4–10k, out~1–5k), ~25% medium, ~25% short probes; "
            f"every row prompt+max_tokens≤{budget} (open max_model_len {MAX_CTX} "
            f"minus headroom {CTX_HEADROOM}). Requires open-pool-capable PD path. "
            "Use --ignore-eos."
        ),
    }

def synthesize_fixed_trajectory(n=200, seed=20251103, prompt_tokens=2048, max_tokens=256):
    """Fixed prompt/output length grid cell for load-scenario search."""
    rng = random.Random(seed)
    rows = []
    t = 0.0
    budget = SAFE_OPEN_CTX
    clamped = _clamp_prompt_out(int(prompt_tokens), int(max_tokens), budget, min_out=16)
    if clamped is None:
        raise ValueError(f"prompt+out exceeds SAFE_OPEN_CTX={budget}")
    prompt_tok, out_tok = clamped
    for i in range(n):
        t += rng.expovariate(1.0)
        kind = "short" if prompt_tok <= 2048 else "long"
        rows.append(
            {
                "id": f"fix-{prompt_tok}-{out_tok}-{i}",
                "prompt_tokens_target": prompt_tok,
                "max_tokens": out_tok,
                "arrival_offset_s": t,
                "kind": kind,
                "source": f"synthetic_fixed_p{prompt_tok}_o{out_tok}_seed_{seed}",
            }
        )
    return {
        "generator": "synthetic_fixed_length_grid",
        "seed": seed,
        "n_raw": n,
        "n_kept": len(rows),
        "prompt_tokens": prompt_tok,
        "max_tokens": out_tok,
        "safe_open_ctx": budget,
        "ctx_headroom": CTX_HEADROOM,
        "requests": rows,
        "recommend_ignore_eos": out_tok >= 512,
        "notes": f"Fixed length cell prompt={prompt_tok} out={out_tok}.",
    }


def synthesize_mixed_aprefill_trajectory(n=400, seed=20251103):
    """Weighted mix of fixed (p,o) cells for A_prefill vs A0 mixed-load search.

    Nominal cells: 4k/1k, 7k/2k, 16k/2k, 512/8k.
    16k/2k is clamped to SAFE_OPEN_CTX (max_model_len 16384 − CTX_HEADROOM):
      prompt=14080, out=2048  (14080+2048=16128).
    Slightly long-biased fractions: 0.20 / 0.25 / 0.30 / 0.25.
    """
    rng = random.Random(seed)
    budget = SAFE_OPEN_CTX
    # nominal → actual. For 16k/2k prefer KEEP out=2048 and shrink prompt
    # (generic _clamp_prompt_out would shrink out to 16 when prompt≈budget).
    specs = [
        {"name": "4k_1k", "prompt": 4096, "out": 1024, "frac": 0.20},
        {"name": "7k_2k", "prompt": 7168, "out": 2048, "frac": 0.25},
        {"name": "16k_2k", "prompt": 16384, "out": 2048, "frac": 0.30},
        {"name": "512_8k", "prompt": 512, "out": 8192, "frac": 0.25},
    ]
    cells = []
    for s in specs:
        p, o = int(s["prompt"]), int(s["out"])
        if p + o > budget:
            # Prefer preserving requested decode length for this mix cell.
            if o < budget - 16:
                p = budget - o
            else:
                clamped = _clamp_prompt_out(p, o, budget, min_out=16)
                if clamped is None:
                    raise ValueError(f"cell {s['name']} cannot fit SAFE_OPEN_CTX={budget}")
                p, o = clamped
        if p + o > budget:
            raise ValueError(f"cell {s['name']} still overflows after clamp: {p}+{o}>{budget}")
        cells.append(
            {
                **s,
                "prompt_actual": p,
                "out_actual": o,
                "clamped": (p != s["prompt"] or o != s["out"]),
            }
        )
    rows = []
    t = 0.0
    counts = {c["name"]: 0 for c in cells}
    for i in range(n):
        u = rng.random()
        acc = 0.0
        chosen = cells[-1]
        for c in cells:
            acc += c["frac"]
            if u <= acc:
                chosen = c
                break
        counts[chosen["name"]] += 1
        t += rng.expovariate(1.0)
        p, o = chosen["prompt_actual"], chosen["out_actual"]
        kind = "short" if p <= 2048 else "long"
        rows.append(
            {
                "id": f"mix-{chosen['name']}-{i}",
                "prompt_tokens_target": p,
                "max_tokens": o,
                "arrival_offset_s": t,
                "kind": kind,
                "cell": chosen["name"],
                "source": f"synthetic_mixed_aprefill_seed_{seed}",
            }
        )
    return {
        "generator": "synthetic_mixed_aprefill_weighted",
        "seed": seed,
        "n_raw": n,
        "n_kept": len(rows),
        "safe_open_ctx": budget,
        "ctx_headroom": CTX_HEADROOM,
        "max_model_len": MAX_CTX,
        "mix_fractions_nominal": {c["name"]: c["frac"] for c in cells},
        "mix_counts": counts,
        "cells": [
            {
                "name": c["name"],
                "nominal_prompt": c["prompt"],
                "nominal_out": c["out"],
                "actual_prompt": c["prompt_actual"],
                "actual_out": c["out_actual"],
                "frac": c["frac"],
                "clamped": c["clamped"],
            }
            for c in cells
        ],
        "requests": rows,
        "recommend_ignore_eos": True,
        "notes": (
            "Slightly long-biased mix of 4k/1k, 7k/2k, 16k/2k(clamped), 512/8k. "
            f"16k/2k → prompt={next(c['prompt_actual'] for c in cells if c['name']=='16k_2k')} "
            f"out={next(c['out_actual'] for c in cells if c['name']=='16k_2k')} "
            f"(SAFE_OPEN_CTX={budget}=max_model_len {MAX_CTX}-headroom {CTX_HEADROOM})."
        ),
    }


def try_load_sharegpt(path: Path, n=800, seed=20251031):
    if not path.exists():
        return None
    rng = random.Random(seed)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "requests" in raw:
        return raw
    # ShareGPT list of {conversations:[{from,value},...]}
    items = raw if isinstance(raw, list) else raw.get("data", [])
    rows = []
    t = 0.0
    # Lazy tokenizer only for length estimate if content present; else skip.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    for i, item in enumerate(items):
        if len(rows) >= n:
            break
        conv = item.get("conversations") or item.get("messages") or []
        human = None
        gpt = None
        for turn in conv:
            role = (turn.get("from") or turn.get("role") or "").lower()
            val = turn.get("value") or turn.get("content") or ""
            if human is None and role in ("human", "user"):
                human = val
            elif human is not None and gpt is None and role in ("gpt", "assistant"):
                gpt = val
                break
        if not human:
            continue
        # Unique nonce prefix so prefix-cache cannot collapse costs.
        nonce = secrets.token_hex(8)
        text = f"[{nonce}] {human}"
        ptok = len(tok.encode(text))
        otok = len(tok.encode(gpt)) if gpt else 128
        otok = max(32, min(1024, otok))
        if ptok + otok > MAX_CTX or ptok < 16:
            continue
        t += rng.expovariate(1.0)
        rows.append(
            {
                "id": f"sg-{i}",
                "prompt_text": text,
                "prompt_tokens_target": ptok,
                "max_tokens": otok,
                "arrival_offset_s": t,
                "source": str(path),
            }
        )
    if not rows:
        return None
    return {
        "generator": "sharegpt_json",
        "seed": seed,
        "path": str(path),
        "n_kept": len(rows),
        "requests": rows,
    }


def build_prompt(tok, target_tokens: int, seed_text: str | None = None):
    nonce = secrets.token_hex(12)
    base = seed_text or (
        "请根据下面的对话上下文继续回答用户问题，保持简洁但信息完整。"
        "这是一段模拟真实多轮对话的材料，用于评测系统吞吐与时延。"
    )
    text = f"[{nonce}] " + base
    # Grow without O(n^2) re-encode loop for large targets.
    ids = tok.encode(text)
    if len(ids) >= target_tokens:
        return tok.decode(ids[:target_tokens]), len(ids[:target_tokens])
    pad = tok.encode(base)
    while len(ids) < target_tokens:
        need = target_tokens - len(ids)
        ids.extend(pad[:need] if need < len(pad) else pad)
    ids = ids[:target_tokens]
    return tok.decode(ids), len(ids)


def materialize_requests(traj, tok, limit=None):
    """Build runnable requests; clamp to SAFE_OPEN_CTX (chat-template headroom)."""
    out = []
    budget = int(traj.get("safe_open_ctx") or SAFE_OPEN_CTX)
    for row in traj["requests"]:
        if limit is not None and len(out) >= limit:
            break
        if "prompt_text" in row:
            text = row["prompt_text"]
            # Re-measure with nonce already present.
            ntok = len(tok.encode(text))
            if ntok != row["prompt_tokens_target"]:
                # Keep measured count as truth for x-prompt-tokens.
                pass
        else:
            text, ntok = build_prompt(tok, int(row["prompt_tokens_target"]))
        mt = int(row["max_tokens"])
        clamped = _clamp_prompt_out(ntok, mt, budget, min_out=16)
        if clamped is None:
            continue
        ntok, mt = clamped
        # Rebuild prompt if clamp shrank the target token count.
        if ntok != len(tok.encode(text)):
            text, ntok = build_prompt(tok, ntok)
            clamped = _clamp_prompt_out(ntok, mt, budget, min_out=16)
            if clamped is None:
                continue
            ntok, mt = clamped
        bucket = "short" if ntok <= 2048 else "long"
        out.append(
            {
                "id": row["id"],
                "prompt": text,
                "prompt_tokens": ntok,
                "max_tokens": mt,
                "arrival_offset_s": float(row["arrival_offset_s"]),
                "bucket": bucket,
                "kind": row.get("kind"),
                "source": row.get("source", traj.get("generator")),
            }
        )
    return out


def scale_arrivals(reqs, rate):
    """Scale trajectory offsets so mean arrival rate ~= rate (req/s)."""
    if not reqs:
        return []
    # Normalize offsets to start at 0, then stretch/compress to target rate.
    offs = [r["arrival_offset_s"] for r in reqs]
    span = max(offs[-1] - offs[0], 1e-6)
    # Desired span for N-1 intervals at rate R is (N-1)/R
    target_span = (len(reqs) - 1) / max(rate, 1e-6)
    scale = target_span / span
    out = []
    base = offs[0]
    for r, o in zip(reqs, offs):
        nr = dict(r)
        nr["arrival_offset_s"] = (o - base) * scale
        out.append(nr)
    return out


def one_request(req, timeout_s=900, ignore_eos=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": req["prompt"]}],
        "max_tokens": req["max_tokens"],
        "temperature": 0,
        "stream": True,
        "skip_special_tokens": True,
    }
    # High-KV pressure arms need residents to actually hold max_tokens;
    # chat prompts otherwise escape in ~100–200 tokens and never fill KV.
    if ignore_eos or req.get("ignore_eos"):
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "x-prompt-tokens": str(req["prompt_tokens"]),
    }
    http_req = urllib.request.Request(ROUTER, data=body, headers=headers)
    t0 = time.perf_counter()
    ttft = None
    t_last = None
    out_tokens = 0
    status = 0
    err = ""
    try:
        with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
            status = resp.status
            while True:
                line = resp.readline()
                if not line:
                    break
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                t_last = now
                try:
                    obj = json.loads(payload)
                    choice = (obj.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        # Approximate token count; usage may be absent in stream.
                        out_tokens += max(1, len(content) // 2)
                    usage = obj.get("usage") or {}
                    if "completion_tokens" in usage:
                        out_tokens = int(usage["completion_tokens"])
                except Exception:
                    out_tokens += 1
    except Exception as exc:
        err = f"{type(exc).__name__}:{str(exc)[:160]}"
    e2e = time.perf_counter() - t0
    tpot = None
    if ttft is not None and t_last is not None and out_tokens > 1:
        gen_s = max(1e-6, t_last - (t0 + ttft))
        tpot = gen_s / (out_tokens - 1)
    return {
        "id": req["id"],
        "bucket": req["bucket"],
        "prompt_tokens": req["prompt_tokens"],
        "max_tokens": req["max_tokens"],
        "status": status,
        "ttft": ttft,
        "e2e": e2e,
        "tpot": tpot,
        "out_tokens": out_tokens,
        "error": err,
        "ok": ttft is not None and status == 200 and not err,
    }


def run_load(reqs, rate, max_workers=64, label="", ignore_eos=False):
    timed = scale_arrivals(reqs, rate)
    sampler = KvSampler(0.5)
    before = snap_metrics()
    sampler.start()
    t_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = []
        for r in timed:
            # Pace submissions by arrival offset.
            target = t_start + r["arrival_offset_s"]
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futs.append(pool.submit(one_request, r, 900, ignore_eos))
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t_start
    sampler.stop()
    after = snap_metrics()
    deltas = delta_metrics(before, after)
    ok = [r for r in results if r["ok"]]
    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    out_tok = sum(r["out_tokens"] for r in ok)
    # Prefer worker generation_tokens delta if available (more accurate).
    gen_delta = sum(
        max(0.0, deltas[str(p)]["generation_tokens"]) for p in DECODE_PORTS
    )
    if gen_delta > 0:
        out_tok_s = gen_delta / wall
        out_tok_total = gen_delta
    else:
        out_tok_s = out_tok / wall if wall > 0 else 0.0
        out_tok_total = out_tok

    short_prompt = sum(
        max(0.0, deltas[str(p)]["prompt_tokens"]) for p in SHORT_PORTS
    )
    open_prompt = sum(
        max(0.0, deltas[str(p)]["prompt_tokens"])
        for p in PREFILL_PORTS
        if p not in SHORT_PORTS
    )

    def bucket_stats(name):
        group = [r for r in ok if r["bucket"] == name]
        ts = [r["ttft"] for r in group]
        return {
            "n": len(group),
            "ttft_p50": pct(ts, 0.50),
            "ttft_p90": pct(ts, 0.90),
            "out_tokens": sum(r["out_tokens"] for r in group),
        }

    return {
        "label": label,
        "rate_rps": rate,
        "n_submitted": len(timed),
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "error_rate": (len(results) - len(ok)) / max(1, len(results)),
        "wall_s": round(wall, 3),
        "output_tok_total": out_tok_total,
        "output_tok_s": round(out_tok_s, 3),
        "ttft_p50": pct(ttfts, 0.50),
        "ttft_p90": pct(ttfts, 0.90),
        "tpot_p50": pct(tpots, 0.50),
        "tpot_p90": pct(tpots, 0.90),
        "buckets": {
            "short_le_2048": bucket_stats("short"),
            "long_gt_2048": bucket_stats("long"),
        },
        "prefill_prompt_delta": {
            "short_pool": short_prompt,
            "open_pool": open_prompt,
            "per_worker": {
                str(p): deltas[str(p)]["prompt_tokens"] for p in PREFILL_PORTS
            },
        },
        "decode_kv": sampler.summary(),
        "metric_delta": deltas,
        "errors_sample": [r for r in results if not r["ok"]][:8],
    }


def warmup(tok, n=20):
    """Send mixed short/long requests to warm Mooncake sessions."""
    reqs = []
    for i in range(n):
        target = 256 if i % 2 == 0 else 2048
        text, ntok = build_prompt(tok, target)
        reqs.append(
            {
                "id": f"warm-{i}",
                "prompt": text,
                "prompt_tokens": ntok,
                "max_tokens": 32,
                "arrival_offset_s": i * 0.05,
                "bucket": "short" if ntok <= 2048 else "long",
            }
        )
    # Fire with mild concurrency, ignore detailed stats.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one_request, reqs))
    wait_decode_idle(timeout_s=120)
    return n


def sample_kv_transfer_latencies(n=20):
    """Best-effort scrape of recent KV transfer lines from decode containers."""
    import subprocess
    import re

    pats = [
        re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*ms", re.I),
        re.compile(r"took\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", re.I),
        re.compile(r"latency[=: ]+([0-9]+(?:\.[0-9]+)?)", re.I),
    ]
    samples = []
    for i in range(4):
        name = f"pd-real-decode-{i}"
        try:
            text = subprocess.check_output(
                ["nerdctl", "logs", "--tail", "400", name],
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        except FileNotFoundError:
            # Client often runs inside the ascend image without nerdctl; skip quietly.
            return []
        except Exception as exc:
            samples.append({"worker": name, "error": str(exc)[:120]})
            continue
        lines = [
            ln
            for ln in text.splitlines()
            if "KV" in ln or "kv" in ln or "transfer" in ln.lower() or "mooncake" in ln.lower()
        ]
        for ln in lines[-30:]:
            for pat in pats:
                m = pat.search(ln)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower() if m.lastindex and m.lastindex >= 2 else "ms"
                    if unit == "s":
                        val *= 1000.0
                    samples.append(
                        {"worker": name, "ms": val, "line": ln[:220]}
                    )
                    break
    return samples[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=[
            "prepare",
            "prepare-high-kv",
            "prepare-fixed",
            "prepare-mixed-aprefill",
            "warmup",
            "run",
            "idle",
        ],
        required=True,
    )
    ap.add_argument("--traj", default="/root/gyh_dev/vllm_router/data/trajectory.json")
    ap.add_argument("--sharegpt", default="/root/gyh_dev/vllm_router/data/sharegpt.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--warmup-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20251031)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force max_tokens residence (required for high-KV pressure arms).",
    )
    args = ap.parse_args()

    if args.mode == "idle":
        ok, m = wait_decode_idle()
        print(json.dumps({"idle": ok, "metrics": m}, ensure_ascii=False))
        return

    if args.mode in (
        "prepare",
        "prepare-high-kv",
        "prepare-fixed",
        "prepare-mixed-aprefill",
    ):
        Path(args.traj).parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "prepare-high-kv":
            traj = synthesize_high_kv_trajectory(n=1200, seed=args.seed)
        elif args.mode == "prepare-fixed":
            traj = synthesize_fixed_trajectory(
                n=max(args.n * 4, 200),
                seed=args.seed,
                prompt_tokens=args.prompt_tokens,
                max_tokens=args.max_tokens,
            )
        elif args.mode == "prepare-mixed-aprefill":
            traj = synthesize_mixed_aprefill_trajectory(
                n=max(args.n * 4, 400), seed=args.seed
            )
        else:
            traj = try_load_sharegpt(Path(args.sharegpt), n=1200, seed=args.seed)
            if traj is None:
                traj = synthesize_trajectory(n=1200, seed=args.seed)
        Path(args.traj).write_text(
            json.dumps(traj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Lightweight shape summary (no tokenizer needed).
        prompts = [int(r["prompt_tokens_target"]) for r in traj["requests"]]
        outs = [int(r["max_tokens"]) for r in traj["requests"]]
        kinds = {}
        for r in traj["requests"]:
            k = r.get("kind") or "default"
            kinds[k] = kinds.get(k, 0) + 1
        print(
            json.dumps(
                {
                    "wrote": args.traj,
                    "generator": traj["generator"],
                    "n_kept": traj["n_kept"],
                    "kinds": kinds,
                    "prompt_p50": pct(prompts, 0.50),
                    "prompt_p90": pct(prompts, 0.90),
                    "out_p50": pct(outs, 0.50),
                    "out_p90": pct(outs, 0.90),
                    "recommend_ignore_eos": traj.get("recommend_ignore_eos", False),
                    "recommend_rates": traj.get("recommend_rates"),
                    "target_kv_usage_band": traj.get("target_kv_usage_band"),
                    "success_criteria": SUCCESS_CRITERIA,
                },
                ensure_ascii=False,
            )
        )
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    if args.mode == "warmup":
        n = warmup(tok, n=args.warmup_n)
        print(json.dumps({"warmup": n}, ensure_ascii=False))
        return

    # mode == run
    traj = json.loads(Path(args.traj).read_text(encoding="utf-8"))
    # Materialize a pool larger than N, then take a contiguous slice so both
    # arms see the same request order.
    pool = materialize_requests(traj, tok, limit=args.offset + args.n + 5)
    reqs = pool[args.offset : args.offset + args.n]
    if len(reqs) < args.n:
        raise SystemExit(f"not enough requests: have {len(reqs)} need {args.n}")

    result = run_load(
        reqs, rate=args.rate, label=args.label, ignore_eos=args.ignore_eos
    )
    result["success_criteria"] = SUCCESS_CRITERIA
    result["ignore_eos"] = bool(args.ignore_eos)
    result["kv_transfer_samples"] = sample_kv_transfer_latencies(24)
    result["traj_meta"] = {
        "generator": traj.get("generator"),
        "seed": traj.get("seed"),
        "offset": args.offset,
        "n": args.n,
        "ignore_eos": bool(args.ignore_eos),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
