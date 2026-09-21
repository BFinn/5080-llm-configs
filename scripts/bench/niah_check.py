#!/usr/bin/env python3
"""niah_check.py - needle-in-a-haystack at real depth for a llama-server.

Gates the one claim perplexity can't: does q4_0 KV hold long-range retrieval
at 112K depth? Plants N unique needles at uniform depths in a token-budgeted
filler haystack, queries each, scores exact-phrase recall per depth band.
Standalone bench tool (stdlib only, prints by design - not platform code).
"""
import argparse, json, os, random, re, string, sys, time, urllib.request

FILLER_SEEDS = [
    "The quarterly reconciliation of the maritime logistics ledger requires cross-referencing bill-of-lading identifiers against customs declarations filed before the vessel departed.",
    "Maintenance protocols for the turbine assembly specify torque sequences, thermal tolerances, and an inspection interval tied to operating hours rather than calendar dates.",
    "Archival records indicate that the provincial survey office renumbered parcels twice during the decade, complicating any retrospective join between tax rolls and deeds.",
    "The firmware changelog describes a race condition in the interrupt handler that manifested only under sustained network load with fragmented packets.",
    "Field notes from the geological survey describe stratified sediment deposits, fault offsets measured in centimeters, and an anomalous magnetic signature near borehole seven.",
]

def http(url, payload=None, timeout=900, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def count_tokens(base, text, api_key=None):
    return len(http(base + "/tokenize", {"content": text, "add_special": False},
                    api_key=api_key)["tokens"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BENCH_BASE", "http://127.0.0.1:8082"))
    ap.add_argument("--ctx", type=int, default=112000)
    ap.add_argument("--needles", type=int, default=20)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--api-key", default=os.environ.get("NIAH_API_KEY", ""))
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="generous: the model may emit <think> reasoning before the answer")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    key = args.api_key or None

    health = http(args.base + "/health", api_key=key)
    if health.get("status") != "ok":
        sys.exit(f"server not healthy: {health}")

    needle_words = [''.join(rng.choices(string.ascii_lowercase, k=5)) + "-" +
                    ''.join(rng.choices(string.digits, k=4)) for _ in range(args.needles)]
    needles = []
    for i, code in enumerate(needle_words):
        pos = int((i + 0.5) / args.needles * args.ctx)
        text = (f"[NEEDLE {i}] The emergency access passphrase for vault "
                f"{i:02d} is: {code}. Repeat it exactly when asked.")
        needles.append((i, pos, code, text))

    block = " ".join(FILLER_SEEDS)
    block_tok = count_tokens(args.base, block, key)
    needles_tok = sum(count_tokens(args.base, t, key) for _, _, _, t in needles)
    per_needle_pad = (args.ctx - needles_tok - 1500) // args.needles
    n_blocks = max(1, per_needle_pad // block_tok)
    parts = []
    for i, pos, code, text in needles:
        parts.append((block + " ") * n_blocks)
        parts.append(text + "\n")
    haystack = "".join(parts)
    target = args.ctx - 1500
    while count_tokens(args.base, haystack, key) > target:
        haystack = haystack[:int(len(haystack) * 0.98)]
    lost = [i for i, _, code, _ in needles if code not in haystack]
    if lost:
        sys.exit(f"ABORT: trim removed needles {lost}; lower --ctx or --needles")
    total = count_tokens(args.base, haystack, key)
    print(f"haystack: {total} tokens, {args.needles} needles, budget={args.ctx}", flush=True)

    results = []
    for i, pos, code, _ in needles:
        q = haystack + (f"\n\nQuestion: What is the emergency access passphrase for vault "
                        f"{i:02d} from [NEEDLE {i}]? Answer with only the passphrase, nothing else.")
        t0 = time.time()
        out = http(args.base + "/v1/completions", {
            "model": args.model, "prompt": q, "max_tokens": args.max_tokens, "temperature": 0,
            "cache_prompt": True}, api_key=key)
        text = out["choices"][0]["text"].strip()
        hit = code in text
        # misattribution: a different needle's code in the answer, ours absent
        others = [c for j, (_, _, c, _) in enumerate(needles) if j != i and c in text]
        band = int(pos / args.ctx * 10) * 10
        results.append((i, band, hit, text[:60]))
        found = re.findall(r"[a-z]{5}-\d{4}", text)
        tag = "HIT " if hit else ("WRONG-NEEDLE" if others else "MISS")
        print(f"  needle {i:02d} @ {pos//1000:4d}K (band {band:2d}-{band+10:3d}%): "
              f"{tag} [{time.time()-t0:5.1f}s] codes_in_answer={found[:3]}", flush=True)

    hits = sum(1 for _, _, h, _ in results if h)
    print(f"\nrecall: {hits}/{args.needles} = {hits/args.needles:.0%}", flush=True)
    for lo in range(0, 100, 25):
        band = [h for _, b, h, _ in results if lo <= b < lo + 25]
        if band:
            print(f"  depth {lo:2d}-{lo+25:3d}%: {sum(band)}/{len(band)}")

if __name__ == "__main__":
    main()
