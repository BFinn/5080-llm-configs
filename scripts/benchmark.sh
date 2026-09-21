#!/usr/bin/env bash
# benchmark.sh — produce the numbers every entry in this repo quotes, the same way.
#
# Every model entry should be measured with this script so the tables are
# comparable. Quote the commit you ran it at.
#
#   ./scripts/benchmark.sh --label flashnext \
#       --model ~/ai-models/flashnext-gsq-q2_0/model-00001-of-00002.gguf \
#       --llama-dir ~/src/llama.cpp --ctx 131072 \
#       -- --no-warmup -lm dio -ngl 99 -ot 'ffn_.*_exps=CPU' -b 4096 -ub 2048
#
# Everything after `--` is passed to llama-server verbatim, so the arms you
# compare differ only in the flags you name there. It starts its own server on
# its own port; it never measures against your deployed one.
set -uo pipefail

LABEL=""; MODEL=""; LLAMA_DIR=""; CTX=32768; PORT=8299; OUT=""
STOP_UNIT=""; PHASES="s"; PPL_CORPUS=""; DEPTH_TOKENS=30000

die(){ echo "error: $*" >&2; exit 1; }
usage(){ sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --llama-dir) LLAMA_DIR="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --stop-unit) STOP_UNIT="$2"; shift 2;;      # systemd --user unit to stop for the run
    --phases) PHASES="$2"; shift 2;;            # s=speed q=perplexity p=structured n=niah
    --ppl-corpus) PPL_CORPUS="$2"; shift 2;;
    --depth-tokens) DEPTH_TOKENS="$2"; shift 2;;
    -h|--help) usage 0;;
    --) shift; break;;
    *) die "unknown option $1 (use -- before llama-server flags)";;
  esac
done
SRV_ARGS=( "$@" )

[ -n "$LABEL" ] || die "--label is required"
[ -n "$MODEL" ] || die "--model is required"
[ -f "$MODEL" ] || die "model not found: $MODEL"
[ -n "$LLAMA_DIR" ] || die "--llama-dir is required"
BIN="$LLAMA_DIR/build/bin"
[ -x "$BIN/llama-server" ] || die "llama-server not found in $BIN"
OUT="${OUT:-bench-$LABEL-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

SRV=""
cleanup(){
  [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null
  # always hand the machine back the way we found it
  [ -n "$STOP_UNIT" ] && systemctl --user start "$STOP_UNIT" 2>/dev/null
  return 0
}
trap cleanup EXIT INT TERM

vram_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0; }
wait_vram_free(){ for _ in $(seq 1 60); do [ "$(vram_used)" -lt 600 ] && return 0; sleep 3; done
                  echo "  warning: GPU still busy, continuing anyway"; }

srv(){ # logfile extra-args...
  local log="$1"; shift
  wait_vram_free
  "$BIN/llama-server" -m "$MODEL" -c "$CTX" --host 127.0.0.1 --port "$PORT" \
      "${SRV_ARGS[@]}" "$@" > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    curl -sf "127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "  server died, see $log"; return 1; }
    sleep 2
  done
  echo "  server did not become healthy, see $log"; return 1
}
stop_srv(){ [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }

gen(){ # label payload-file
  curl -s --max-time 3000 "127.0.0.1:$PORT/v1/completions" \
       -H 'Content-Type: application/json' -d @"$2" \
  | python3 -c "
import sys, json
try:
    t = json.load(sys.stdin)['timings']
    n = t.get('prompt_n') or 0
    # A reused prompt is served from the prefix cache, so its prompt_per_second
    # describes the handful of uncached tokens and means nothing. Say so rather
    # than printing a number someone might paste into a table.
    pf = ('prefix hit, %d tok prefilled' % n) if n < 100 \
         else ('prefill %8.1f t/s (%d tok)' % (t['prompt_per_second'], n))
    print('  %-24s decode %6.2f t/s   %s'
          % ('$1', t['predicted_per_second'], pf))
except Exception:
    print('  %-24s NO RESPONSE' % '$1')"
}

# Deterministic long prompt, generated rather than vendored, so the fixture is
# reproducible and carries no third-party text.
make_prompts(){
  python3 - "$OUT" "$DEPTH_TOKENS" <<'PY'
import json, random, sys
out, target = sys.argv[1], int(sys.argv[2])
random.seed(20260921)
seeds = [
 "The reconciliation of the logistics ledger cross-references bill-of-lading identifiers against customs declarations filed before departure.",
 "Maintenance protocols specify torque sequences, thermal tolerances, and an inspection interval tied to operating hours rather than calendar dates.",
 "Archival records indicate the survey office renumbered parcels twice, complicating any retrospective join between tax rolls and deeds.",
 "The firmware changelog describes a race in the interrupt handler that appeared only under sustained load with fragmented packets.",
]
# 1.24 tokens per word, measured against this tokenizer on this filler text.
# Approximate by construction — the real count is reported as prompt_n below.
words, need = [], int(target / 1.24)
while len(words) < 2 * need:          # enough for two independent prompts
    words.extend(random.choice(seeds).split())
tail = "\n\nSummarise the preceding text in three sentences."
# Two independent long prompts. Timing the same one twice measures the prefix
# cache, not prefill, so every timed prefill below gets fresh text.
for name, lo in (("req_depth.json", 0), ("req_depth2.json", need)):
    json.dump({"prompt": " ".join(words[lo:lo + need]) + tail,
               "max_tokens": 192, "temperature": 0, "cache_prompt": True},
              open(f"{out}/{name}", "w"))
json.dump({"prompt": "Explain how a B-tree index works, in detail.",
           "max_tokens": 192, "temperature": 0, "cache_prompt": False},
          open(f"{out}/req_short.json", "w"))
PY
}

echo "#### BENCHMARK $LABEL  $(date -Is)"
echo "     model   $MODEL"
echo "     ctx     $CTX"
echo "     server  ${SRV_ARGS[*]:-(no extra flags)}"
echo "     out     $OUT"
[ -n "$STOP_UNIT" ] && { echo "     stopping $STOP_UNIT for the run"; systemctl --user stop "$STOP_UNIT"; }

if [[ "$PHASES" == *s* ]]; then
  echo "#### PHASE S: speed (short prompt, then ~${DEPTH_TOKENS} tokens cold and warm)"
  make_prompts
  if srv "$OUT/srv_speed.log"; then
    gen "short r1"                "$OUT/req_short.json"
    gen "short r2"                "$OUT/req_short.json"
    gen "depth r1 (cold)"         "$OUT/req_depth.json"
    gen "depth r2 (same prompt)"  "$OUT/req_depth.json"
    gen "depth r3 (fresh prompt)" "$OUT/req_depth2.json"
    echo "  VRAM at idle+load: $(vram_used) MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
    stop_srv
  fi
fi

if [[ "$PHASES" == *q* ]]; then
  echo "#### PHASE Q: perplexity (c=8192)"
  if [ -n "$PPL_CORPUS" ] && [ -f "$PPL_CORPUS" ]; then
    wait_vram_free
    "$BIN/llama-perplexity" -m "$MODEL" -f "$PPL_CORPUS" -c 8192 --no-warmup \
        "${SRV_ARGS[@]}" 2>&1 | tee "$OUT/ppl.log" | grep -E 'Final estimate|error' | tail -1
  else
    echo "  skipped: pass --ppl-corpus /path/to/wiki.test.raw"
    echo "  (wikitext-2-raw, not vendored here: it is CC BY-SA and not ours to redistribute)"
  fi
fi

if [[ "$PHASES" == *p* ]]; then
  echo "#### PHASE P: structured-output probe (24 tasks)"
  if srv "$OUT/srv_probe.log"; then
    BENCH_OUT="$OUT" python3 "$(dirname "$0")/bench/structured_probe.py" \
        "http://127.0.0.1:$PORT" "$LABEL" "$OUT" | tail -3
    stop_srv
  fi
fi

if [[ "$PHASES" == *n* ]]; then
  echo "#### PHASE N: needle-in-a-haystack near full context"
  if srv "$OUT/srv_niah.log"; then
    python3 "$(dirname "$0")/bench/niah_check.py" \
        --base "http://127.0.0.1:$PORT" \
        --ctx "$(( CTX * 85 / 100 ))" --needles 15 --max-tokens 800 2>&1 | tail -8
    stop_srv
  fi
fi

echo "#### DONE $(date -Is)   logs in $OUT"
