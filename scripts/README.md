# scripts

## benchmark.sh — the measurement method

Every throughput, quality and recall number quoted in a model entry should come
from this script, so the tables can be compared across entries. An entry that
quotes numbers should say which commit it was run at.

```bash
./scripts/benchmark.sh --label flashnext \
    --model ~/ai-models/flashnext-gsq-q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --llama-dir ~/src/llama.cpp --ctx 131072 --phases sn \
    -- --no-warmup -lm dio -ngl 99 -ot 'ffn_.*_exps=CPU' -fa on \
       -ctk q8_0 -ctv q8_0 -b 4096 -ub 2048
```

Everything after `--` goes to `llama-server` verbatim. To A/B a single flag, run
it twice changing only what follows `--`; nothing else about the run differs.

| Option | Meaning |
|---|---|
| `--label` | name for the run, used in output paths |
| `--model` | GGUF path (first shard for a split model) |
| `--llama-dir` | llama.cpp checkout; binaries expected in `build/bin` |
| `--ctx` | context to serve at |
| `--phases` | `s` speed, `q` perplexity, `p` structured output, `n` recall |
| `--stop-unit` | systemd `--user` unit to stop for the run, restarted on exit |
| `--ppl-corpus` | path to `wiki.test.raw` for phase `q` |
| `--depth-tokens` | size of the long prompt in phase `s` (default 30,000) |
| `--port` | defaults to 8299, deliberately not the serving port |
| `--out` | output directory, defaults to `bench-<label>-<timestamp>` |

### Things it does on purpose

- **Starts its own server on its own port.** It never measures against a
  deployed endpoint, whose cache state you do not control.
- **Restarts `--stop-unit` from an EXIT trap**, so an interrupted or failed run
  still hands the machine back the way it found it. Only one large model fits
  at a time, which is why stopping one is necessary at all.
- **Generates its long prompt deterministically** from a fixed seed rather than
  shipping a text fixture. Reproducible, and no third-party text in the repo.
  The word-to-token ratio is approximate; the real count is reported as
  `prompt_n` on every line, so quote that.
- **Every timed prefill uses fresh text.** Phase `s` generates two independent
  long prompts: `r1` prefills one cold, `r2` re-sends the same one so you can
  see the prefix cache work, `r3` prefills the second cold as an independent
  sample. When a response comes back from the prefix cache the line says
  `prefix hit` instead of printing a prefill rate, because that rate describes
  only the handful of uncached tokens. Both the original version of this
  measurement and the first version of this script quoted such a number as
  prefill; it is a 4-token figure wearing a 30,000-token label.
- **Skips perplexity unless you point at a corpus.** wikitext-2-raw is CC BY-SA
  and not ours to redistribute; fetch it yourself and pass `--ppl-corpus`.

### Helpers

`bench/niah_check.py` plants unique needles at uniform depths and scores exact
recall per depth band. `bench/structured_probe.py` runs 24 scored
structured-output tasks. Both are stdlib-only and can be run directly against
any OpenAI-compatible endpoint.

## llm-run.sh — guarded launcher

Refuses to start when another llama.cpp loader is alive, then runs the command
in its own systemd scope with a hard memory cap and swap disabled. Use it for
ad-hoc experiments so a runaway is reclaimed inside that cgroup instead of
inviting the OOM killer to pick a victim for you.
