# Qwen3.8-Flash-Next (512x56B MoE) on a single RTX 5080 16 GB

**A 512-expert mixture-of-experts model with a 262,144-token context, served at
~1,080 tok/s prefill and ~33 tok/s decode on one consumer 16 GB GPU.**

Status: deployed and serving. All numbers below are measured on the hardware in
[`../../README.md`](../../README.md), not estimated, unless a row says otherwise.

---

## Why this fits at all

The model's GGUF metadata reports `size_label = 512x56B`: 48 layers, 512 experts per
layer, **10 active per token**, 2560 embedding width, plus a 16-head PLE (per-layer
embedding) lookup table. Native context is 262,144.

Only ~2% of expert weights are touched for any given token. So the weights do not need
to live on the GPU. They live in **pinned host RAM**, and the GPU pulls the ~10 active
experts per layer across PCIe and does the arithmetic. The GPU holds attention, the
GDN layers, and the KV cache. That is the whole trick:

| Component | Lives on | Size |
|---|---|---|
| Routed experts (Q2_0) | CPU RAM, pinned | 29.0 GiB |
| PLE lookup table | Page cache, file-backed | ~27 GiB |
| Attention / GDN / KV cache | GPU (16 GB) | 14.8 GiB |

The binding constraint is **PCIe bandwidth**, not VRAM and not the SSD. Measured: a
single 30K prefill moves 640-690 GiB host-to-device. Getting that transfer fast is what
the whole configuration is about.

---

## Final configuration

Full unit file: [`flashnext-server.service`](flashnext-server.service).
Replace `__API_KEY__` and `__HOST_IP__` before use.

```bash
llama-server \
  -m Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --no-warmup -lm dio \
  -ngl 99 -ot ffn_.*_exps=CPU \
  -fa on -ctk q8_0 -ctv q8_0 -c 262144 -b 4096 -ub 2048 -t 12 -tb 12 \
  --cache-reuse 256 --jinja --parallel 1
# plus: GGML_OP_OFFLOAD_MIN_BATCH=1
```

Every flag that matters, and why:

| Flag | Why |
|---|---|
| `-lm dio` | Pinned host memory via direct IO. Pageable mmap caps PCIe at ~10.7 GB/s; pinned reaches 20.0 GB/s. Worth 1.55x prefill on its own. |
| `-ot ffn_.*_exps=CPU` with `-ngl 99` | Explicit split: experts on CPU, everything else on GPU. **Do not use `--fit` here.** At 262K, `--fit` makes a larger KV reservation and silently pushes non-expert layers to the CPU, collapsing decode from ~33 to 15.5 tok/s at identical VRAM. |
| `GGML_OP_OFFLOAD_MIN_BATCH=1` | Default is 32. Below that batch size, ggml keeps the matmul on the CPU. Decode is batch-1, so every token hit the CPU kernel. Setting 1 routes per-token expert matmuls to the GPU. Decode 10.4 to 38.5 tok/s with no recompile. |
| `--no-warmup` | Default warmup touches all weights and evicts the page cache that the PLE table depends on. |
| `-ub 2048 -b 4096` | Prefill cost scales with expert bytes per micro-batch. Larger ubatch amortizes the expert sweep. Worth ~2.6x over `-ub 512`. |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache. Required to hold 262K in the remaining VRAM. |
| `--parallel 1` | One slot. Concurrency would multiply the KV reservation. |

Note: `--cache-reuse` is accepted but the server disables it for this architecture.
Ordinary prefix caching still works, which matters a lot for multi-turn use.

---

## Benchmarks

### How the configuration was arrived at

Each row changes one thing from the row above. 30K-token prompt, same hardware.

| Configuration | Prefill (cold/warm) | Decode | PCIe | VRAM |
|---|---|---|---|---|
| UD-IQ4_XS, `-lm mmap`, c=114688 (starting point) | 288 / 449 tok/s | 26.5 tok/s | 10-12.7 GB/s | 13.0 GiB |
| GSQ-RCO Q2_0, `-lm mmap`, c=65536 | 759 / 854 tok/s | 10.3 tok/s | 10.7 GB/s | 13.4 GiB |
| GSQ-RCO Q2_0, `-lm dio` (pinned), c=114688 | 786 / 1323 tok/s | 9.8 tok/s | 20.0 GB/s | 13.8 GiB |
| + `GGML_OP_OFFLOAD_MIN_BATCH=1`, explicit split, c=262144 | 1081 tok/s | 33 tok/s | — | 14.8 GiB |

Clean attribution for the prefill gain: **1.9x from halving expert bytes** (55.4 to
29.0 GiB) **x 1.55x from pinning the host buffers** = 2.95x overall.

### Deployed configuration, measured on the live endpoint

| Metric | Value |
|---|---|
| Prefill, 30K-token prompt | 1,079 tok/s (27.7 s) |
| Decode | 28.7 - 34.7 tok/s |
| Time to first token, 30K prompt | 28 s |
| Time to first token, 190K prompt | ~245 s |
| Cold start to serving | 26 s |
| VRAM | 14.8 GiB of 16.3 |
| Host RSS | 34.8 GiB |
| Context | 262,144 tokens, single slot |

### Throughput vs context length

The decode number above comes from a short prompt. Both rates drop as the prompt grows.
Measured cold on the live endpoint, thinking off, temperature 0.

| Prompt tokens | Prefill | Time to first token | Decode |
|---|---|---|---|
| 27 | — | — | 31.8 tok/s |
| ~1.5K | — | — | 33.5 - 34.1 tok/s |
| 31,571 | 1,189 tok/s | 26.5 s | — |
| 58,006 | 1,025 tok/s | 56.6 s | 25.4 tok/s |
| 187,236 | 728 tok/s | 257.1 s | 16.9 tok/s |

At 187K, decode is about half the headline rate. Use the row nearest your prompt size.

Accuracy held up. Planted facts came back 2/2 at 58K and 3/3 at 187K, with them placed at
the start, middle and end of the prompt.

Prefix caching is worth more than any of this. The second turn of a 31.5K conversation
reused 31,576 tokens and came back in 1.8 s instead of 27.6 s. Keep the conversation
going rather than re-sending the context.

### Queueing behaviour

`--parallel 1` means a second caller waits instead of sharing the GPU. Three requests of
about 5 s each, sent at the same time, came back at 5.4 s, 10.5 s and 15.7 s. Each ran at
full speed in turn. Queue time simply adds up, so firing agent calls in parallel buys
nothing here.

### The SSD drops out entirely

Measured with caches dropped first. This is the payoff of picking a quant that fits RAM.

| Request | Prefill | Major page faults | Disk read |
|---|---|---|---|
| 1st (cold PLE) | 694 tok/s | 3,219 | 1.45 GiB |
| 2nd | 1,375 tok/s | 3,098 | **0.00 GiB** |
| 3rd | 1,372 tok/s | 3,083 | **0.00 GiB** |
| 4th | 1,371 tok/s | 3,167 | **0.00 GiB** |

Steady state is perfectly reproducible with zero disk traffic. The earlier IQ4_XS build
re-read 2.0-2.7 GiB from NVMe on *every* request forever, because its 82 GiB working set
never fit in 61 GB of RAM.

### Quality

| Measure | GSQ Q2_0 | UD-IQ4_XS | Qwen3.8-27B IQ3_S |
|---|---|---|---|
| wikitext-2 perplexity @ c8192 | 4.558 | 4.124 | 6.792 |
| Needle-in-a-haystack @ 190K | 19/20 | — | — |
| Structured-output probe (64K/112K/262K) | 24/24 | 24/24 | — |

Quantizing to ~2.4 bpw costs 10.5% perplexity against the 4-bit build, and still beats a
dense 27B at 3 bits by a wide margin. The vendor reports 89.07 task average against
~93.1 for the base model, so about 95.6% of full quality retained.

---

## Where the ceiling actually is

Everything above tunes a machine that is already at a hard limit. It is worth stating the
limit precisely, because it decides which optimisations are worth attempting and which are
guaranteed to fail.

Summing the expert tensors from the GGUF: routed experts are 31.64 GiB across 48 layers,
675 MiB per layer. With 10 of 512 experts firing per layer, **every decoded token requires
633 MiB of expert weights to move out of DRAM.**

| Available bandwidth | Implied decode ceiling | Observed |
|---|---|---|
| 20 GB/s (measured PCIe during prefill) | 30.1 tok/s | 33-34 tok/s |
| 33.6 GB/s (this machine's full DRAM bandwidth, STREAM triad) | 50.6 tok/s | — |
| ~56 GB/s (same DIMMs at their rated speed) | 84.4 tok/s | — |

Decode is therefore not near the ceiling, it *is* the ceiling. This also explains a result
that looked like noise: the AVX2 CPU expert path and the GPU offload path measure the same,
30.0-34.7 against 32.7-34.1. Both must pull the same 633 MiB per token out of DRAM. One
computes locally, the other ships it over PCIe. The wall is upstream of both.

### The host memory clock is the single biggest lever

The DIMMs in this machine are a Kingston KF560C36 kit, rated DDR5-6000 CL36. They are
running at **3600 MT/s**, below even the 4800 JEDEC default, because EXPO is off in BIOS.
Measured bandwidth is 33.6 GB/s against roughly 56 GB/s at rated speed.

Because decode scales directly with bandwidth, this is worth up to ~1.5x on decode and
prefill for a BIOS toggle, and unlike every software lever below it costs no VRAM. Two
dual-rank 32 GB modules are hard on an AM5 memory controller, so expect to land at
5200-5600 rather than the full 6000. Even 5200 is 1.44x. **Not yet applied.**

## Tuning: what helped, what did not

Measured on this machine at 131,072 context unless stated. Each timed prefill uses a
distinct prompt, because an earlier attempt measured a prefix-cache hit (4 tokens, 0.2 s)
and reported a meaningless number.

| Configuration | VRAM | Decode | Prefill |
|---|---|---|---|
| micro-batch 2048, no residency | 9.6 GiB | 34.0 tok/s | 1,103 tok/s |
| **micro-batch 4096**, no residency | 13.5 GiB | 34.0 tok/s | **1,378 tok/s** |
| micro-batch 2048, **6 expert layers resident** | 13.6 GiB | **36.7 tok/s** | 1,149 tok/s |
| micro-batch 2048, 9 expert layers resident | 15.6 GiB | 38.8 tok/s | OOM on long prefill |

**Micro-batch 4096 buys 25% prefill** at no decode cost. **Pinning expert layers in VRAM
buys decode** in line with the bandwidth model: 6 of 48 layers removes 12.5% of the
per-token budget and returns 7.9%; 9 layers removes 18.75% and returns 14.1%. Nine layers
idles fine and dies with CUDA OOM when a long prefill needs its compute buffers.

The catch: **both levers cost the same ~4 GiB of VRAM, so you can have one, not both.** And
at the deployed 262K context neither fits at all, because context itself has already spent
the headroom. Hence the deployed configuration stays as it is.

| Goal | Context | Config | Prefill | Decode |
|---|---|---|---|---|
| Full context (deployed) | 262,144 | ub 2048, q8_0 KV | 1,102 | 34.1 |
| Max prefill | 131,072 | ub 4096, q8_0 KV | 1,378 | 34.0 |
| Max decode | 131,072 | ub 2048, 6 layers resident | 1,149 | 36.7 |

### Rejected: speculative decoding

This is the counterintuitive one, and it generalises to any sparse MoE.

n-gram speculative decoding (no draft model needed) drafted almost perfectly on an
extractive task: **0.98 acceptance, 186 of 189 tokens accepted, mean accepted run 5.89.**
Decode still got *slower*, 30.1 against 33.8.

| Speculation mode | Fresh prose | Extractive |
|---|---|---|
| none | 32.8 tok/s | 33.8 tok/s |
| ngram-mod | 34.0 | 30.1 |
| ngram-map-k4v | 34.2 | 33.5 |

Speculation pays on **dense** models because verifying six drafted tokens reuses the same
weights six times in one pass. On a **sparse MoE with per-token routing**, six tokens route
to the union of their experts, so the verification batch streams roughly six times the bytes
to produce six tokens. You convert six small transfers into one large transfer of the same
total size, then pay the drafting overhead. No win is available at any acceptance rate.

Corollary: the MTP draft head being incompatible with this GGUF costs nothing.

### Rejected: a wider SIMD kernel

This CPU is Zen 4 and has full AVX-512 including VNNI; the build detects it. An AVX-512
VNNI version of the Q2_0 kernel is writable and would not help. The existing AVX2 kernel
already reaches 8.0 GB/s single-threaded, so roughly four threads saturate the 33.6 GB/s
the memory system can deliver, and there are twelve. Adding arithmetic throughput to a
kernel that is waiting on DRAM buys nothing.

This is also why the 9.5x microbenchmark figure became 3.1x at model level. The
microbenchmark ran on cache-resident data. The model does not.

For completeness, Q2_0 also has no repack GEMM/GEMV kernels, which normally accelerate CPU
prompt processing. Also moot here, since prefill runs on the GPU.

### Rejected: q4_0 KV cache

The theory was that halving the KV cache at 262K would free enough VRAM to pin expert layers
without giving up context. It freed only 1.73 GiB, not the ~5 GiB estimated, so the KV cache
is a smaller share of VRAM than the 131K-versus-262K difference suggests. That is not enough
for even two resident layers, and decode was marginally lower.

| 262K context | VRAM | Decode | Prefill | Recall @59.7K |
|---|---|---|---|---|
| q8_0 KV (deployed) | 14.8 GiB | 34.1 tok/s | 1,102 tok/s | 3/3 |
| q4_0 KV | 13.1 GiB | 33.4 tok/s | 1,100 tok/s | 3/3 |

Recall was unaffected, so q4_0 KV remains a reasonable choice if VRAM is ever needed for
something else. It just does not unlock expert residency.

### Routing skew, measured on this model

Using [`tools/moe-skew.cpp`](../../tools/), 512 real tokens each of an encyclopedia
article and of C source, serving configuration (experts on CPU, GPU op-offload on).
Layer 47 reads as degenerate in both and is excluded as unexplained.

| Layer | Prose: distinct experts used | Prose: top 10% share | Code: distinct | Code: top 10% share |
|---|---|---|---|---|
| 0 | 392 of 512 | 37.8% | 417 | 42.6% |
| 12 | 306 | 63.4% | 334 | 62.7% |
| 24 | 300 | 63.7% | 297 | 70.9% |
| 36 | 212 | 85.4% | 228 | 78.2% |
| mean, all layers | — | 72.0% | — | 69.7% |

Two conclusions. Within a document, routing is moderately concentrated and gets more so
with depth: the top 20% of experts take 85-87% of selections, which is what makes a
recency cache of 64-128 slots per layer worthwhile. But the hot sets **do not overlap**
between prose and code; the twelve most-used experts in layer 0 are disjoint between the
two texts. The skew is per-context, not global. That is why a static "pin the popular
experts" scheme fails on this model and an LRU cache does not, and it matches the
cross-workload traces others have published for this model family.

### On choosing *which* experts to pin

You cannot. All 512 experts of a layer live in a single tensor
(`blk.N.ffn_down_exps.weight`), and `--override-tensor` matches tensor names, so the
smallest placeable unit is one layer's entire 675 MiB expert stack. Research that exploits
expert popularity skew (LRU expert caches, activation tracing, speculative expert prefetch)
needs a dynamic caching layer llama.cpp does not have. Layer choice itself appears close to
uniform, so maximise the count and do not agonise over identity.

What *does* matter, and what this configuration gets right, is keeping every always-on
tensor on the GPU. The `ffn_.*_exps` pattern deliberately does not match `ffn_*_shexp`, so
the shared experts (about 2 MiB per layer, fired on every token) stay resident along with
the router and attention. Getting that wrong costs you on every token.

## The Q2_0 decode problem

Worth its own section, because the 10 tok/s decode looked like a dead end.

**Symptom.** Switching to Q2_0 tripled prefill but dropped decode from 26.5 to ~10 tok/s,
in both load modes. Load mode was therefore not the cause.

**Root cause.** Mainline ggml has no x86 SIMD kernel for Q2_0. In
`ggml/src/ggml-cpu/arch-fallback.h`, `ggml_vec_dot_q2_0_q8_0` is bound to the scalar
`_generic` implementation on x86. Comparable types (iq3_s, iq4_xs, q4_K, q1_0) all have
AVX2 paths. Decode is batch-1 and was running on the CPU, so every token paid scalar cost.

**Fix 1, no recompile.** `GGML_OP_OFFLOAD_MIN_BATCH=1` sends the per-token expert matmul
to the GPU, which does have a Q2_0 kernel. Decode 10.4 to 38.5 tok/s at 64K.

**Fix 2, an actual AVX2 kernel.** See
[`patches/q2_0-avx2-kernel.patch`](patches/q2_0-avx2-kernel.patch), 48 lines against
llama.cpp master `ec92815`. Unpack 8 packed bytes into 32 int8 codes without per-byte
shifts: `cvtepu8_epi32`, spread the even and odd bit-pairs with shifts of 12 and 6/18,
mask `0x03030303`, subtract 1 to map `{0,1,2,3}` to `{-1,0,1,2}`, then
`mul_sum_i8_pairs_float`.

Validation of the kernel:

| Check | Result |
|---|---|
| Correctness vs scalar, 2,000 random trials | 0 mismatches |
| Single-thread throughput | 0.8 to 8.0 GB/s (**9.5x**) |
| Model-level decode, 64K, CPU path | 10.4 to 31.7-33.1 tok/s (**3.1x**) |
| Perplexity, CPU path vs GPU path, 6-chunk subset | 3.3396 vs 3.3397 |

That last row is a path-equality check, not a quality score. Two different kernels
producing the same perplexity to four decimals is strong evidence the AVX2 kernel is
numerically correct at model level.

Both fixes are deployed together. The env var is the active path; deleting that one line
from the unit falls back to the AVX2 CPU kernel at roughly the same decode speed and less
PCIe traffic. The patch is **not upstreamed**.

---

## Gotchas

Things that cost real time to discover.

1. **Never use `--fit` with an explicit `-ot` split at high context.** It rebalances
   against the larger KV reservation and moves non-expert layers to the CPU. VRAM looks
   identical while decode halves.
2. **Never pass `-ngl` to a `--fit` unit.** `--fit` aborts with "already set by user" and
   then attempts a 61 GiB `cudaMalloc`.
3. **Never `--no-mmap` on a model larger than RAM.** For the IQ4_XS build, mmap is
   load-bearing. (The flag is now `-lm/--load-mode`.)
4. **Always `--no-warmup`.** Warmup evicts the page cache the PLE table lives in.
5. **Only one large loader at a time.** Two processes mapping a 66 GB model into 61 GB of
   RAM will invoke the OOM killer, and it may take out your network daemons before it
   takes out the model. Use [`../../scripts/llm-run.sh`](../../scripts/llm-run.sh).
6. **`-ot` accepts only `CPU` and `CUDA0` in mainline**, not `CUDA_Host`. Pinning is done
   with `-lm dio` or `-lm none`, not through `-ot`.
7. **Benchmark prompts need `ignore_eos`** or the model ends instantly on raw text.
8. **This model is a reasoner.** With thinking enabled, a small `max_tokens` is consumed
   entirely by `reasoning_content` and `content` comes back empty. An early A/B of mine
   silently compared two empty strings because of this.
9. **An old alias does not fail. It serves whatever is loaded.** Ask for
   `"model": "qwen3.8-27b"` and you get HTTP 200, answered by Flash-Next, with
   `qwen3.8-flash-next` in the response. Old callers keep working and record the wrong
   model against their output. Check the `model` field that comes back.

---

## Reproducing

```bash
# 1. Model (66.4 GB, 2 shards) from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, Q2_0
# 2. Build llama.cpp master ec92815 with CUDA, then apply the kernel patch:
git apply patches/q2_0-avx2-kernel.patch && cmake --build build -j
# 3. Install the unit, substituting __API_KEY__ and __HOST_IP__:
cp flashnext-server.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now flashnext-server
```

## Known limits and open items

- One request at a time. Concurrent callers queue.
- No vision. The BF16 projector was dropped to fit 262K context in VRAM. Re-adding it
  costs roughly 1 GiB and some context.
- Long prompts cost real latency up front: ~28 s at 30K, ~4 min at 190K. Prefix caching
  means a multi-turn conversation pays this once.
- Cannot run alongside another large model on this machine.
- MTP draft head is incompatible with this GSQ build (`output_hc_norm` dimensions), so
  speculative decode is not available to recover decode speed.
- Untested: GSQ IQ2_XS variant (63.4 GiB), REAP-256-duo Q3_K_XL (57.7 GiB).
- The AVX2 kernel is not upstreamed.
