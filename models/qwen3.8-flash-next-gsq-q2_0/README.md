# Qwen3.8-Flash-Next (177B MoE) on a single RTX 5080 16 GB

**A 512-expert mixture-of-experts model with a 131,072-token context, served at
~1,100 tok/s prefill and 50-54 tok/s decode on one consumer 16 GB GPU.**

The GPU-resident expert cache that buys that decode rate costs context: the same machine
will serve 262,144 tokens without it, at ~33 tok/s. Both arms are documented here, and
[the trade is spelled out](#breaking-the-wall-a-gpu-resident-lru-expert-cache).

Status: deployed and serving. All numbers below are measured on the hardware in
[`../../README.md`](../../README.md), not estimated, unless a row says otherwise.

---

## Why this fits at all

**177B parameters, 66.4 GB on disk.** The two shards split along the architecture,
and are quantized differently:

| Shard | Parameters | On disk | Effective |
|---|---|---|---|
| 1 of 2 — MoE backbone | 125.74B | 37.62 GB | 2.39 bpw |
| 2 of 2 — n-gram / PLE tables | 51.20B | 28.80 GB | 4.50 bpw |
| **served total** | **176.94B** | **66.42 GB** | **3.00 bpw** |

The model card reads "125B with 6B activated, plus 51B n-gram embedding and 4B MTP", so
the tables are *additional* to the headline 125B rather than part of it. Worth stating,
because the two figures get conflated and only one of them is arithmetically possible:
this file at 125B would be 4.25 bpw, which no Q2_0 build is. The 4B MTP head and the
vision projector ship as separate files and neither is loaded here.

The GGUF also reports `size_label = 512x56B`: 48 layers, 512 experts per layer,
**10 active per token**, 2560 embedding width, plus a 16-head n-gram embedding table.
llama.cpp calls that table the PLE, after its `ple.ngram_size` and `ple.heads_per_ngram`
metadata keys, and this page uses that name throughout — it is the same thing the tech
report calls n-gram embeddings. Native context is 262,144.

Only ~2% of expert weights are touched for any given token. So the weights do not need
to live on the GPU. They live in **pinned host RAM**, and the GPU pulls the ~10 active
experts per layer across PCIe and does the arithmetic. The GPU holds attention, the
GDN layers, and the KV cache. That is the whole trick:

| Component | Lives on | Size |
|---|---|---|
| Routed experts (Q2_0) | CPU RAM, pinned | 29.0 GiB |
| N-gram / PLE tables | Page cache, file-backed | ~27 GiB |
| Attention / GDN / KV cache | GPU (16 GB) | 14.8 GiB |

The binding constraint is **PCIe bandwidth**, not VRAM and not the SSD. Measured: a
single 30K prefill moves 640-690 GiB host-to-device. Getting that transfer fast is what
the whole configuration is about.

---

## Final configuration

Full unit file: [`flashnext-server.service`](flashnext-server.service).
Replace `__HOST_IP__`, and put the API key in `~/.config/llama/api-key` at mode 600 —
`--api-key` on the command line is readable by every local user via `ps`.

This is the deployed arm: expert cache on, 131K context. It needs the **whole** patch
series in [`patches/`](patches/), not just the AVX2 kernel — `--moe-expert-cache` does
not exist until `0002` is applied. See [Reproducing](#reproducing).

```bash
llama-server \
  -m Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --no-warmup -lm dio \
  -ngl 99 -ot ffn_.*_exps=CPU \
  --moe-expert-cache 64 --moe-expert-cache-inserts 2 \
  -fa on -ctk q8_0 -ctv q8_0 -c 131072 -b 4096 -ub 2048 -t 12 -tb 12 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --cache-reuse 256 --jinja --parallel 1
# plus: LLAMA_MOE_CACHE_ADMIT=3 LLAMA_MOE_CACHE_WINDOW=32
# and deliberately NOT GGML_OP_OFFLOAD_MIN_BATCH -- see the flag table below
```

For the 262K arm instead: drop both `--moe-expert-cache*` flags, set
`GGML_OP_OFFLOAD_MIN_BATCH=1`, and raise `-c` to 262144. That is what every table from
here down to [Quality](#quality) was measured on.

Every flag that matters, and why:

| Flag | Why |
|---|---|
| `-lm dio` | Pinned host memory via direct IO. Pageable mmap caps PCIe at ~10.7 GB/s; pinned reaches 20.0 GB/s. Worth 1.55x prefill on its own. |
| `-ot ffn_.*_exps=CPU` with `-ngl 99` | Explicit split: experts on CPU, everything else on GPU. **Do not use `--fit` here.** At 262K, `--fit` makes a larger KV reservation and silently pushes non-expert layers to the CPU, collapsing decode from ~33 to 15.5 tok/s at identical VRAM. |
| `--moe-expert-cache 64 --moe-expert-cache-inserts 2` | GPU-resident LRU cache of recently used expert slices, 64 slots per layer, ~4 GiB of VRAM. Decode 34 to 50-54 tok/s, at the cost of dropping context from 262K to 131K. Requires patches `0002`-`0005`. |
| `GGML_OP_OFFLOAD_MIN_BATCH=1` | **262K arm only.** Default is 32. Below that batch size, ggml keeps the matmul on the CPU. Decode is batch-1, so every token hit the CPU kernel. Setting 1 routes per-token expert matmuls to the GPU. Decode 10.4 to 38.5 tok/s with no recompile. **Leave it at the default when the expert cache is on** — cache misses are computed by the CPU matmul, and combining the two measured at half speed (16.8 tok/s). |
| `--no-warmup` | Default warmup touches all weights and evicts the page cache that the PLE table depends on. |
| `-ub 2048 -b 4096` | Prefill cost scales with expert bytes per micro-batch. Larger ubatch amortizes the expert sweep. Worth ~2.6x over `-ub 512`. |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache. Required to hold this much context in the VRAM the experts and the cache leave free. |
| `--parallel 1` | One slot. Concurrency would multiply the KV reservation. |

Note: `--cache-reuse` is accepted but the server disables it for this architecture.
Ordinary prefix caching still works, which matters a lot for multi-turn use.

---

## Benchmarks

Everything from here to [Quality](#quality) was measured on the **262K arm, before the
expert cache**: `-c 262144`, `GGML_OP_OFFLOAD_MIN_BATCH=1`, no `--moe-expert-cache`. It
is kept because it is how the configuration was found and because 262K is still a real
option. For the deployed 131K numbers, go to
[breaking the wall](#breaking-the-wall-a-gpu-resident-lru-expert-cache).

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

### The 262K arm, measured on the live endpoint

Deployed configuration at the time of measurement; superseded by the expert-cache arm.

| Metric | Value |
|---|---|
| Prefill, 30K-token prompt | 1,079 tok/s (27.7 s) |
| Decode | 28.7 - 34.7 tok/s |
| Time to first token, 30K prompt | 28 s |
| Time to first token, 190K prompt | ~245 s |
| Cold start to serving | 26 s |
| VRAM | 14.8 GiB of 15.92 usable (16303 MiB) |
| Host RSS | 34.8 GiB |
| Context | 262,144 tokens, single slot |

### Throughput vs context length

The decode number above comes from a short prompt. Both rates drop as the prompt grows.
Measured cold on the live endpoint, thinking off, temperature 0. Still the 262K arm — the
187K row does not exist on the deployed 131K configuration, whose nearest measured point
is a 119K prompt at 825 tok/s prefill and 19.7 tok/s decode.

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

All three measured with a q8_0 KV cache, so the column compares weight quantization and
nothing else. The 27B's [own entry](../qwen3.8-27b/) quotes 6.8412 instead, because it is
deployed with a q4_0 KV cache.

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
at 262K context neither fits at all, because context itself has already spent the
headroom. That is what made the expert cache, rather than static residency, the thing
worth building.

| Goal | Context | Config | Prefill | Decode |
|---|---|---|---|---|
| Full context | 262,144 | ub 2048, q8_0 KV | 1,102 | 34.1 |
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
is a smaller share of VRAM than the 131K-versus-262K difference suggests. At 683 MiB per
layer that buys **two** resident layers, not the seven the estimate implied, and decode was
marginally lower.

Two is not worth taking. By the bandwidth model above, two layers remove 4.2% of the
per-token byte budget and should return under 3%, roughly +1 tok/s on 34.1. They would also
leave about 1.5 GiB of headroom at 262K, between the 2.3 GiB that worked at six resident
layers and the 0.3 GiB that OOMed at nine — and a 262K prefill needs more scratch than the
131K those rows were measured at. Untested, small upside, real OOM risk.

Why the KV cache is so small in the first place, from the GGUF metadata: only **12 of the 48
layers hold one**, because `full_attention_interval` is 4 and the other 36 are gated-delta-net
layers whose state is fixed-size regardless of sequence length. Those 12 use `head_count_kv` 2
against `head_count` 24. That is 12 × 2 × (K+V) × 256 = 12,288 values per token, about
13.1 KB/token at q8_0, so ~3.2 GiB at 262K. Estimating KV from context length alone, as if
every layer cached and heads were ungrouped, is what produced the ~5 GiB figure.

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

## Breaking the wall: a GPU-resident LRU expert cache

Everything above says decode is bounded by 633 MiB of expert streaming per token and that
no static placement helps. The routing measurement says the working set *within a
context* is small and recency-local. The matching fix is a cache of recently used expert
slices in VRAM, which is exactly what an unmerged llama.cpp pull request implements
(ggml-org/llama.cpp#27861). It is deployed here.

**Mechanism.** The fused expert tensor stays whole in host RAM. Each cached layer gets a
companion device tensor of K slots plus one all-zero slot. Expert ids are remapped
through a device table into a second `mul_mat_id` over the cache; uncached ids hit the
zero slot and contribute nothing. The host `mul_mat_id` receives the same table and skips
cached ids. The two outputs are summed, so the split is exact. Uploads are throttled and
asynchronous. Decode-only (`n_tokens == 1`), so prefill is untouched.

**What is deployed** (branch `lru-new`, patches in [`patches/`](patches/)):
the PR commit cherry-picked onto current master, the AVX2 Q2_0 kernel, and a one-line
fix that synchronizes the backend before the cache publishes new slot tables. The PR head
has that sync commented out; a still-running CUDA graph could read a torn table. Flags:
`--moe-expert-cache 64 --moe-expert-cache-inserts 2`, plus `LLAMA_MOE_CACHE_ADMIT=3`
`LLAMA_MOE_CACHE_WINDOW=32` for the admission gate below.

**Two things you must get right.** Misses are computed by the CPU matmul, because only the
CPU op understands the skip table. So `GGML_OP_OFFLOAD_MIN_BATCH` must stay at its default
when the cache is on; combining them is not corrupt, just half the speed (16.8 tok/s). That
also means the CPU path needs a real Q2_0 kernel, i.e. the AVX2 patch is a prerequisite,
not an optimisation. And the PR's own base predates lazy PLE loading, so on that base
`-lm dio` pins all 62 GiB and the process is OOM-killed; use current master or `-lm mmap`.

### Measured, clean run (newer base, direct IO, nothing else running)

| 131K context | VRAM | Fresh prose | Extractive | Prefill 30K |
|---|---|---|---|---|
| No cache (CPU miss path) | 9.6 GiB | 32.9-34.8 tok/s | 34.0 | 654 (cold) |
| **Cache, 64 slots/layer** | 13.7 GiB | **48.8-51.5** | **44.5** | 1,101 |
| Cache, 80 slots/layer | 14.7 GiB | 50.6-54.3 | 44.8 | 1,086 |
| Cache, 96 slots/layer | OOM at load | | | |

Decode **+50%** on fresh prose and **+31%** on extractive output. One slot per layer
costs 63 MiB across 48 layers, so 64 slots is 4 GiB. The prefill column is not a cache
effect: the no-cache arm ran first with a cold page cache; warm prefill at this micro-batch
is ~1,100 either way.

Correctness: extractive output at temperature 0 was **identical word-for-word** to the
uncached run at both slot counts, and decode-mode perplexity (`-b 1 -ub 1`, which forces
`n_tokens == 1` so the cache is exercised) was 3.269 with the cache against 3.293 without,
inside the error bars.

**Live endpoint after deployment:** 44 tok/s with a cold cache, 49.7 warm. A 119K-token
prompt prefilled at 825 tok/s (145 s) and decoded at 19.7 tok/s with 3/3 planted facts
recalled and no allocation failure at 14.2 GiB. At that depth attention dominates and the
cache gain shrinks; the headline gain is for prompts under ~60K.

**The cost is context.** 64 slots do not fit beside a 262K KV cache; 20 slots do not
either. The deployed context is now 131,072. That is the trade this machine offers: 262K
at 34 tok/s, or 131K at 50.

### Slot count on a 16 GB card

Headroom is against the card's 15.92 GiB usable (16303 MiB), not a round 16.

| Slots/layer | VRAM for cache | Total VRAM | Fits at 131K? |
|---|---|---|---|
| 64 | 4.0 GiB | 13.7 GiB | yes, ~2.2 GiB headroom |
| 80 | 5.0 GiB | 14.7 GiB | yes, ~1.2 GiB headroom, no gain over 64 |
| 96 | 6.0 GiB | — | no, OOM at load |

Confirmed in service: the deployed 64-slot configuration sits at 13.82 GiB with **1.65 GiB
free**, and that survives a 121K-token prefill.

### Admission gate: the second win

The PR admits every miss to the cache unconditionally. A contributor to its thread found
that this churns on weak host RAM, with uploads roughly equal to evictions and the hit
rate collapsing *because* of the churn, and that a windowed use counter fixed it. No code
had been published at that point, so the policy was reimplemented from the description
([`patches/0004-*`](patches/)): a `uint8` counter per layer and expert, halved every
`LLAMA_MOE_CACHE_WINDOW` steps, and an uncached expert is uploaded only after
`LLAMA_MOE_CACHE_ADMIT` sightings. Eviction stays LRU, which the same contributor found
to beat frequency-based victims. They have since posted a patch of their own to that
thread, carrying the same policy as proper CLI options rather than environment variables;
theirs is the better interface if this ever lands upstream.

Same binary, 64 slots, 131K, ungated arms run first and last as the order control:

| Policy | Steady prose | Six shifting topics (mean) | Extractive |
|---|---|---|---|
| Ungated (PR default), first | 47.6 / 49.5 tok/s | 42.9 | 43.0 |
| admit 3, window 16 | 50.9 / 51.8 | 46.9 | 46.5 |
| admit 2, window 16 | 51.1 / 51.2 | 44.6 | 47.9 |
| **admit 3, window 32** | **53.4 / 54.1** | **47.7** | **48.0** |
| Ungated, last | 47.5 / 46.3 | 43.9 | 41.9 |

Admit 3 with a 32-step window is 12-14% faster steady, 9-11% under topic shifts and
12-15% on extractive output, over an already-cached baseline. It is deployed.

Telemetry from the live service (`LLAMA_MOE_CACHE_DEBUG=1`, printed to stderr because the
server does not surface library INFO logs at default verbosity):

```
steps=768 hits=256906 misses=114994 hit-rate=69.1% uploads=17709 evicts=14637
up=22.8GiB served=330.8GiB yield=14.51x admit=3 window=32
```

A 69% hit rate at 64 slots on fresh prose sits on the published LRU curve for this model
class. Yield, bytes served per byte uploaded, is the number to watch rather than hit rate,
because hit rate alone hides churn.

Combined with the cache itself, decode on this machine went **34 to 50-54 tokens per
second** at 131K context, with output unchanged.

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
[`patches/0001-*`](patches/), 48 lines against
llama.cpp master `ec92815`. Unpack 8 packed bytes into 32 int8 codes without per-byte
shifts: `cvtepu8_epi32`, spread the even and odd bit-pairs with shifts of 12 and 6/18,
mask `0x03030303`, subtract 1 to map `{0,1,2,3}` to `{-1,0,1,2}`, then
`mul_sum_i8_pairs_float`.

Validation of the kernel:

| Check | Result |
|---|---|
| Correctness vs scalar, 2,000 random trials | 0 mismatches |
| Unpack logic, all 256 packed-byte patterns | 0 mismatches |
| Single-thread throughput | 0.8 to 8.0 GB/s (**9.5x**) |
| `test-quantize-perf`, plain AVX2, no VNNI | 53.2 to 6.2 cycles/32 vals (**8.5x**) |
| `test-quantize-perf`, with AVX-512-VNNI | 53.2 to 5.1 cycles/32 vals (**10.4x**) |
| Model-level decode, 64K, CPU path | 10.4 to 31.7-33.1 tok/s (**3.1x**) |
| Perplexity, CPU path vs GPU path, 6-chunk subset | 3.3396 vs 3.3397 |

The perplexity row is a path-equality check, not a quality score. Two different kernels
producing the same perplexity to four decimals is strong evidence the AVX2 kernel is
numerically correct at model level.

The two `test-quantize-perf` rows separate what the 9.5x row does not. This machine has
AVX-512-VNNI, and `mul_sum_i8_pairs_float` uses it when present, so the original figure
is really AVX2-plus-VNNI against scalar. Measured apart on current master, the plain-AVX2
path is worth 8.5x on its own and VNNI adds a further 1.2x. That split is the interesting
one, because most x86 CPUs in use have AVX2 and no VNNI.

Both fixes are deployed together. The env var is the active path; deleting that one line
from the unit falls back to the AVX2 CPU kernel at roughly the same decode speed and less
PCIe traffic. The patch is **not upstreamed** — see
[`patches/`](patches/#provenance) for how it relates to the open upstream PR.

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

This builds the deployed arm. The unit passes `--moe-expert-cache` and expects its
binaries in `~/src/llama.cpp-lru-new/build/bin`, so a kernel-only build will not start
it — apply the whole series, and clone to that path or edit the two path fragments in
the unit.

```bash
# 1. Model (66.4 GB, 2 shards) from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, Q2_0
# 2. llama.cpp at master ec92815, with all five patches:
#      0001 AVX2 Q2_0 kernel        0002 expert cache (llama.cpp PR #27861, csantiago78)
#      0003 backend sync fix        0004 admission gate        0005 telemetry
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp-lru-new
git -C ~/src/llama.cpp-lru-new checkout ec92815
git -C ~/src/llama.cpp-lru-new am "$PWD"/patches/000*.patch
# 3. Build with CUDA:
cmake -S ~/src/llama.cpp-lru-new -B ~/src/llama.cpp-lru-new/build -DGGML_CUDA=ON
cmake --build ~/src/llama.cpp-lru-new/build -j
# 4. Put the API key where the unit expects it, readable only by you:
install -d -m 700 ~/.config/llama && install -m 600 /dev/null ~/.config/llama/api-key
printf '%s\n' 'YOUR-KEY' > ~/.config/llama/api-key
# 5. Install the unit, substituting __HOST_IP__:
cp flashnext-server.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now flashnext-server
```

For the 262K arm, apply only `0001` and edit the flags as described under
[Final configuration](#final-configuration).

## Known limits and open items

- One request at a time. Concurrent callers queue.
- No vision. The BF16 projector was dropped to spend that VRAM on context and the expert
  cache instead. Re-adding it costs roughly 1 GiB and some context.
- Long prompts cost real latency up front: ~28 s at 30K, ~2.5 min at 119K on the deployed
  arm (~4 min at 190K on the 262K arm). Prefix caching means a multi-turn conversation
  pays this once.
- Cannot run alongside another large model on this machine.
- MTP draft head is incompatible with this GSQ build (`output_hc_norm` dimensions), so
  speculative decode is not available to recover decode speed.
- Untested: GSQ IQ2_XS variant (63.4 GiB), REAP-256-duo Q3_K_XL (57.7 GiB).
- The AVX2 kernel is not upstreamed.
