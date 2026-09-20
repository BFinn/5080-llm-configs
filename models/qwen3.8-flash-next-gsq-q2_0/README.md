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

---

## Reproducing

```bash
# 1. Model (67.3 GB, 2 shards) from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, Q2_0
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
