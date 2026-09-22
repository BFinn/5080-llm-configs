# patches

Five patches against llama.cpp master `ec92815`. Applied in order they produce the
binary the [Flash-Next entry](../) deploys: a Q2_0 MoE with its experts in host RAM and
a GPU-resident LRU cache in front of them.

They are `git format-patch` output, so apply them with `git am` and they keep their
authorship:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp-lru-new
git -C ~/src/llama.cpp-lru-new checkout ec92815
git -C ~/src/llama.cpp-lru-new am "$PWD"/000*.patch
```

## What each one is for

| | Patch | Author | Role |
|---|---|---|---|
| `0001` | AVX2 `vec_dot` kernel for Q2_0 x Q8_0 | mine | Mainline bound `ggml_vec_dot_q2_0_q8_0` to the scalar `_generic` on x86. This is the real kernel. 9.5x single-thread over scalar, 0 mismatches in 2000 random trials. |
| `0002` | MoE expert cache: GPU-resident LRU cache | **csantiago78** | The cache itself, cherry-picked from [llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861). Adds `--moe-expert-cache`. Not my work — see [NOTICE](../../../NOTICE). |
| `0003` | Synchronize the backend before publishing cache updates | mine | Correctness fix on top of `0002`. |
| `0004` | Windowed admission gate and upload telemetry | mine | Performance policy on top of `0002`. Worth 12-14% decode over the PR's default. |
| `0005` | Emit telemetry on stderr | mine | Ergonomics. `llama-server` does not surface library `INFO` logs at default verbosity, so the periodic stats line never reached the journal. |

Sizes, so you know what you are reading: `0001` is 2 files and 48 lines, `0002` is 12
files and 645, `0003` is 3 files and 14, `0004` is 1 file and 44, `0005` is 1 file and 4.

## How they depend on each other

```
0001  AVX2 kernel ......... independent, but see below
0002  expert cache ........ needs nothing here; 0003-0005 all patch its files
  0003  sync fix .......... needs 0002
  0004  admission gate .... needs 0002
  0005  stderr telemetry .. needs 0002 and 0004 (it prints 0004's counters)
```

`0001` and `0002` touch disjoint files, so the ordering between them is a convenience,
not a constraint. Everything else patches `src/llama-moecache.cpp`, which `0002`
creates.

**`0001` is a prerequisite of the cache, not an optimisation of it.** Only the CPU
`mul_mat_id` understands the cache's skip table, so every cache *miss* is computed on
the CPU. Without a real Q2_0 kernel there, the miss path runs scalar and eats the gain
the cache just bought.

## Which patches you need

| You want | Apply | Context | Decode |
|---|---|---|---|
| The deployed arm | all five | 131,072 | 50-54 tok/s |
| Maximum context, no cache | `0001` only | 262,144 | ~33 tok/s |

The unit file in this entry is the first row. It passes `--moe-expert-cache`, which does
not exist until `0002` is applied, so a kernel-only build will refuse the argument and
never start.

## Flags and environment variables these add

| Introduced by | Name | Default | Meaning |
|---|---|---|---|
| `0002` | `--moe-expert-cache N` | 0 (off) | GPU cache slots per host-resident expert layer. 64 is ~4 GiB on this model. Also settable as `LLAMA_ARG_MOE_EXPERT_CACHE`. |
| `0002` | `--moe-expert-cache-inserts N` | 2 | Max expert uploads per layer per decode step, i.e. the upload throttle. The unit passes `2` explicitly, which is the default. Also `LLAMA_ARG_MOE_EXPERT_CACHE_INSERTS`. |
| `0004` | `LLAMA_MOE_CACHE_ADMIT` | 1 | Sightings before an uncached expert is admitted. 1 reproduces the PR's unconditional behaviour. |
| `0004` | `LLAMA_MOE_CACHE_WINDOW` | 16 | Steps between halvings of the use counter. 0 = never halve. |
| `0004` | `LLAMA_MOE_CACHE_DEBUG` | off | Print hit/upload/yield telemetry every 256 steps. |

Gotcha: `0002`'s commit message says to enable the cache with `LLAMA_MOE_CACHE_SLOTS`.
That is from an earlier revision of the PR. The code in this cherry-pick is driven by
the CLI parameter, and `LLAMA_MOE_CACHE_SLOTS` survives only in one log string and a
couple of comments. Use `--moe-expert-cache`.

## Two things that will bite you

- **Do not combine the cache with `GGML_OP_OFFLOAD_MIN_BATCH=1`.** Both want the
  per-token expert matmul, and together they measured at half speed (16.8 tok/s). With
  the cache on, leave that variable at its default.
- **`0002`'s own upstream base predates lazy PLE loading.** On that base `-lm dio` pins
  all 62 GiB and the process is OOM-killed. Applying the series to `ec92815`, as above,
  avoids it; if you rebase onto something older, use `-lm mmap`.

## Provenance

`0002` is redistributed with csantiago78's authorship intact, including their own commit
trailers, because the configuration documented here cannot be reproduced without it. It
was unmerged at the time of writing, so expect it to need rebasing — or to be replaced
by whatever lands upstream. `0001` is not upstreamed either.

[NOTICE](../../../NOTICE) records who wrote what and under which license.
