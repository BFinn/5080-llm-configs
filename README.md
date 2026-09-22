# RTX 5080 LLM configurations

Measured, working llama.cpp configurations for running large models on a single
**NVIDIA RTX 5080 (16 GB)** desktop. Each entry records the exact configuration, what was
tested, and the numbers, including the settings that turned out to be wrong.

The theme across these entries: on a 16 GB consumer card, the interesting question is
rarely "does it fit in VRAM". It is which tensors to leave in host RAM and how fast you
can stream them across PCIe.

Consequence worth stating up front: for a sparse MoE served this way, throughput is set by
**bytes moved per token divided by memory bandwidth**, not by arithmetic. That single fact
predicts which optimisations work and which cannot. See
[where the ceiling actually is](models/qwen3.8-flash-next-gsq-q2_0/#where-the-ceiling-actually-is).

## Hardware baseline

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GB (16303 MiB = **15.92 GiB usable**), driver 575.64.03, CUDA 12.9 |
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads, 64 MiB L3 |
| RAM | 64 GB DDR5 (2 x 32 GB Kingston KF560C36, rated 6000 CL36), **running at 3600 MT/s, EXPO off** |
| RAM bandwidth | 33.6 GB/s measured (STREAM triad, 12 threads) |
| Storage | Samsung 990 PRO 2 TB NVMe |
| PCIe | Gen5 x16 (nvidia-smi reports Gen1 at idle; that is a power state, not the link) |
| OS | Ubuntu 24.04.3 LTS, kernel 6.14, gcc 12.4 |
| Serving | llama.cpp `llama-server` under systemd user units |

## Models

| Model | Quant | Size | Context | Prefill | Decode | Entry |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (177B MoE) | GSQ-RCO Q2_0 + LRU expert cache | 66.4 GB | 131,072 | 1,101 tok/s | 50-54 tok/s | [entry](models/qwen3.8-flash-next-gsq-q2_0/) |
| Qwen3.8-27B (dense, vision) | UD-IQ3_S | 12 GB | 98,304 | 1,756 tok/s | 95-96 tok/s | [entry](models/qwen3.8-27b/) |

Prefill figures are for a 30K-token prompt. Decode for the 27B is with its deployed
speculative decoding, which is worth +127% at that depth; without it the same model
decodes at 42 tok/s. Speculation *loses* on the MoE — see
[why](models/qwen3.8-27b/#speculative-decoding-wins-here-and-that-is-the-point).

Both Flash-Next rates drop on longer prompts: near its 131K limit, a 119K prompt
prefills at 825 tok/s and decodes at 19.7 tok/s. The same machine will serve 262K
without the expert cache, at ~33 tok/s decode. See
[throughput vs context length](models/qwen3.8-flash-next-gsq-q2_0/#throughput-vs-context-length).

## Layout

```
models/<model>/README.md     what was configured, tested, and measured
models/<model>/*.service     the systemd unit actually used, secrets redacted
models/<model>/patches/      source patches the configuration depends on, with
                             a README explaining the series and its order
scripts/benchmark.sh         the measurement method — every entry uses it
scripts/bench/               recall and structured-output probes it calls
scripts/llm-run.sh           guarded launcher for experiments
tools/moe-skew.cpp           measure MoE expert-routing skew on your own model
```

## Conventions

- Units are redacted. Replace `__HOST_IP__` before use.
- **The API key is read from a file, not passed on the command line.** Put it in
  `~/.config/llama/api-key` at mode 600 (`llama-server --api-key-file`). A key given as
  `--api-key` ends up in `/proc/<pid>/cmdline`, which is world-readable, so every local
  user on the machine can read it out of `ps`. An environment variable is better but not
  equivalent: it still lands in the process environment and is inherited by children.
- Numbers come from [`scripts/benchmark.sh`](scripts/README.md), so entries compare
  like for like. An entry quoting throughput says which commit it was measured at.
  Numbers that predate the current method are left blank rather than quoted from
  memory — a blank cell is honest, a stale one is not.
- Paths in the units use systemd's `%h` specifier, which expands to the invoking user's
  home directory. They assume llama.cpp builds live in `~/src` and GGUFs in `~/ai-models`;
  adjust those two path fragments if yours differ.
- Every number is measured on the hardware above. Derived or vendor-reported figures are
  labelled as such.
- Configurations that were tried and rejected are kept, with the reason. They are usually
  more useful than the winning line.

## Running experiments safely

Two processes loading a multi-gigabyte model into 64 GB of RAM will invoke the kernel OOM
killer, and it does not necessarily pick the model. On this machine it once took out the
network daemons and ended the remote session.

[`scripts/llm-run.sh`](scripts/llm-run.sh) refuses to start when another llama.cpp loader
is alive, then runs the command in its own systemd scope with a hard memory cap and swap
disabled, so a runaway is reclaimed inside that cgroup:

```bash
./scripts/llm-run.sh --max 48G -- llama-server -m model.gguf ...
```

Worth pairing with `OOMScoreAdjust=-1000` drop-ins on sshd, NetworkManager, and your VPN
or mesh daemon, so that losing a benchmark never means losing remote access.

## License

MIT, see [LICENSE](LICENSE).

The patches under `models/*/patches/` are a separate matter: they are diffs against
llama.cpp, which is MIT licensed by The ggml authors, and one of them is not my work at
all — it is the expert cache from [llama.cpp PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)
by csantiago78, redistributed with its authorship intact so the configuration documented
here can be reproduced. [NOTICE](NOTICE) records who wrote what.

Model weights are not covered by any of this. They carry their own licenses.
