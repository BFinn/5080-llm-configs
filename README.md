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
| Qwen3.8-Flash-Next (512x56B MoE) | GSQ-RCO Q2_0 + LRU expert cache | 66.4 GB | 131,072 | 1,101 tok/s | 50-54 tok/s | [entry](models/qwen3.8-flash-next-gsq-q2_0/) |
| Qwen3.8-27B (dense, vision) | UD-IQ3_S | 12 GB | 98,304 | — | — | [entry](models/qwen3.8-27b/) |

Prefill figures are for a 30K-token prompt, warm. Both rates drop on longer prompts: at
187K it is 728 tok/s prefill and 16.9 tok/s decode. See
[throughput vs context length](models/qwen3.8-flash-next-gsq-q2_0/#throughput-vs-context-length).

## Layout

```
models/<model>/README.md     what was configured, tested, and measured
models/<model>/*.service     the systemd unit actually used, secrets redacted
models/<model>/patches/      any source patches the configuration depends on
scripts/benchmark.sh         the measurement method — every entry uses it
scripts/bench/               recall and structured-output probes it calls
scripts/llm-run.sh           guarded launcher for experiments
tools/moe-skew.cpp           measure MoE expert-routing skew on your own model
```

## Conventions

- Units are redacted. Replace `__API_KEY__` and `__HOST_IP__` before use.
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
