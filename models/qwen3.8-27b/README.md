# Qwen3.8-27B (dense) on RTX 5080 16 GB

The everyday model on this machine: fully GPU-resident, vision-capable, fast to first
token. It trades away context and raw quality for latency and multimodality.

Kept here mainly as the counterpoint to the
[Flash-Next entry](../qwen3.8-flash-next-gsq-q2_0/): same GPU, same server, opposite
design point.

| | |
|---|---|
| Quant | Unsloth UD-IQ3_S, 12 GB |
| Context | 98,304, q4_0 KV cache |
| Placement | Fully on GPU (`-ngl 99`) |
| Vision | Yes, Q8_0 projector |
| Speculative decode | draft-MTP, 3 draft tokens |
| wikitext-2 perplexity @ c8192 | **6.8412** as deployed (q4_0 KV) |

Unit: [`qwen38-server.service`](qwen38-server.service). Replace `__API_KEY__` and
`__HOST_IP__`.

### What the KV quantization costs

Measured on this model at c8192, varying only the KV cache precision:

| KV cache | Perplexity | vs f16 |
|---|---|---|
| f16 | 6.7954 ± 0.0468 | — |
| q8_0 | 6.7922 ± 0.0467 | −0.05%, i.e. free |
| **q4_0 (deployed)** | **6.8412 ± 0.0473** | +0.67% |

q8_0 KV is free at this size — it lands marginally *below* f16, which is noise, not
an improvement. q4_0 costs 0.67%, and the gap is about one standard error, so treat it
as small-but-real rather than proven. It buys the context that makes this entry useful,
so it stays.

Earlier versions of this page quoted 6.792 here. That is the q8_0 figure, and this unit
runs q4_0.

## Throughput

Measured 2026-09-21 with [`scripts/benchmark.sh`](../../scripts/README.md), phases `sn`,
at `--ctx 98304` with the deployed flags. Two arms differing only in speculative decoding.

| | Short prompt | 30K prompt, cold | 30K prompt, warm decode | VRAM |
|---|---|---|---|---|
| No speculation | 58.9 / 59.1 tok/s | 1,865 tok/s prefill, 41.9 decode | 42.1 / 42.0 | 13.93 GiB |
| **Deployed, draft-MTP n=3** | **106.8 / 109.0** | **1,756 tok/s prefill, 94.5 decode** | **95.7 / 96.0** | 15.24 GiB |

Recall at depth, 15 needles planted through an 83K-token haystack: **14/15**, and the one
miss is in the shallowest band (0-25%: 4/5; 25-50%: 2/2; 50-75%: 5/5; 75-100%: 3/3).

### Speculative decoding wins here, and that is the point

Draft-MTP is worth **+83% on a short prompt and +127% at 30K** on this model. The
[Flash-Next entry](../qwen3.8-flash-next-gsq-q2_0/) measures the same feature, on the same
GPU, with the same script, *losing* — 33.8 down to 30.1 tok/s at an almost perfect 0.98
acceptance rate.

Both results are correct, and together they are more useful than either alone. This model
is dense: verifying four drafted tokens reads the 12 GB of weights once instead of four
times, so acceptance converts directly into throughput. A sparse MoE routes each token to
its own experts, so a verification batch streams roughly the union of four tokens' experts
to produce four tokens, and there is no reuse to harvest. **Whether speculation pays is a
property of the architecture, not of the acceptance rate.**

The gain is larger at depth than on a short prompt because the baseline is slower there —
attention over 30K of KV costs per token, and speculation amortises that across every
accepted token too.

The cost is headroom: the deployed arm sits at 15,609 MiB of 16,303, leaving 694 MiB.
That is tighter than anything in the Flash-Next entry, and it includes the vision
projector. Adding to this configuration means taking something out of it.

Both units bind the same host, port, and API key with different `--alias` values, so
swapping which model is live needs no client changes:

```bash
systemctl --user disable --now flashnext-server
systemctl --user enable  --now qwen38-server
# confirm which is live:
curl -H "Authorization: Bearer $KEY" http://$HOST:8082/v1/models
```

Only one of the two can run at a time.

**Division of labour.** 27B for interactive work, vision, and agent loops. Flash-Next for
long-context and hard reasoning, where a 4-minute prefill on a 190K prompt is acceptable.

That split is now measured rather than asserted: this model decodes about twice as fast
as Flash-Next on a short prompt and carries a third of the context.
