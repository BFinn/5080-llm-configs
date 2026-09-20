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
| wikitext-2 perplexity @ c8192 | 6.792 |

Unit: [`qwen38-server.service`](qwen38-server.service). Replace `__API_KEY__` and
`__HOST_IP__`.

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

*Throughput numbers for this entry have not been re-measured under the current benchmark
method and are deliberately left blank in the index rather than quoted from memory.*
