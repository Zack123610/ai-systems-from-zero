# Project 02 — KV Cache

## Goal

Build on Project 01 by caching past key/value tensors during autoregressive
decoding. Compare the cached implementation against the naive baseline and
measure the compute-memory tradeoff.

## Learning objectives

By the end of this project, you should be able to explain:

- the difference between prefill and decode
- why old key/value tensors can be reused
- the shape of a per-layer KV cache
- why cached decoding processes one new token at a time
- how cache memory scales with sequence length, layers, and model width
- why a KV cache reduces compute but increases memory usage

## Core idea

Naive generation recomputes Q, K, and V for every old token:

```text
step 1: run [prompt]
step 2: run [prompt, token 1]
step 3: run [prompt, token 1, token 2]
```

Cached generation computes the prompt once, saves its K/V tensors, and only
projects each new token:

```text
PREFILL: [prompt] → logits + cache
DECODE:  [token 1] + cache → logits + extended cache
DECODE:  [token 2] + cache → logits + extended cache
```

For each layer, use this cache layout:

```text
key:   [batch, heads, cached_tokens, head_dim]
value: [batch, heads, cached_tokens, head_dim]
```

## What you will build

```text
02-kv-cache/
├── README.md
├── requirements.txt
├── implementation/
│   ├── __init__.py
│   ├── attention.py
│   ├── layers.py
│   ├── model.py
│   └── generate.py
├── benchmark.py
├── results/
├── figures/
└── report.md
```

Project 01's completed normalization and MLP are carried forward. The guided
TODOs in this project focus on cache-aware attention, position offsets, cache
propagation through the model, and cached generation.

## Suggested order

1. Implement cache concatenation in `implementation/attention.py`.
2. Build the causal mask using absolute query and key positions.
3. Return the updated K/V tensors from attention.
4. Pass one cache entry through each Transformer block in `model.py`.
5. Offset position IDs by the number of cached tokens.
6. Implement prefill followed by one-token decode in `generate.py`.
7. Verify cached and naive greedy generation produce identical tokens.
8. Run `benchmark.py` and analyze the latency-memory tradeoff.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 02-kv-cache/requirements.txt
```

## Run

After completing the TODOs:

```bash
python 02-kv-cache/implementation/generate.py
python 02-kv-cache/benchmark.py
```

## Required experiments

Use at least four prompt lengths and a fixed output length. Record:

- method: naive or cached
- prompt and output lengths
- total generation time
- average time per generated token
- generated tokens per second
- estimated KV-cache size

Before trusting performance numbers, verify both methods produce exactly the
same greedy output from the same model and prompt.

## Cache memory

For standard multi-head attention, the approximate cache size is:

```text
2 × layers × batch × cached_tokens × model_dim × bytes_per_element
```

The factor of two accounts for keys and values.

## Questions for the report

1. Which work is performed during prefill?
2. Which tensors remain unchanged during later decode steps?
3. Why does the current query attend to all cached keys and values?
4. How does decode latency scale with context length before and after caching?
5. Why does cached decoding still become more expensive at longer contexts?
6. How much memory does the cache require per additional token?

## Success criteria

- [ ] cached and naive greedy outputs match
- [ ] prefill processes the full prompt once
- [ ] each decode step processes only the newest token
- [ ] every Transformer layer maintains its own K/V cache
- [ ] causal masking works for both prefill and decode
- [ ] at least four context lengths are benchmarked
- [ ] latency and cache-memory results are plotted
- [ ] `report.md` explains the compute-memory tradeoff

## Next project

Project 03 turns these model-level measurements into serving metrics such as
time to first token, time per output token, throughput, and request latency.
