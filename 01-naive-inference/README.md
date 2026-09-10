# Project 01 — Naive LLM Inference

## Goal

Build a tiny decoder-only Transformer inference path from first principles and use it to understand the cost of naive autoregressive generation.

You are **not** trying to train a competitive model. You are trying to understand what an inference engine actually does.

## Learning objectives

By the end of this project, you should be able to explain:

- token embeddings
- RMSNorm
- Q/K/V projections
- causal self-attention
- rotary positional embeddings (RoPE)
- MLP / gated feed-forward layers
- residual connections
- logits and sampling
- autoregressive decoding
- why naive decoding repeatedly recomputes previous tokens
- why inference latency grows with context length

## Architecture

```text
Tokens
  ↓
Embedding
  ↓
Transformer Block × N
  ├── RMSNorm
  ├── Q/K/V projections
  ├── RoPE
  ├── Causal Attention
  ├── Residual
  ├── RMSNorm
  ├── MLP
  └── Residual
  ↓
Final RMSNorm
  ↓
LM Head
  ↓
Logits
```

## What you will build

```text
01-naive-inference/
├── README.md
├── requirements.txt
├── implementation/
│   ├── __init__.py
│   ├── model.py
│   ├── attention.py
│   ├── layers.py
│   └── generate.py
├── benchmark.py
├── results/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
└── report.md
```

## Suggested order

1. Implement `RMSNorm`.
2. Implement causal self-attention without KV caching.
3. Implement a gated MLP.
4. Assemble one Transformer block.
5. Stack several blocks into a tiny language model.
6. Implement greedy autoregressive generation.
7. Benchmark generation at increasing context lengths.
8. Explain the scaling behavior in `report.md`.

## Important constraint

For Project 01, **do not implement a KV cache**. The inefficiency is intentional. You need to experience the baseline before Project 02 fixes it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 01-naive-inference/requirements.txt
```

On Apple Silicon, PyTorch can use MPS if available. CUDA is not required for this project.

## Run

Once the TODOs are implemented:

```bash
python 01-naive-inference/implementation/generate.py
```

Then benchmark:

```bash
python 01-naive-inference/benchmark.py
```

## Required experiments

Use several prompt/context lengths, for example:

```text
128
256
512
1024
2048
```

Keep output length fixed, for example 32 or 64 generated tokens.

Record at least:

- prompt length
- output length
- total generation time
- average time/token
- tokens/second

Optional:

- device
- peak GPU memory
- first-token latency

## Questions to answer in report.md

1. Why does naive autoregressive decoding become slower as sequence length grows?
2. Which tensor operations dominate attention?
3. What is the complexity of the attention score matrix with respect to sequence length?
4. During generation, what computations are unnecessarily repeated for old tokens?
5. What state could we preserve between decoding steps?
6. What do you predict Project 02 will improve?

## Success criteria

You have completed Project 01 when:

- [ ] the model forward pass works
- [ ] causal masking is correct
- [ ] autoregressive generation works
- [ ] no KV cache is used
- [ ] you have benchmarked at least 4 context lengths
- [ ] you produced at least one latency/throughput plot
- [ ] `report.md` explains the observed scaling behavior

## Stretch goals

- Add temperature sampling and top-k sampling.
- Compare CPU vs Apple MPS vs CUDA if available.
- Profile the model with PyTorch Profiler.
- Compare your attention output against `torch.nn.functional.scaled_dot_product_attention` for correctness.

## Next project

Project 02 will add KV caching and split your mental model into two serving phases:

```text
PREFILL → first token → DECODE → DECODE → DECODE ...
```

That distinction becomes foundational for modern LLM serving systems.
