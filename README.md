# AI Systems from Zero

A hands-on curriculum for learning modern AI systems by building, benchmarking, profiling, and reproducing ideas from real systems papers.

The guiding rule is simple:

> Build the naive version first. Measure it. Understand the bottleneck. Then study the system that solves it.

## Curriculum

| Project | Topic | Core idea | Paper / system it prepares for |
|---|---|---|---|
| 01 | Naive LLM inference | Transformer forward pass and autoregressive decoding | Transformer / inference fundamentals |
| 02 | KV cache | Prefill vs decode, reuse of past K/V states | DistServe motivation |
| 03 | Inference benchmarking | TTFT, TPOT, throughput, latency | vLLM / LLM serving literature |
| 04 | Triton kernels | GPU memory hierarchy, fusion, tiling | Triton / kernel optimization |
| 05 | Flash-style attention | IO-aware attention | FlashAttention, FlashAttention-2 |
| 06 | Continuous batching | Request scheduling and latency-throughput tradeoffs | Orca / serving schedulers |
| 07 | Paged KV cache | Memory fragmentation and block allocation | PagedAttention / vLLM (SOSP '23) |
| 08 | Quantized inference | Memory-bound inference and precision tradeoffs | llama.cpp / quantization systems |
| 09 | Chunked prefill | Prefill/decode interference | Sarathi-Serve (OSDI '24) |
| 10 | Disaggregated serving | Separate prefill and decode resources | DistServe (OSDI '24) |

## Repository structure

```text
ai-systems-from-zero/
├── 01-naive-inference/
├── 02-kv-cache/
├── 03-serving-benchmark/
├── 04-triton-kernels/
├── 05-flash-attention/
├── 06-continuous-batching/
├── 07-paged-kv-cache/
├── 08-quantization/
├── 09-chunked-prefill/
├── 10-disaggregated-serving/
├── papers/
└── README.md
```

Each project should eventually contain:

```text
README.md
implementation/
benchmark.py
results/
figures/
report.md
```

## Research workflow

For every project, answer these questions:

1. What is the problem?
2. What is the naive baseline?
3. What bottleneck do I expect?
4. What do the measurements show?
5. What optimization did I add?
6. Why did it help or fail to help?
7. What would I test next?

Treat every project as an experiment, not just a coding exercise.

## Start here

Begin with [`01-naive-inference`](./01-naive-inference/README.md).

The first project intentionally uses a tiny Transformer implementation so that every major inference component is visible and understandable.
