# Project 02 Report — KV Cache

## 1. Problem

What repeated work does naive autoregressive decoding perform?

## 2. Baseline

Describe the Project 01 model and its full-sequence generation loop.

## 3. Hypothesis

Before benchmarking, predict how a KV cache will affect:

- prefill latency
- decode latency
- generated tokens per second
- memory usage
- scaling as prompt length increases

## 4. Implementation

Document:

- the K/V tensor shape for one layer
- the dimension used to append new tokens
- how position IDs are offset during decode
- how the causal mask differs between prefill and one-token decode
- how one cache entry is passed through each Transformer layer

## 5. Correctness

Confirm that naive and cached greedy generation produce identical token IDs.
Describe any bugs found while establishing equivalence.

## 6. Experimental setup

- Hardware:
- Device backend:
- Python version:
- PyTorch version:
- Tensor dtype:
- Model dimension:
- Number of layers:
- Number of heads:
- Prompt lengths:
- Output length:
- Benchmark repetitions:

## 7. Results

Add the benchmark table and figures here.

Suggested figures:

1. Prompt length vs seconds per output token
2. Prompt length vs generated tokens per second
3. Prompt length vs estimated KV-cache memory

## 8. Analysis

Answer:

1. Why is prefill still expensive when using a KV cache?
2. What computation is avoided during decode?
3. Why must every layer have its own cache?
4. Why does cached decode latency still increase with context length?
5. Does the measured speedup match your hypothesis?
6. How many bytes does one additional cached token require?

## 9. Things that failed or surprised me

Record implementation mistakes, incorrect assumptions, and noisy measurements.

## 10. Conclusion

Summarize the compute-memory tradeoff of KV caching in 3–5 sentences.

## 11. Prediction for Project 03

Which additional metrics are needed to evaluate a multi-request inference
service rather than a single generation loop?
