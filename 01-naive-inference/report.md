# Project 01 Report — Naive LLM Inference

## 1. Problem

What are we trying to understand about naive autoregressive Transformer inference?

## 2. Baseline

Describe the model architecture, device, PyTorch version, context lengths, and output length.

## 3. Hypothesis

Write your prediction before looking at the final results.

Example questions:

- Will time/token increase with context length?
- Will tokens/sec decrease as the sequence grows?
- Which operation do you expect to dominate?

## 4. Experimental setup

Record:

- Hardware:
- Device backend:
- Python version:
- PyTorch version:
- Model dimension:
- Number of layers:
- Number of attention heads:
- Context lengths:
- Output length:

## 5. Results

Add your table and figures here.

Suggested figures:

1. Context length vs seconds/token
2. Context length vs tokens/second

## 6. Analysis

Explain what happened and why.

Questions to answer:

1. Why does naive decoding repeatedly recompute old tokens?
2. How does the attention-score matrix grow with sequence length?
3. Which Q/K/V values from previous tokens are identical between decoding steps?
4. What state could be cached instead of recomputed?
5. Why will a KV cache improve compute efficiency but consume additional memory?

## 7. Things that failed or surprised me

Document bugs, incorrect assumptions, unexpected performance behavior, or implementation mistakes.

## 8. Conclusion

Summarize the main systems insight from this project in 3–5 sentences.

## 9. Prediction for Project 02

Before implementing KV caching, predict what will happen to:

- decode latency
- tokens/sec
- memory usage
- scaling with context length
