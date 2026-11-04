import time

import torch

from model import ModelConfig, TinyCachedDecoderLM


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate_naive(
    model: TinyCachedDecoderLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate by recomputing the full sequence at every step."""
    tokens = prompt

    for _ in range(max_new_tokens):
        logits, _ = model(tokens, use_cache=False)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


@torch.no_grad()
def generate_cached(
    model: TinyCachedDecoderLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate with one full prefill followed by one-token decode steps."""
    tokens = prompt
    input_ids = prompt
    past_key_values = None

    for _ in range(max_new_tokens):
        # TODO 1: Call the model with input_ids, past_key_values, and
        # use_cache=True.
        # TODO 2: Select the next token from the final-position logits.
        # TODO 3: Append it to tokens.
        # TODO 4: Set input_ids to only that new token for the next iteration.
        raise NotImplementedError("Implement cached generation")

    return tokens


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    config = ModelConfig()
    model = TinyCachedDecoderLM(config).to(device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=device)

    start = time.perf_counter()
    naive_output = generate_naive(model, prompt, max_new_tokens=16)
    naive_seconds = time.perf_counter() - start

    start = time.perf_counter()
    cached_output = generate_cached(model, prompt, max_new_tokens=16)
    cached_seconds = time.perf_counter() - start

    if not torch.equal(naive_output, cached_output):
        raise AssertionError("Cached and naive generation produced different tokens")

    print(f"device: {device}")
    print(f"output shape: {tuple(cached_output.shape)}")
    print(f"naive: {naive_seconds:.4f}s")
    print(f"cached: {cached_seconds:.4f}s")
    print(f"speedup: {naive_seconds / cached_seconds:.2f}x")


if __name__ == "__main__":
    main()
