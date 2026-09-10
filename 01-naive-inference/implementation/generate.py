import time

import torch

from model import ModelConfig, TinyDecoderLM


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate(model: TinyDecoderLM, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Naive autoregressive decoding.

    IMPORTANT: each iteration feeds the entire sequence back through the model.
    Project 02 will remove this repeated work with a KV cache.
    """
    tokens = prompt

    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()

    config = ModelConfig()
    model = TinyDecoderLM(config).to(device).eval()

    # Random token IDs are intentional. Project 01 studies inference mechanics,
    # not language quality.
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=device)

    start = time.perf_counter()
    output = generate(model, prompt, max_new_tokens=16)
    elapsed = time.perf_counter() - start

    print(f"device: {device}")
    print(f"input shape: {tuple(prompt.shape)}")
    print(f"output shape: {tuple(output.shape)}")
    print(f"elapsed: {elapsed:.4f}s")
    print(f"generated token ids: {output[0, -16:].tolist()}")


if __name__ == "__main__":
    main()
