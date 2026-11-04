import csv
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_DIR = PROJECT_DIR / "implementation"
sys.path.insert(0, str(IMPLEMENTATION_DIR))

from generate import generate_cached, generate_naive, pick_device  # noqa: E402
from model import ModelConfig, TinyCachedDecoderLM  # noqa: E402

CONTEXT_LENGTHS = [128, 256, 512, 1024]
MAX_NEW_TOKENS = 32
REPEATS = 3

GenerateFunction = Callable[
    [TinyCachedDecoderLM, torch.Tensor, int],
    torch.Tensor,
]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(
    generate_fn: GenerateFunction,
    model: TinyCachedDecoderLM,
    prompt: torch.Tensor,
    device: torch.device,
) -> float:
    durations = []

    for _ in range(REPEATS):
        synchronize(device)
        start = time.perf_counter()
        generate_fn(model, prompt, MAX_NEW_TOKENS)
        synchronize(device)
        durations.append(time.perf_counter() - start)

    return statistics.median(durations)


def estimate_cache_bytes(
    config: ModelConfig,
    batch_size: int,
    cached_tokens: int,
    bytes_per_element: int,
) -> int:
    return (
        2
        * config.num_layers
        * batch_size
        * cached_tokens
        * config.dim
        * bytes_per_element
    )


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    config = ModelConfig(max_seq_len=max(CONTEXT_LENGTHS) + MAX_NEW_TOKENS)
    model = TinyCachedDecoderLM(config).to(device).eval()

    # Correctness comes before benchmarking.
    check_prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
    naive_check = generate_naive(model, check_prompt, 4)
    cached_check = generate_cached(model, check_prompt, 4)
    if not torch.equal(naive_check, cached_check):
        raise AssertionError("Cached generation does not match the naive baseline")

    # Warm up both execution paths.
    generate_naive(model, check_prompt, 2)
    generate_cached(model, check_prompt, 2)
    synchronize(device)

    rows = []
    element_size = next(model.parameters()).element_size()
    methods = {
        "naive": generate_naive,
        "cached": generate_cached,
    }

    print(f"Benchmarking on {device}...")

    for context_len in CONTEXT_LENGTHS:
        prompt = torch.randint(
            0,
            config.vocab_size,
            (1, context_len),
            device=device,
        )

        for method_name, generate_fn in methods.items():
            elapsed = measure(generate_fn, model, prompt, device)
            cache_bytes = 0
            if method_name == "cached":
                cache_bytes = estimate_cache_bytes(
                    config=config,
                    batch_size=prompt.shape[0],
                    cached_tokens=context_len + MAX_NEW_TOKENS - 1,
                    bytes_per_element=element_size,
                )

            row = {
                "device": str(device),
                "method": method_name,
                "context_length": context_len,
                "output_length": MAX_NEW_TOKENS,
                "median_seconds": round(elapsed, 6),
                "seconds_per_token": round(elapsed / MAX_NEW_TOKENS, 6),
                "tokens_per_second": round(MAX_NEW_TOKENS / elapsed, 4),
                "estimated_cache_bytes": cache_bytes,
            }
            rows.append(row)
            print(row)

    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "kv_cache_benchmark.csv"

    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
