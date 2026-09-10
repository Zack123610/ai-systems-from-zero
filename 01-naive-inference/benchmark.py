import csv
import sys
import time
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_DIR = PROJECT_DIR / "implementation"
sys.path.insert(0, str(IMPLEMENTATION_DIR))

from generate import generate, pick_device  # noqa: E402
from model import ModelConfig, TinyDecoderLM  # noqa: E402


CONTEXT_LENGTHS = [128, 256, 512, 1024]
MAX_NEW_TOKENS = 32


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    config = ModelConfig(max_seq_len=max(CONTEXT_LENGTHS) + MAX_NEW_TOKENS)
    model = TinyDecoderLM(config).to(device).eval()

    rows = []
    print(f"Benchmarking on {device}...")

    for context_len in CONTEXT_LENGTHS:
        prompt = torch.randint(
            0,
            config.vocab_size,
            (1, context_len),
            device=device,
        )

        # Warmup.
        _ = model(prompt[:, : min(context_len, 32)])
        synchronize(device)

        start = time.perf_counter()
        _ = generate(model, prompt, MAX_NEW_TOKENS)
        synchronize(device)
        elapsed = time.perf_counter() - start

        seconds_per_token = elapsed / MAX_NEW_TOKENS
        tokens_per_second = MAX_NEW_TOKENS / elapsed

        row = {
            "device": str(device),
            "context_length": context_len,
            "output_length": MAX_NEW_TOKENS,
            "total_seconds": round(elapsed, 6),
            "seconds_per_token": round(seconds_per_token, 6),
            "tokens_per_second": round(tokens_per_second, 4),
        }
        rows.append(row)
        print(row)

    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "naive_inference.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {output_file}")
    print("Next: plot context_length vs seconds_per_token and tokens_per_second.")


if __name__ == "__main__":
    main()
