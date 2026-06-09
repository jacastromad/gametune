"""
generate.py

Generate token sequences from a trained GameTune checkpoint
and save them as a MIDI file.
"""

from pathlib import Path
import argparse

import torch
import torch.nn.functional as F

from src.model import GameTuneModel, GameTuneModelConfig
from src.tokenizer import GameTuneTokenizer


def load_checkpoint(path: Path, device: str) -> dict:
    """
    Load a training checkpoint.
    """
    return torch.load(path, map_location=device)


def load_model(checkpoint: dict, device: str) -> GameTuneModel:
    """
    Restore the model from checkpoint data.
    """
    config = checkpoint["config"]
    model_config = GameTuneModelConfig(**config["model"])

    model = GameTuneModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> int:
    """
    Sample one token ID from model logits.
    """
    logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        cutoff = values[-1]
        logits = torch.where(
            logits < cutoff,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    probabilities = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, num_samples=1)

    return int(next_token.item())


def load_prompt_token_ids(
    prompt_midi: Path | None,
    prompt_tokens: int,
    tokenizer: GameTuneTokenizer,
) -> list[int]:
    """
    Load optional prompt tokens from a MIDI file.

    If no prompt is provided, generation starts from BOS.
    """
    if prompt_midi is None:
        return [tokenizer.token_to_id["BOS"]]

    token_ids = tokenizer.tokenize_to_ids(prompt_midi)

    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be greater than zero.")

    return token_ids[:prompt_tokens]


def generate_token_ids(
    model: GameTuneModel,
    initial_token_ids: list[int],
    tokenizer: GameTuneTokenizer,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> list[int]:
    """
    Generate token IDs autoregressively.
    """
    token_ids = initial_token_ids.copy()

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [token_ids[-model.config.context_size:]],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits, _ = model(input_ids)

        next_logits = logits[0, -1]
        next_token_id = sample_next_token(
            logits=next_logits,
            temperature=temperature,
            top_k=top_k,
        )

        token_ids.append(next_token_id)

        if next_token_id == tokenizer.token_to_id["EOS"]:
            break

    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text tokens from a trained GameTune checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gametune.pt"),
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Restrict sampling to the top K tokens.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run generation on.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/generated.mid"),
        help="Path to generated MIDI file.",
    )

    parser.add_argument(
        "--prompt-midi",
        type=Path,
        default=None,
        help="Optional MIDI file used as a generation prompt.",
    )

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=256,
        help="Number of prompt tokens to keep from --prompt-midi.",
    )

    args = parser.parse_args()

    tokenizer = GameTuneTokenizer()

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model = load_model(checkpoint, args.device)

    initial_token_ids = load_prompt_token_ids(
        prompt_midi=args.prompt_midi,
        prompt_tokens=args.prompt_tokens,
        tokenizer=tokenizer,
    )

    token_ids = generate_token_ids(
        model=model,
        initial_token_ids=initial_token_ids,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    tokens = tokenizer.decode(token_ids)
    tokenizer.tokens_to_midi(tokens, args.output)

    print(f"saved_midi: {args.output}")
    print(f"checkpoint: {args.checkpoint}")

    if args.prompt_midi is not None:
        print(f"prompt_midi: {args.prompt_midi}")
        print(f"prompt_tokens: {args.prompt_tokens}")

    print(f"generated_tokens: {len(tokens)}")
    print()

    for token in tokens:
        print(token)


if __name__ == "__main__":
    main()

