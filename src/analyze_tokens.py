"""
analyze_tokens.py

Analyze tokenized sequence lengths across the dataset.
"""

from pathlib import Path
import argparse
from statistics import mean, median
from src.tokenizer import GameTuneTokenizer


def percentile(values: list[int], percent: float) -> int:
    """
    Return the nearest-rank percentile value.
    """
    if not values:
        raise ValueError("Cannot calculate percentile of empty list.")

    index = round((percent / 100) * (len(values) - 1))
    return values[index]


def analyze_token_lengths(root: Path) -> None:
    """
    Tokenize every MIDI file and summarize token sequence lengths.
    """
    midi_files = sorted(root.rglob("*.mid"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under: {root}")

    tokenizer = GameTuneTokenizer()
    token_lengths: list[int] = []
    file_lengths: list[tuple[Path, int]] = []

    for index, midi_file in enumerate(midi_files, start=1):
        tokens = tokenizer.tokenize(midi_file)
        token_count = len(tokens)

        token_lengths.append(token_count)
        file_lengths.append((midi_file, token_count))

        if index % 100 == 0:
            print(f"processed: {index}/{len(midi_files)}")

    sorted_lengths = sorted(token_lengths)
    longest_files = sorted(
        file_lengths,
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    print()
    print(f"dataset_root: {root}")
    print(f"files: {len(midi_files)}")
    print()

    print("token_count:")
    print(f"  min: {min(sorted_lengths)}")
    print(f"  mean: {mean(sorted_lengths):.2f}")
    print(f"  median: {median(sorted_lengths):.2f}")
    print(f"  p90: {percentile(sorted_lengths, 90)}")
    print(f"  p95: {percentile(sorted_lengths, 95)}")
    print(f"  p99: {percentile(sorted_lengths, 99)}")
    print(f"  max: {max(sorted_lengths)}")
    print()

    print("longest_files:")
    for path, token_count in longest_files:
        print(f"  {token_count}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze token sequence lengths for NES-MDB MIDI files."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the root directory containing MIDI files.",
    )

    args = parser.parse_args()
    analyze_token_lengths(args.dataset_root)


if __name__ == "__main__":
    main()

