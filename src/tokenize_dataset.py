"""
tokenize_dataset.py

Tokenize MIDI files and save token ID sequences to disk.
"""

from pathlib import Path
import argparse
import torch
from src.tokenizer import GameTuneTokenizer


def output_path_for_midi(
    midi_file: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """
    Build the output path for a tokenized MIDI file.
    """
    relative_path = midi_file.relative_to(input_root)
    return output_root / relative_path.with_suffix(".pt")


def tokenize_dataset(
    input_root: Path,
    output_root: Path,
    overwrite: bool,
) -> None:
    """
    Tokenize all MIDI files under input_root and save them under output_root.
    """
    midi_files = sorted(input_root.rglob("*.mid"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under: {input_root}")

    tokenizer = GameTuneTokenizer()

    written = 0
    skipped = 0

    for index, midi_file in enumerate(midi_files, start=1):
        output_file = output_path_for_midi(
            midi_file=midi_file,
            input_root=input_root,
            output_root=output_root,
        )

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)

        token_ids = tokenizer.tokenize_to_ids(midi_file)
        tensor = torch.tensor(token_ids, dtype=torch.long)

        torch.save(tensor, output_file)

        written += 1

        if index % 100 == 0:
            print(f"processed: {index}/{len(midi_files)}")

    print()
    print(f"input_root: {input_root}")
    print(f"output_root: {output_root}")
    print(f"files_found: {len(midi_files)}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize NES-MDB MIDI files into token ID tensors."
    )

    parser.add_argument(
        "input_root",
        type=Path,
        help="Path to the root directory containing MIDI files.",
    )

    parser.add_argument(
        "output_root",
        type=Path,
        help="Path where tokenized .pt files will be written.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tokenized files.",
    )

    args = parser.parse_args()

    tokenize_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

