"""
dataset.py

PyTorch dataset for loading tokenized GameTune sequences.
"""

from pathlib import Path
import argparse
import torch
from torch.utils.data import Dataset


DEFAULT_CONTEXT_SIZE = 2048


class GameTuneDataset(Dataset):
    """
    Dataset that turns tokenized songs into fixed-length training samples.
    """

    def __init__(
        self,
        tokenized_root: Path,
        context_size: int = DEFAULT_CONTEXT_SIZE,
        stride: int | None = None,
    ) -> None:
        self.tokenized_root = tokenized_root
        self.context_size = context_size
        self.stride = stride or context_size

        self.samples: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """
        Build an index of file paths and start positions.
        """
        token_files = sorted(self.tokenized_root.rglob("*.pt"))

        if not token_files:
            raise FileNotFoundError(
                f"No tokenized files found under: {self.tokenized_root}"
            )

        for token_file in token_files:
            tokens = torch.load(token_file, map_location="cpu")
            length = len(tokens)

            if length < 2:
                continue

            if length <= self.context_size + 1:
                self.samples.append((token_file, 0))
                continue

            max_start = length - self.context_size - 1

            for start in range(0, max_start + 1, self.stride):
                self.samples.append((token_file, start))

    def __len__(self) -> int:
        """
        Return the number of training samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one input/target pair.

        input:  tokens[0:n]
        target: tokens[1:n+1]
        """
        token_file, start = self.samples[index]
        tokens = torch.load(token_file, map_location="cpu")

        end = start + self.context_size + 1
        chunk = tokens[start:end]

        if len(chunk) < self.context_size + 1:
            padding = torch.zeros(
                self.context_size + 1 - len(chunk),
                dtype=torch.long,
            )
            chunk = torch.cat([chunk, padding])

        inputs = chunk[:-1]
        targets = chunk[1:]

        return inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the GameTune PyTorch dataset."
    )

    parser.add_argument(
        "tokenized_root",
        type=Path,
        help="Path to tokenized .pt files.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help="Number of tokens used as model context.",
    )

    args = parser.parse_args()

    dataset = GameTuneDataset(
        tokenized_root=args.tokenized_root,
        context_size=args.context_size,
    )

    inputs, targets = dataset[0]

    print(f"tokenized_root: {args.tokenized_root}")
    print(f"samples: {len(dataset)}")
    print(f"input_shape: {tuple(inputs.shape)}")
    print(f"target_shape: {tuple(targets.shape)}")
    print(f"input_dtype: {inputs.dtype}")
    print(f"target_dtype: {targets.dtype}")


if __name__ == "__main__":
    main()

