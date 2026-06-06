"""
test_dataset.py

Basic tests for the GameTune PyTorch dataset.
"""

from pathlib import Path
import torch
from src.dataset import DEFAULT_CONTEXT_SIZE, GameTuneDataset


TOKENIZED_ROOT = Path("data/tokenized/train")


def test_dataset_is_not_empty() -> None:
    """
    Verify the dataset contains training samples.
    """
    dataset = GameTuneDataset(TOKENIZED_ROOT)

    assert len(dataset) > 0


def test_dataset_sample_shapes() -> None:
    """
    Verify input and target tensors have the expected shape.
    """
    dataset = GameTuneDataset(TOKENIZED_ROOT)
    inputs, targets = dataset[0]

    assert inputs.shape == (DEFAULT_CONTEXT_SIZE,)
    assert targets.shape == (DEFAULT_CONTEXT_SIZE,)


def test_dataset_sample_dtypes() -> None:
    """
    Verify input and target tensors use integer token IDs.
    """
    dataset = GameTuneDataset(TOKENIZED_ROOT)
    inputs, targets = dataset[0]

    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64


def test_input_target_shift() -> None:
    """
    Verify targets are inputs shifted by one token.
    """
    dataset = GameTuneDataset(TOKENIZED_ROOT)
    inputs, targets = dataset[0]

    assert torch.equal(inputs[1:], targets[:-1])

