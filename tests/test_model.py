"""
test_model.py

Basic tests for the GameTune Transformer model.
"""

import pytest
import torch

from src.model import GameTuneModel, GameTuneModelConfig


def tiny_config() -> GameTuneModelConfig:
    """
    Return a small model config for fast tests.
    """
    return GameTuneModelConfig(
        vocab_size=898,
        context_size=32,
        embedding_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )


def test_model_creates_successfully() -> None:
    """
    Verify the model can be constructed.
    """
    model = GameTuneModel(tiny_config())

    assert isinstance(model, GameTuneModel)


def test_forward_pass_output_shape() -> None:
    """
    Verify the model returns logits with the expected shape.
    """
    config = tiny_config()
    model = GameTuneModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_size),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (
        2,
        config.context_size,
        config.vocab_size,
    )
    assert loss is None


def test_forward_pass_with_targets_computes_loss() -> None:
    """
    Verify the model computes training loss when targets are provided.
    """
    config = tiny_config()
    model = GameTuneModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_size),
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_size),
    )

    logits, loss = model(input_ids, targets)

    assert logits.shape == (
        2,
        config.context_size,
        config.vocab_size,
    )
    assert loss is not None
    assert loss.ndim == 0


def test_parameter_count_is_positive() -> None:
    """
    Verify the model reports trainable parameters.
    """
    model = GameTuneModel(tiny_config())

    assert model.parameter_count() > 0


def test_invalid_attention_head_config_fails() -> None:
    """
    Verify embedding size must be divisible by number of heads.
    """
    config = GameTuneModelConfig(
        vocab_size=898,
        context_size=32,
        embedding_size=65,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        GameTuneModel(config)


def test_sequence_longer_than_context_fails() -> None:
    """
    Verify inputs cannot exceed the configured context size.
    """
    config = tiny_config()
    model = GameTuneModel(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, config.context_size + 1),
    )

    with pytest.raises(ValueError):
        model(input_ids)

