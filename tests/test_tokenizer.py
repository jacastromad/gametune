"""
test_tokenizer.py

Basic tests for the GameTune tokenizer.
"""

from pathlib import Path
from src.tokenizer import GameTuneTokenizer


TEST_MIDI = Path(
    "data/raw/nesmdb_midi/train/045_Castlevania_01_02VampireKiller.mid"
)


def test_tokenizer_outputs_tokens() -> None:
    """
    Verify that tokenization produces a non-empty token list.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    assert len(tokens) > 0


def test_tokens_start_and_end_correctly() -> None:
    """
    Verify that token sequences use BOS and EOS markers.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    assert tokens[0] == "BOS"
    assert tokens[-1] == "EOS"


def test_tokens_contain_expected_event_types() -> None:
    """
    Verify that the token stream contains the expected music events.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    assert any(token.startswith("CHANNEL_") for token in tokens)
    assert any(token.startswith("NOTE_") for token in tokens)
    assert any(token.startswith("VELOCITY_") for token in tokens)
    assert any(token.startswith("DURATION_") for token in tokens)


def test_no_zero_time_shifts() -> None:
    """
    Verify that TIME_SHIFT_0 is never emitted.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    assert "TIME_SHIFT_0" not in tokens


def test_durations_are_positive() -> None:
    """
    Verify that all duration tokens are positive.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    durations = [
        int(token.removeprefix("DURATION_"))
        for token in tokens
        if token.startswith("DURATION_")
    ]

    assert len(durations) > 0
    assert all(duration >= 1 for duration in durations)
