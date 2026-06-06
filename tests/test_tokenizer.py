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


def test_vocabulary_size() -> None:
    """
    Verify the expected vocabulary size.
    """
    tokenizer = GameTuneTokenizer()

    assert tokenizer.vocab_size == 898


def test_special_token_ids_are_stable() -> None:
    """
    Verify fixed IDs for special tokens.
    """
    tokenizer = GameTuneTokenizer()

    assert tokenizer.token_to_id["PAD"] == 0
    assert tokenizer.token_to_id["BOS"] == 1
    assert tokenizer.token_to_id["EOS"] == 2


def test_encode_decode_round_trip() -> None:
    """
    Verify tokens can be encoded to IDs and decoded back unchanged.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    token_ids = tokenizer.encode(tokens)
    decoded_tokens = tokenizer.decode(token_ids)

    assert decoded_tokens == tokens


def test_tokenize_to_ids_outputs_integers() -> None:
    """
    Verify tokenization can directly produce integer token IDs.
    """
    tokenizer = GameTuneTokenizer()
    token_ids = tokenizer.tokenize_to_ids(TEST_MIDI)

    assert len(token_ids) > 0
    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_all_generated_tokens_exist_in_vocabulary() -> None:
    """
    Verify every generated token is part of the vocabulary.
    """
    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(TEST_MIDI)

    missing_tokens = [
        token
        for token in tokens
        if token not in tokenizer.token_to_id
    ]

    assert missing_tokens == []


