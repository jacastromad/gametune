"""
test_inspect_midi.py

Basic tests for MIDI inspection utilities.
"""

from pathlib import Path

from src.inspect_midi import analyze_midi


TEST_MIDI = Path(
    "data/raw/nesmdb_midi/train/000_10_YardFight_00_01GameStart.mid"
)


def test_midi_file_exists() -> None:
    """
    Verify the test MIDI file is available.
    """
    assert TEST_MIDI.exists()


def test_analyze_midi_returns_valid_stats() -> None:
    """
    Verify a MIDI file can be analyzed successfully.
    """
    stats = analyze_midi(TEST_MIDI)

    assert stats.path == TEST_MIDI

    assert stats.midi_type >= 0
    assert stats.ticks_per_beat > 0

    assert stats.length_seconds > 0

    assert stats.track_count > 0
    assert len(stats.tracks) == stats.track_count

    assert stats.total_note_count >= 0


def test_tracks_have_valid_statistics() -> None:
    """
    Verify track statistics are internally consistent.
    """
    stats = analyze_midi(TEST_MIDI)

    for track in stats.tracks:
        assert track.message_count >= 0
        assert track.note_count >= 0

        if track.note_count > 0:
            assert track.min_pitch is not None
            assert track.max_pitch is not None

            assert track.min_velocity is not None
            assert track.max_velocity is not None

            assert track.first_note_time_ticks is not None
            assert track.last_note_time_ticks is not None

            assert track.min_pitch <= track.max_pitch
            assert track.min_velocity <= track.max_velocity
            assert (
                track.first_note_time_ticks
                <= track.last_note_time_ticks
            )

