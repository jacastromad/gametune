"""
analyze_dataset.py

Analyze all MIDI files and summarize dataset-wide statistics.
"""

from pathlib import Path
import argparse
from collections import Counter, defaultdict
from statistics import mean

from src.inspect_midi import analyze_midi


def update_min_max(
    values: dict[str, dict[str, int | None]],
    channel: str,
    min_value: int | None,
    max_value: int | None,
) -> None:
    """
    Update min/max statistics for one channel.
    """
    if min_value is None or max_value is None:
        return

    current = values[channel]

    if current["min"] is None or min_value < current["min"]:
        current["min"] = min_value

    if current["max"] is None or max_value > current["max"]:
        current["max"] = max_value


def print_min_max(
    title: str,
    values: dict[str, dict[str, int | None]],
) -> None:
    """
    Print min/max statistics grouped by channel.
    """
    print(title)

    for channel in sorted(values):
        channel_values = values[channel]
        print(
            f"  {channel}: "
            f"{channel_values['min']} - {channel_values['max']}"
        )

    print()


def analyze_dataset(root: Path) -> None:
    """
    Analyze every MIDI file under a dataset root directory.
    """
    midi_files = sorted(root.rglob("*.mid"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under: {root}")

    midi_types = Counter()
    ticks_per_beat_values = Counter()
    track_counts = Counter()
    track_names = Counter()
    tempos_bpm = Counter()

    durations_seconds = []
    note_counts = []

    pitch_ranges = defaultdict(lambda: {"min": None, "max": None})
    velocity_ranges = defaultdict(lambda: {"min": None, "max": None})

    for midi_file in midi_files:
        stats = analyze_midi(midi_file)

        midi_types[stats.midi_type] += 1
        ticks_per_beat_values[stats.ticks_per_beat] += 1
        track_counts[stats.track_count] += 1

        durations_seconds.append(stats.length_seconds)
        note_counts.append(stats.total_note_count)

        for track in stats.tracks:
            if track.name is not None:
                track_names[track.name] += 1

            for tempo in track.tempos_bpm:
                tempos_bpm[tempo] += 1

            channel = track.name or "metadata"

            update_min_max(
                pitch_ranges,
                channel,
                track.min_pitch,
                track.max_pitch,
            )

            update_min_max(
                velocity_ranges,
                channel,
                track.min_velocity,
                track.max_velocity,
            )

    print(f"dataset_root: {root}")
    print(f"files: {len(midi_files)}")
    print()

    print("midi_types:")
    for midi_type, count in sorted(midi_types.items()):
        print(f"  {midi_type}: {count}")
    print()

    print("ticks_per_beat:")
    for ticks_per_beat, count in sorted(ticks_per_beat_values.items()):
        print(f"  {ticks_per_beat}: {count}")
    print()

    print("track_counts:")
    for track_count, count in sorted(track_counts.items()):
        print(f"  {track_count}: {count}")
    print()

    print("track_names:")
    for name, count in sorted(track_names.items()):
        print(f"  {name}: {count}")
    print()

    print("tempos_bpm:")
    for tempo, count in sorted(tempos_bpm.items()):
        print(f"  {tempo}: {count}")
    print()

    print_min_max("pitch_ranges:", pitch_ranges)
    print_min_max("velocity_ranges:", velocity_ranges)

    print("duration_seconds:")
    print(f"  min: {min(durations_seconds):.2f}")
    print(f"  mean: {mean(durations_seconds):.2f}")
    print(f"  max: {max(durations_seconds):.2f}")
    print()

    print("note_counts:")
    print(f"  min: {min(note_counts)}")
    print(f"  mean: {mean(note_counts):.2f}")
    print(f"  max: {max(note_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze all MIDI files in the NES-MDB dataset."
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the root directory containing MIDI files.",
    )

    args = parser.parse_args()
    analyze_dataset(args.dataset_root)


if __name__ == "__main__":
    main()

