"""
inspect_midi.py

Utilities for loading and inspecting MIDI files.
"""

from pathlib import Path
from dataclasses import dataclass
import mido


@dataclass
class TrackStats:
    name: str | None
    message_count: int
    note_count: int

    min_pitch: int | None
    max_pitch: int | None

    min_velocity: int | None
    max_velocity: int | None

    first_note_time_ticks: int | None
    last_note_time_ticks: int | None

    tempos_bpm: list[float]


@dataclass
class MidiStats:
    path: Path

    midi_type: int
    ticks_per_beat: int
    length_seconds: float

    track_count: int
    total_note_count: int

    tracks: list[TrackStats]


def load_midi(path: Path) -> mido.MidiFile:
    """
    Load a MIDI file.
    """
    return mido.MidiFile(path)


def analyze_track(track: mido.MidiTrack) -> TrackStats:
    """
    Extract statistics from a single MIDI track.
    """
    current_time_ticks = 0

    track_name = None

    note_count = 0

    pitches = []
    velocities = []
    note_times = []

    tempos_bpm = []

    for message in track:
        current_time_ticks += message.time

        if message.type == "track_name":
            track_name = message.name

        elif message.type == "set_tempo":
            tempos_bpm.append(
                round(mido.tempo2bpm(message.tempo), 2)
            )

        elif message.type == "note_on" and message.velocity > 0:
            note_count += 1

            pitches.append(message.note)
            velocities.append(message.velocity)
            note_times.append(current_time_ticks)

    return TrackStats(
        name=track_name,
        message_count=len(track),
        note_count=note_count,

        min_pitch=min(pitches) if pitches else None,
        max_pitch=max(pitches) if pitches else None,

        min_velocity=min(velocities) if velocities else None,
        max_velocity=max(velocities) if velocities else None,

        first_note_time_ticks=min(note_times) if note_times else None,
        last_note_time_ticks=max(note_times) if note_times else None,

        tempos_bpm=tempos_bpm,
    )


def analyze_midi(path: Path) -> MidiStats:
    """
    Analyze a MIDI file and return its statistics.
    """
    midi = load_midi(path)

    tracks = [
        analyze_track(track)
        for track in midi.tracks
    ]

    return MidiStats(
        path=path,

        midi_type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        length_seconds=midi.length,

        track_count=len(midi.tracks),

        total_note_count=sum(
            track.note_count
            for track in tracks
        ),

        tracks=tracks,
    )


def print_midi_stats(stats: MidiStats) -> None:
    """
    Pretty-print MIDI statistics.
    """
    print(f"file: {stats.path}")
    print(f"type: {stats.midi_type}")
    print(f"ticks_per_beat: {stats.ticks_per_beat}")
    print(f"length_seconds: {stats.length_seconds:.2f}")
    print(f"tracks: {stats.track_count}")
    print()

    for index, track in enumerate(stats.tracks):
        print(f"track {index}")
        print(f"  name: {track.name}")
        print(f"  messages: {track.message_count}")
        print(f"  note_on_events: {track.note_count}")
        print(f"  tempos_bpm: {track.tempos_bpm}")
        print(f"  min_pitch: {track.min_pitch}")
        print(f"  max_pitch: {track.max_pitch}")
        print(f"  min_velocity: {track.min_velocity}")
        print(f"  max_velocity: {track.max_velocity}")
        print(f"  first_note_time_ticks: {track.first_note_time_ticks}")
        print(f"  last_note_time_ticks: {track.last_note_time_ticks}")
        print()

    print(f"total_note_on_events: {stats.total_note_count}")


