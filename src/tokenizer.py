"""
tokenizer.py

Convert NES-MDB MIDI files into a simple text event stream.
This first version supports MIDI -> structured note events -> text tokens.
"""

from pathlib import Path
from dataclasses import dataclass
import argparse
import mido


CHANNEL_TRACKS = {
    "p1": "CHANNEL_P1",
    "p2": "CHANNEL_P2",
    "tr": "CHANNEL_TR",
    "no": "CHANNEL_NO",
}

TIME_STEPS_PER_BEAT = 24


@dataclass(frozen=True)
class NoteEvent:
    """
    A note extracted from one NES-MDB channel.
    """

    start_tick: int
    duration_tick: int
    channel: str
    pitch: int
    velocity: int


class GameTuneTokenizer:
    """
    Tokenizer for NES-MDB MIDI files.
    """

    def quantize_ticks(self, ticks: int, ticks_per_beat: int) -> int:
        """
        Convert raw MIDI ticks into a fixed musical grid.
        """
        ticks_per_step = ticks_per_beat / TIME_STEPS_PER_BEAT
        return round(ticks / ticks_per_step)

    def extract_track_notes(
        self,
        track: mido.MidiTrack,
        channel_token: str,
    ) -> list[NoteEvent]:
        """
        Extract note events from one MIDI track.
        """
        current_tick = 0
        active_notes: dict[int, tuple[int, int]] = {}
        notes: list[NoteEvent] = []

        for message in track:
            current_tick += message.time

            if message.type == "note_on" and message.velocity > 0:
                active_notes[message.note] = (
                    current_tick,
                    message.velocity,
                )

            elif (
                message.type == "note_off"
                or (message.type == "note_on" and message.velocity == 0)
            ):
                if message.note not in active_notes:
                    continue

                start_tick, velocity = active_notes.pop(message.note)
                duration_tick = current_tick - start_tick

                if duration_tick <= 0:
                    continue

                notes.append(
                    NoteEvent(
                        start_tick=start_tick,
                        duration_tick=duration_tick,
                        channel=channel_token,
                        pitch=message.note,
                        velocity=velocity,
                    )
                )

        return notes

    def extract_midi_notes(self, path: Path) -> tuple[list[NoteEvent], int]:
        """
        Extract all note events from the musical tracks of one MIDI file.
        """
        midi = mido.MidiFile(path)
        notes: list[NoteEvent] = []

        for track in midi.tracks:
            track_name = self._get_track_name(track)

            if track_name not in CHANNEL_TRACKS:
                continue

            notes.extend(
                self.extract_track_notes(
                    track=track,
                    channel_token=CHANNEL_TRACKS[track_name],
                )
            )

        notes.sort(
            key=lambda note: (
                note.start_tick,
                note.channel,
                note.pitch,
            )
        )

        return notes, midi.ticks_per_beat

    def tokens_from_notes(
        self,
        notes: list[NoteEvent],
        ticks_per_beat: int,
    ) -> list[str]:
        """
        Convert note events into text tokens.
        """
        tokens: list[str] = ["BOS"]
        current_step = 0

        for note in notes:
            start_step = self.quantize_ticks(
                note.start_tick,
                ticks_per_beat,
            )

            duration_step = max(
                1, self.quantize_ticks(note.duration_tick, ticks_per_beat)
            )

            time_delta = start_step - current_step

            if time_delta > 0:
                tokens.append(f"TIME_SHIFT_{time_delta}")
                current_step = start_step

            tokens.append(note.channel)
            tokens.append(f"NOTE_{note.pitch}")
            tokens.append(f"VELOCITY_{note.velocity}")
            tokens.append(f"DURATION_{duration_step}")

        tokens.append("EOS")
        return tokens

    def tokenize(self, path: Path) -> list[str]:
        """
        Convert one MIDI file into text tokens.
        """
        notes, ticks_per_beat = self.extract_midi_notes(path)
        return self.tokens_from_notes(notes, ticks_per_beat)

    def _get_track_name(self, track: mido.MidiTrack) -> str | None:
        """
        Return the MIDI track name, if present.
        """
        for message in track:
            if message.type == "track_name":
                return message.name

        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize one NES-MDB MIDI file and print text tokens."
    )

    parser.add_argument(
        "midi_file",
        type=Path,
        help="Path to a MIDI file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of tokens to print.",
    )

    args = parser.parse_args()

    if not args.midi_file.exists():
        raise FileNotFoundError(args.midi_file)

    tokenizer = GameTuneTokenizer()
    tokens = tokenizer.tokenize(args.midi_file)

    print(f"file: {args.midi_file}")
    print(f"tokens: {len(tokens)}")
    print()

    for token in tokens[: args.limit]:
        print(token)

    if len(tokens) > args.limit:
        print()
        print(f"... truncated, showing {args.limit}/{len(tokens)} tokens")


if __name__ == "__main__":
    main()

