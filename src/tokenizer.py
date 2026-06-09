"""
tokenizer.py

Convert MIDI files into GameTune tokens.
Supports MIDI -> structured note events -> text tokens -> token IDs.
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

MAX_TIME_SHIFT = 384
MAX_DURATION = 384

SPECIAL_TOKENS = [
    "PAD",
    "BOS",
    "EOS",
]

CHANNEL_TOKENS = [
    "CHANNEL_P1",
    "CHANNEL_P2",
    "CHANNEL_TR",
    "CHANNEL_NO",
]


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

    def __init__(self) -> None:
        self.token_to_id = self._build_token_to_id()
        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

    def _build_token_to_id(self) -> dict[str, int]:
        tokens: list[str] = []

        tokens.extend(SPECIAL_TOKENS)
        tokens.extend(CHANNEL_TOKENS)

        tokens.extend(
            f"NOTE_{pitch}"
            for pitch in range(1, 109)
        )

        tokens.extend(
            f"VELOCITY_{velocity}"
            for velocity in range(1, 16)
        )

        tokens.extend(
            f"TIME_SHIFT_{step}"
            for step in range(1, MAX_TIME_SHIFT + 1)
        )

        tokens.extend(
            f"DURATION_{step}"
            for step in range(1, MAX_DURATION + 1)
        )

        return {
            token: token_id
            for token_id, token in enumerate(tokens)
        }

    def encode(self, tokens: list[str]) -> list[int]:
        return [
            self.token_to_id[token]
            for token in tokens
        ]

    def decode(self, token_ids: list[int]) -> list[str]:
        return [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

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
        active_notes: dict[int, list[tuple[int, int]]] = {}
        notes: list[NoteEvent] = []
    
        for message in track:
            current_tick += message.time
    
            if message.type == "note_on" and message.velocity > 0:
                active_notes.setdefault(message.note, []).append(
                    (
                        current_tick,
                        message.velocity,
                    )
                )
    
            elif (
                message.type == "note_off"
                or (message.type == "note_on" and message.velocity == 0)
            ):
                if message.note not in active_notes:
                    continue
    
                if not active_notes[message.note]:
                    continue
    
                start_tick, velocity = active_notes[message.note].pop(0)
    
                if not active_notes[message.note]:
                    del active_notes[message.note]
    
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

            duration_step = min(
                MAX_DURATION,
                max(
                    1,
                    self.quantize_ticks(note.duration_tick, ticks_per_beat),
                ),
            )

            time_delta = start_step - current_step

            while time_delta > 0:
                shift = min(time_delta, MAX_TIME_SHIFT)
                tokens.append(f"TIME_SHIFT_{shift}")
                time_delta -= shift

            current_step = start_step

            tokens.append(note.channel)
            tokens.append(f"NOTE_{note.pitch}")
            tokens.append(f"VELOCITY_{note.velocity}")
            tokens.append(f"DURATION_{duration_step}")

        tokens.append("EOS")
        return tokens

    def tokenize_to_ids(self, path: Path) -> list[int]:
        """
        Convert one MIDI file into integer token IDs.
        """
        tokens = self.tokenize(path)
        return self.encode(tokens)

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

    def tokens_to_midi(
        self,
        tokens: list[str],
        output_path: Path,
        ticks_per_beat: int = 22050,
        tempo_bpm: int = 120,
    ) -> None:
        """
        Convert text tokens back into a MIDI file.
        """
        ticks_per_step = ticks_per_beat / TIME_STEPS_PER_BEAT
    
        midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    
        tempo_track = mido.MidiTrack()
        tempo_track.append(
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(tempo_bpm),
                time=0,
            )
        )
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(tempo_track)
    
        channel_to_track_name = {
            "CHANNEL_P1": "p1",
            "CHANNEL_P2": "p2",
            "CHANNEL_TR": "tr",
            "CHANNEL_NO": "no",
        }
    
        channel_to_midi_channel = {
            "CHANNEL_P1": 0,
            "CHANNEL_P2": 1,
            "CHANNEL_TR": 2,
            "CHANNEL_NO": 9,
        }
    
        tracks: dict[str, list[tuple[int, str, int, int]]] = {}
    
        for channel_token, track_name in channel_to_track_name.items():
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=track_name, time=0))
            tracks[channel_token] = []
            midi.tracks.append(track)
    
        current_step = 0
        index = 0
    
        while index < len(tokens):
            token = tokens[index]
    
            if token in {"BOS", "EOS", "PAD"}:
                index += 1
                continue
    
            if token.startswith("TIME_SHIFT_"):
                current_step += int(token.removeprefix("TIME_SHIFT_"))
                index += 1
                continue
    
            if token.startswith("CHANNEL_"):
                if index + 3 >= len(tokens):
                    break
    
                channel = token
                note_token = tokens[index + 1]
                velocity_token = tokens[index + 2]
                duration_token = tokens[index + 3]
    
                if (
                    not note_token.startswith("NOTE_")
                    or not velocity_token.startswith("VELOCITY_")
                    or not duration_token.startswith("DURATION_")
                ):
                    index += 1
                    continue
    
                if channel not in tracks:
                    index += 4
                    continue
    
                pitch = int(note_token.removeprefix("NOTE_"))
                velocity = int(velocity_token.removeprefix("VELOCITY_"))
                duration = int(duration_token.removeprefix("DURATION_"))
    
                # NES velocities are 1-15.
                # Scale them to a more usable General MIDI range.
                velocity = min(127, velocity * 8)
    
                start_tick = round(current_step * ticks_per_step)
                duration_tick = max(1, round(duration * ticks_per_step))
                end_tick = start_tick + duration_tick
    
                tracks[channel].append((start_tick, "note_on", pitch, velocity))
                tracks[channel].append((end_tick, "note_off", pitch, 0))
    
                index += 4
                continue
    
            index += 1
    
        for midi_track, channel_token in zip(
            midi.tracks[1:],
            channel_to_track_name,
        ):
            events = sorted(
                tracks[channel_token],
                key=lambda event: event[0],
            )
    
            previous_tick = 0
            midi_channel = channel_to_midi_channel[channel_token]
    
            for tick, message_type, pitch, velocity in events:
                delta = max(0, tick - previous_tick)
                previous_tick = tick
    
                midi_track.append(
                    mido.Message(
                        message_type,
                        note=pitch,
                        velocity=velocity,
                        time=delta,
                        channel=midi_channel,
                    )
                )
    
            midi_track.append(mido.MetaMessage("end_of_track", time=0))
    
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.save(output_path)


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

