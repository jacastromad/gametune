# GameTune

Train a small Transformer from scratch on NES-style video game music.

## Environment

* Docker
* PyTorch
* NVIDIA RTX 3090 (24 GB VRAM)

## Project Structure

```text
.
├── checkpoints/
├── data/
├── src/
├── tests/
├── config.yml
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Dataset

Dataset: NES-MDB (https://github.com/chrisdonahue/nesmdb)

Downloaded:

```text
data/raw/nesmdb_midi/
├── train/
└── test/
```

The dataset contains MIDI representations of NES game music.

### Dataset Observations

Files analyzed: 5278

Structure:

* MIDI type: 1
* Tracks per file: 5
* Track 0: metadata / tempo
* Track 1: p1
* Track 2: p2
* Track 3: tr
* Track 4: no

Timing:

* Tempo: 120 BPM
* Ticks per beat: 22050

Pitch ranges:

| Channel |  Range  |
|---------|---------|
| p1      | 33-108  |
| p2      | 33-108  |
| tr      | 21-108  |
| no      | 1-16    |

Velocity ranges:

| Channel | Range |
|---------|-------|
| p1      | 1-15  |
| p2      | 1-15  |
| tr      | 1     |
| no      | 1-15  |

Song statistics:

|     Metric      |   Value   |
|-----------------|-----------|
| Min duration    | 0.09 s    |
| Mean duration   | 31.45 s   |
| Max duration    | 1517.68 s |
| Min note count  | 0         |
| Mean note count | 647.30    |
| Max note count  | 41839     |

## Tokenizer Design

### Vocabulary

Channels:

```text
CHANNEL_P1
CHANNEL_P2
CHANNEL_TR
CHANNEL_NO
```

Special tokens:

```text
PAD
BOS
EOS
```

### Representation

Notes:

```text
CHANNEL_P1
NOTE_64
DURATION_12
```

Time:

```text
TIME_SHIFT_n
```

Velocity:

```text
VELOCITY_n
```

Example:

```text
TIME_SHIFT_4
CHANNEL_P1
NOTE_64
VELOCITY_2
DURATION_12

CHANNEL_TR
NOTE_48
VELOCITY_1
DURATION_12

TIME_SHIFT_12

CHANNEL_NO
NOTE_13
VELOCITY_3
DURATION_2
```

### Decisions

* Channels: Keep as explicit tokens
* Notes: NOTE + DURATION
* Time: TIME_SHIFT tokens
* Velocity: Keep (1-15)
* Special tokens: PAD, BOS, EOS
* Track handling: All channels merged into one event stream
* Tempo: Ignore
* Metadata track: Ignore
* Timing: Quantized musical grid
* Training samples: One song per sequence

## Token Statistics

Tokenized files analyzed: 5278

|    Metric     |  Value   |
|---------------|----------|
| Min tokens    | 2        |
| Mean tokens   | 2912.10  |
| Median tokens | 1390.50  |
| P90           | 7034     |
| P95           | 9883     |
| P99           | 20843    |
| Max tokens    | 190170   |

### Training Decisions

* Initial context size: 2048 tokens
* Longer songs will be split into multiple training sequences
* Extremely long songs are chunked during training

## Data Pipeline

Analyze dataset:

```bash
docker compose run --rm gametune \
    python src/analyze_dataset.py data/raw/nesmdb_midi
```

Analyze token lengths:

```bash
docker compose run --rm gametune \
    python src/analyze_tokens.py data/raw/nesmdb_midi
```

Tokenize dataset:

```bash
docker compose run --rm gametune \
    python src/tokenize_dataset.py \
    data/raw/nesmdb_midi \
    data/tokenized
```

Inspect dataset:

```bash
docker compose run --rm gametune \
    python src/dataset.py data/tokenized/train
```

## Train

Train the model:

```bash
docker compose run --rm gametune \
    python src/train.py --config config.yml
```

Resume training:

```bash
docker compose run --rm gametune \
    python src/train.py --config config.yml \
    --resume checkpoints/gametune_checkpoint.pt
```

## Generate

Generate token sequences and MIDI:

```bash
docker compose run --rm gametune python src/generate.py \
    --checkpoint checkpoints/gametune_checkpoint.pt \
    --prompt-midi data/raw/nesmdb_midi/test/midi_file.mid \
    --prompt-tokens 256 \
    --max-new-tokens 2000 \
    --temperature 1.0 \
    --top-k 50 \
    --output outputs/generated.mid
```

