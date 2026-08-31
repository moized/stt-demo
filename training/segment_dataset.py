from pathlib import Path
import re
import csv
import shutil

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_DIR = PROJECT_ROOT / "training" / "manifests"

INPUT_FILES = [
    MANIFEST_DIR / "train.csv",
    MANIFEST_DIR / "validation.csv",
    MANIFEST_DIR / "test.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "training" / "segments"
AUDIO_DIR = OUTPUT_DIR / "audio"

SEGMENT_SECONDS = 30

TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.*)$"
)


def parse_timestamp(value):
    minutes = int(value.group(1))
    seconds = int(value.group(2))

    if value.group(3) is not None:
        hours = minutes
        minutes = seconds
        seconds = int(value.group(3))
        return hours * 3600 + minutes * 60 + seconds

    return minutes * 60 + seconds


def parse_transcript(text):
    """
    Transcript'i timestamp + text çiftlerine ayırır.
    Metadata satırlarını atlar.
    """

    entries = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("# taslak:"):
            continue

        if line.startswith("# duzelt:"):
            continue

        if line.startswith("Model:"):
            continue

        if line.startswith("Prompt-SHA256:"):
            continue

        if line == "--- TRANSKRIPT ---":
            continue

        match = TIMESTAMP_PATTERN.match(line)

        if not match:
            continue

        timestamp = parse_timestamp(match)

        content = match.group(4).strip()

        speaker_match = re.match(
            r"Konuşmacı\s+\d+\s*:\s*(.*)",
            content,
            flags=re.IGNORECASE,
        )

        if speaker_match:
            content = speaker_match.group(1).strip()

        if not content:
            continue

        if content.lower() in {
            "[sessizlik]",
            "[anlaşılmıyor]",
        }:
            continue

        content = re.sub(
            r"\[anlaşılmıyor\]",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        if content:
            entries.append((timestamp, content))

    return entries


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def process_record(row, split):

    audio_path = PROJECT_ROOT / Path(
        row["audio"].replace("\\", "/")
    )

    transcript_path = PROJECT_ROOT / Path(
        row["transcript"].replace("\\", "/")
    )

    if not audio_path.exists():
        print(f"[UYARI] Audio yok: {audio_path}")
        return []

    if not transcript_path.exists():
        print(f"[UYARI] Transcript yok: {transcript_path}")
        return []

    transcript = transcript_path.read_text(
        encoding="utf-8"
    )

    entries = parse_transcript(transcript)

    if not entries:
        print(
            f"[UYARI] Transcript parse edilemedi: "
            f"{transcript_path}"
        )
        return []

    audio, sample_rate = sf.read(audio_path)

    total_samples = len(audio)
    total_duration = total_samples / sample_rate

    segments = []

    current_start = None
    current_texts = []
    segment_index = 0

    for timestamp, text in entries:

        # İlk konuşma
        if current_start is None:
            current_start = timestamp
            current_texts = [text]
            continue

        # Bu konuşmayı eklersek 30 saniyeyi aşacak mı?
        current_duration = timestamp - current_start

        if current_duration > SEGMENT_SECONDS:

            # Önce mevcut segmenti kaydet
            end_sec = min(
                timestamp,
                total_duration
            )

            start_sample = int(
                current_start * sample_rate
            )

            end_sample = int(
                end_sec * sample_rate
            )

            segment_audio = audio[
                start_sample:end_sample
            ]

            segment_text = " ".join(
                current_texts
            ).strip()

            if (
                len(segment_audio) > 0
                and segment_text
            ):

                segment_index += 1

                record_name = (
                    f"{row['id']}_part_{segment_index:03d}"
                )

                split_audio_dir = (
                    AUDIO_DIR / split
                )

                split_audio_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_audio = (
                    split_audio_dir
                    / f"{record_name}.wav"
                )

                sf.write(
                    output_audio,
                    segment_audio,
                    sample_rate
                )

                segments.append({
                    "id": record_name,
                    "original_id": row["id"],
                    "split": split,
                    "audio": str(
                        output_audio.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "text": segment_text,
                    "start": current_start,
                    "end": end_sec,
                })

            # Yeni segmenti bu utterance ile başlat
            current_start = timestamp
            current_texts = [text]

        else:
            current_texts.append(text)

    # Son segment
    if current_start is not None and current_texts:

        end_sec = total_duration

        start_sample = int(
            current_start * sample_rate
        )

        end_sample = int(
            end_sec * sample_rate
        )

        segment_audio = audio[
            start_sample:end_sample
        ]

        segment_text = " ".join(
            current_texts
        ).strip()

        if (
            len(segment_audio) > 0
            and segment_text
        ):

            segment_index += 1

            record_name = (
                f"{row['id']}_part_{segment_index:03d}"
            )

            split_audio_dir = (
                AUDIO_DIR / split
            )

            split_audio_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_audio = (
                split_audio_dir
                / f"{record_name}.wav"
            )

            sf.write(
                output_audio,
                segment_audio,
                sample_rate
            )

            segments.append({
                "id": record_name,
                "original_id": row["id"],
                "split": split,
                "audio": str(
                    output_audio.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "text": segment_text,
                "start": current_start,
                "end": end_sec,
            })

    return segments

def main():

    print("=" * 80)
    print("WHISPER SEGMENT DATASET")
    print("=" * 80)

    if OUTPUT_DIR.exists():
        print("\nEski segmentler temizleniyor...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_segments = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for input_file in INPUT_FILES:

        if not input_file.exists():
            print(f"[UYARI] Manifest bulunamadı: {input_file}")
            continue

        split = input_file.stem

        rows = load_csv(input_file)

        print(
            f"\n{split.upper()} "
            f"orijinal kayıt: {len(rows)}"
        )

        for row in rows:

            segments = process_record(
                row,
                split
            )

            all_segments[split].extend(
                segments
            )

    fieldnames = [
        "id",
        "original_id",
        "split",
        "audio",
        "text",
        "start",
        "end",
    ]

    for split, records in all_segments.items():

        output_file = (
            OUTPUT_DIR
            / f"{split}_segments.csv"
        )

        with output_file.open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(records)

        print(
            f"{split.upper()} segment sayısı: "
            f"{len(records)}"
        )

    print()
    print("=" * 80)
    print("SEGMENTATION TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()