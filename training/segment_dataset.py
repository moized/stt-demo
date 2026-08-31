from pathlib import Path
import csv
import re
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

SEGMENT_SECONDS = 30.0

TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.*)$"
)


def parse_timestamp(match):
    first = int(match.group(1))
    second = int(match.group(2))

    if match.group(3) is not None:
        # HH:MM:SS
        hours = first
        minutes = second
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    # MM:SS
    return first * 60 + second


def parse_transcript(text):
    """
    Transcript'i utterance seviyesinde parse eder.

    Metadata:
        # taslak
        # duzelt
        Model:
        Prompt-SHA256:
        --- TRANSKRIPT ---

    kaldırılır.

    Çıktı:
        [(start_seconds, text), ...]
    """

    entries = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Metadata
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

        # Konuşmacı etiketi
        content = re.sub(
            r"^Konuşmacı\s+\d+\s*:\s*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        # Sessizlik / anlaşılmıyor
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
            entries.append(
                (timestamp, content)
            )

    return entries


def load_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def write_segment_audio(
    audio,
    sample_rate,
    start_sec,
    end_sec,
    output_path,
):
    """
    Audio'yu [start_sec, end_sec] aralığında çıkarır.

    Stereo ise iki kanalı mono'ya çevirir.
    """

    start_sec = max(0.0, start_sec)
    end_sec = min(end_sec, len(audio) / sample_rate)

    if end_sec <= start_sec:
        return False

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    segment = audio[
        start_sample:end_sample
    ]

    if len(segment) == 0:
        return False

    # Stereo -> mono
    if segment.ndim == 2:
        segment = segment.mean(axis=1)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        segment,
        sample_rate
    )

    return True


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
        print(
            f"[UYARI] Transcript yok: "
            f"{transcript_path}"
        )
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

    audio, sample_rate = sf.read(
        audio_path
    )

    total_duration = (
        len(audio) / sample_rate
    )

    segments = []

    # ------------------------------------------------------------
    # Utterance'ları 30 sn sınırını aşmadan grupla.
    #
    # Bir sonraki utterance 30 saniyeyi aşacaksa
    # yeni segment başlat.
    # ------------------------------------------------------------

    groups = []

    current_group = []
    current_start = None

    for index, (timestamp, text) in enumerate(entries):

        if current_start is None:
            current_start = timestamp
            current_group = [
                (timestamp, text)
            ]
            continue

        proposed_duration = (
            timestamp - current_start
        )

        if proposed_duration > SEGMENT_SECONDS:

            groups.append(current_group)

            current_start = timestamp

            current_group = [
                (timestamp, text)
            ]

        else:
            current_group.append(
                (timestamp, text)
            )

    if current_group:
        groups.append(current_group)

    # ------------------------------------------------------------
    # Grupları audio + text olarak oluştur
    # ------------------------------------------------------------

    for group_index, group in enumerate(
        groups,
        start=1
    ):

        start_sec = group[0][0]

        if group_index < len(groups):
            end_sec = groups[group_index][0][0]
        else:
            end_sec = total_duration

        # Güvenlik: 30 saniyeyi aşmasın
        if end_sec - start_sec > SEGMENT_SECONDS:

            end_sec = (
                start_sec + SEGMENT_SECONDS
            )

        segment_text = " ".join(
            text
            for _, text in group
        ).strip()

        if not segment_text:
            continue

        record_name = (
            f"{row['id']}"
            f"_part_{group_index:03d}"
        )

        output_dir = (
            AUDIO_DIR / split
        )

        output_audio = (
            output_dir
            / f"{record_name}.wav"
        )

        success = write_segment_audio(
            audio=audio,
            sample_rate=sample_rate,
            start_sec=start_sec,
            end_sec=end_sec,
            output_path=output_audio,
        )

        if not success:
            continue

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
            "start": start_sec,
            "end": end_sec,
            "duration": round(
                end_sec - start_sec,
                3
            ),
        })

    return segments


def main():

    print("=" * 80)
    print("WHISPER SEGMENT DATASET")
    print("=" * 80)

    # Eski segmentleri tamamen temizle
    if OUTPUT_DIR.exists():

        print(
            "\nEski segmentler temizleniyor..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_segments = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # ------------------------------------------------------------
    # Process
    # ------------------------------------------------------------

    for input_file in INPUT_FILES:

        if not input_file.exists():

            print(
                f"[UYARI] Manifest bulunamadı: "
                f"{input_file}"
            )

            continue

        split = input_file.stem

        rows = load_csv(input_file)

        print()
        print(
            f"{split.upper()} "
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

    # ------------------------------------------------------------
    # Save manifests
    # ------------------------------------------------------------

    fieldnames = [
        "id",
        "original_id",
        "split",
        "audio",
        "text",
        "start",
        "end",
        "duration",
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
            f"{split.upper()} "
            f"segment sayısı: "
            f"{len(records)}"
        )

        # Ek kontrol
        too_long = [
            r for r in records
            if float(r["duration"]) > SEGMENT_SECONDS
        ]

        if too_long:
            print(
                f"[HATA] {split}: "
                f"{len(too_long)} segment 30 saniyeden uzun!"
            )

    print()
    print("=" * 80)
    print("SEGMENTATION TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()