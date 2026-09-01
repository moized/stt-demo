import csv
from pathlib import Path
import re
import shutil
import unicodedata

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


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(match):
    first = int(match.group(1))
    second = int(match.group(2))

    if match.group(3) is not None:
        hours = first
        minutes = second
        seconds = int(match.group(3))

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    return first * 60 + second


# ============================================================
# TRANSCRIPT PARSING
# ============================================================

def parse_transcript(text):
    """
    Transcript'i utterance / silence event olarak parse eder.

    Çıktı:
        [
            {
                "timestamp": 3.0,
                "type": "speech",
                "text": "Alo."
            },
            {
                "timestamp": 5.0,
                "type": "silence",
                "text": ""
            }
        ]
    """

    if not text:
        return []

    text = unicodedata.normalize(
        "NFC",
        text
    )

    entries = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        match = TIMESTAMP_PATTERN.match(line)

        if not match:
            continue

        timestamp = parse_timestamp(match)

        content = match.group(4).strip()

        # ----------------------------------------------------
        # Speaker
        # ----------------------------------------------------

        content = re.sub(
            r"^Konuşmacı\s+\d+\s*:\s*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        # ----------------------------------------------------
        # Silence
        # ----------------------------------------------------

        if content.lower() == "[sessizlik]":

            entries.append({
                "timestamp": timestamp,
                "type": "silence",
                "text": "",
            })

            continue

        # ----------------------------------------------------
        # Unclear
        # ----------------------------------------------------

        if content.lower() == "[anlaşılmıyor]":

            entries.append({
                "timestamp": timestamp,
                "type": "unclear",
                "text": "",
            })

            continue

        # ----------------------------------------------------
        # Inline unclear marker
        # ----------------------------------------------------

        content = re.sub(
            r"\[anlaşılmıyor\]",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        if not content:
            continue

        content = unicodedata.normalize(
            "NFC",
            content
        )

        entries.append({
            "timestamp": timestamp,
            "type": "speech",
            "text": content,
        })

    return entries


# ============================================================
# CSV
# ============================================================

def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    audio, sample_rate = sf.read(path)

    # Stereo -> mono
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)

    return audio, sample_rate


def write_audio_segment(
    audio,
    sample_rate,
    start_sec,
    end_sec,
    output_path,
):

    total_duration = (
        len(audio) / sample_rate
    )

    start_sec = max(
        0.0,
        start_sec
    )

    end_sec = min(
        total_duration,
        end_sec
    )

    if end_sec <= start_sec:
        return False

    start_sample = int(
        start_sec * sample_rate
    )

    end_sample = int(
        end_sec * sample_rate
    )

    segment = audio[
        start_sample:end_sample
    ]

    if len(segment) == 0:
        return False

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


# ============================================================
# RECORD PROCESSING
# ============================================================

def process_record(
    row,
    split,
    long_utterances_report,
):

    audio_path = (
        PROJECT_ROOT
        / Path(
            row["audio"].replace(
                "\\",
                "/"
            )
        )
    )

    transcript_path = (
        PROJECT_ROOT
        / Path(
            row["transcript"].replace(
                "\\",
                "/"
            )
        )
    )

    if not audio_path.exists():

        print(
            f"[UYARI] Audio yok: "
            f"{audio_path}"
        )

        return []

    if not transcript_path.exists():

        print(
            f"[UYARI] Transcript yok: "
            f"{transcript_path}"
        )

        return []

    transcript = (
        transcript_path.read_text(
            encoding="utf-8"
        )
    )

    entries = parse_transcript(
        transcript
    )

    if not entries:
        print(
            f"[UYARI] Transcript parse "
            f"edilemedi: {transcript_path}"
        )
        return []

    audio, sample_rate = load_audio(
        audio_path
    )

    total_duration = (
        len(audio) / sample_rate
    )

    # ========================================================
    # 1. Speech utterance'larını çıkar
    # ========================================================

    speech_entries = [
        entry
        for entry in entries
        if entry["type"] == "speech"
    ]

    # ========================================================
    # 2. Her speech için bitiş timestamp'ini belirle
    #
    # Öncelik:
    #
    # A) Arada silence varsa -> silence timestamp
    # B) Arada unclear varsa -> unclear timestamp
    # C) Yoksa -> sonraki event timestamp
    # D) Son speech -> audio sonu
    # ========================================================

    utterances = []

    for i, entry in enumerate(entries):

        if entry["type"] != "speech":
            continue

        start = entry["timestamp"]

        end = None

        # Sonraki event
        for next_entry in entries[i + 1:]:

            # Silence / unclear:
            # mevcut konuşmanın bitişi
            if next_entry["type"] in {
                "silence",
                "unclear",
            }:

                end = next_entry["timestamp"]
                break

            # Bir sonraki speech
            if next_entry["type"] == "speech":

                end = next_entry["timestamp"]
                break

        if end is None:
            end = total_duration

        end = min(
            end,
            total_duration
        )

        if end <= start:
            continue

        duration = end - start

        utterances.append({
            "start": start,
            "end": end,
            "duration": duration,
            "text": entry["text"],
        })

    # ========================================================
    # 3. 30 sn sınırına göre utterance grouping
    # ========================================================

    groups = []

    current_group = []
    current_start = None

    for utterance in utterances:

        start = utterance["start"]
        end = utterance["end"]
        duration = utterance["duration"]

        # ----------------------------------------------------
        # Tek utterance zaten >30s
        # ----------------------------------------------------

        if duration > SEGMENT_SECONDS:

            # Önce mevcut grubu kapat
            if current_group:

                groups.append(
                    current_group
                )

                current_group = []
                current_start = None

            # Şimdilik atma!
            # Rapora ekle.
            long_utterances_report.append({
                "split": split,
                "record_id": row["id"],
                "start": start,
                "end": end,
                "duration": duration,
                "text": utterance["text"],
            })

            # Ayrı segment olarak koru.
            groups.append([
                utterance
            ])

            continue

        # ----------------------------------------------------
        # İlk normal utterance
        # ----------------------------------------------------

        if current_start is None:

            current_start = start
            current_group = [
                utterance
            ]

            continue

        # ----------------------------------------------------
        # Yeni utterance eklenebilir mi?
        # ----------------------------------------------------

        proposed_duration = (
            end - current_start
        )

        if proposed_duration <= SEGMENT_SECONDS:

            current_group.append(
                utterance
            )

        else:

            groups.append(
                current_group
            )

            current_group = [
                utterance
            ]

            current_start = start

    if current_group:
        groups.append(
            current_group
        )

    # ========================================================
    # 4. Audio + text oluştur
    # ========================================================

    segments = []

    for group_index, group in enumerate(
        groups,
        start=1
    ):

        start_sec = group[0]["start"]
        end_sec = group[-1]["end"]

        duration = (
            end_sec - start_sec
        )

        # Güvenlik
        if (
            duration > SEGMENT_SECONDS
            and len(group) > 1
        ):

            print(
                f"[UYARI] Beklenmeyen >30s grup: "
                f"{row['id']} "
                f"{start_sec:.2f}-"
                f"{end_sec:.2f}"
            )

            continue

        segment_text = " ".join(
            item["text"]
            for item in group
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

        success = write_audio_segment(
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
            "audio": output_audio
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
            "text": unicodedata.normalize(
                "NFC",
                segment_text
            ),
            "start": start_sec,
            "end": end_sec,
            "duration": round(
                duration,
                3
            ),
        })

    return segments


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("WHISPER SEGMENT DATASET")
    print("=" * 80)

    # --------------------------------------------------------
    # Eski segmentleri temizle
    # --------------------------------------------------------

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

    long_utterances_report = []

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for input_file in INPUT_FILES:

        if not input_file.exists():

            print(
                f"[UYARI] Manifest bulunamadı: "
                f"{input_file}"
            )

            continue

        split = input_file.stem

        rows = load_csv(
            input_file
        )

        print()
        print(
            f"{split.upper()} "
            f"orijinal kayıt: "
            f"{len(rows)}"
        )

        for row in rows:

            segments = process_record(
                row=row,
                split=split,
                long_utterances_report=(
                    long_utterances_report
                ),
            )

            all_segments[
                split
            ].extend(segments)

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

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

    for split, records in (
        all_segments.items()
    ):

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

        too_long = [
            r
            for r in records
            if float(r["duration"])
            > SEGMENT_SECONDS
        ]

        if too_long:

            print(
                f"[UYARI] {split}: "
                f"{len(too_long)} "
                f"segment 30 saniyeden uzun."
            )

    # --------------------------------------------------------
    # Long utterance report
    # --------------------------------------------------------

    report_file = (
        OUTPUT_DIR
        / "long_utterances.csv"
    )

    report_fields = [
        "split",
        "record_id",
        "start",
        "end",
        "duration",
        "text",
    ]

    with report_file.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=report_fields
        )

        writer.writeheader()
        writer.writerows(
            long_utterances_report
        )

    print()
    print("=" * 80)
    print("UZUN UTTERANCE RAPORU")
    print("=" * 80)

    if long_utterances_report:

        print(
            f"Toplam: "
            f"{len(long_utterances_report)}"
        )

        for item in long_utterances_report:

            print()
            print(
                f"[{item['split']}] "
                f"{item['record_id']}"
            )

            print(
                f"  zaman: "
                f"{item['start']:.2f} - "
                f"{item['end']:.2f}"
            )

            print(
                f"  süre: "
                f"{item['duration']:.2f} s"
            )

            print(
                f"  text: "
                f"{item['text']}"
            )

    else:

        print(
            "30 saniyeden uzun utterance yok."
        )

    print()
    print(
        f"Rapor: {report_file}"
    )

    print()
    print("=" * 80)
    print("SEGMENTATION TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()