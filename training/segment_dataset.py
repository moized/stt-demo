import argparse
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

SEGMENT_SECONDS = 30.0
MAX_SPEAKER_GAP_SECONDS = 2.5

TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{2}):(\d{2})(?::(\d{2}))?\]\s*(.*)$"
)


# ============================================================
# TIMESTAMP & TRANSCRIPT PARSING
# ============================================================

def parse_timestamp(match):
    first = int(match.group(1))
    second = int(match.group(2))

    if match.group(3) is not None:
        hours = first
        minutes = second
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    return first * 60 + second


def parse_transcript(text):
    if not text:
        return []

    text = unicodedata.normalize("NFC", text)
    entries = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if (
            line.startswith("# taslak:")
            or line.startswith("# duzelt:")
            or line.startswith("Model:")
            or line.startswith("Prompt-SHA256:")
            or line == "--- TRANSKRIPT ---"
        ):
            continue

        match = TIMESTAMP_PATTERN.match(line)
        if not match:
            continue

        timestamp = parse_timestamp(match)
        content = match.group(4).strip()

        # Konuşmacı kimliğini tespit et (1: Sol / Ajan, 2: Sağ / Müşteri)
        speaker_id = 1
        speaker_match = re.match(r"^Konuşmacı\s+(\d+)\s*:\s*(.*)$", content, flags=re.IGNORECASE)
        if speaker_match:
            speaker_id = int(speaker_match.group(1))
            content = speaker_match.group(2).strip()

        if content.lower() == "[sessizlik]":
            entries.append({
                "timestamp": timestamp,
                "type": "silence",
                "speaker": None,
                "text": "",
            })
            continue

        if content.lower() == "[anlaşılmıyor]":
            entries.append({
                "timestamp": timestamp,
                "type": "unclear",
                "speaker": None,
                "text": "",
            })
            continue

        content = re.sub(r"\[anlaşılmıyor\]", "", content, flags=re.IGNORECASE).strip()
        if not content:
            continue

        content = unicodedata.normalize("NFC", content)
        entries.append({
            "timestamp": timestamp,
            "type": "speech",
            "speaker": speaker_id,
            "text": content,
        })

    return entries


def write_audio_segment(audio, sample_rate, start_sec, end_sec, output_path):
    total_duration = len(audio) / sample_rate
    start_sec = max(0.0, start_sec)
    end_sec = min(total_duration, end_sec)

    if end_sec <= start_sec:
        return False

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, segment, sample_rate)
    return True


def extract_utterances(entries, total_duration):
    utterances = []
    for i, entry in enumerate(entries):
        if entry["type"] != "speech":
            continue

        start = entry["timestamp"]
        end = None

        for next_entry in entries[i + 1:]:
            if next_entry["type"] in {"silence", "unclear", "speech"}:
                end = next_entry["timestamp"]
                break

        if end is None:
            end = total_duration

        end = min(end, total_duration)
        if end <= start:
            continue

        utterances.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "speaker": entry.get("speaker", 1),
            "text": entry["text"],
        })

    return utterances


def group_utterances_mono(utterances):
    groups = []
    current_group = []
    current_start = None

    for utterance in utterances:
        start = utterance["start"]
        end = utterance["end"]
        duration = utterance["duration"]

        if duration > SEGMENT_SECONDS:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_start = None
            groups.append([utterance])
            continue

        if current_start is None:
            current_start = start
            current_group = [utterance]
            continue

        if (end - current_start) <= SEGMENT_SECONDS:
            current_group.append(utterance)
        else:
            groups.append(current_group)
            current_group = [utterance]
            current_start = start

    if current_group:
        groups.append(current_group)

    return groups


def group_utterances_dual(utterances):
    groups = []
    current_group = []
    current_start = None

    for utterance in utterances:
        start = utterance["start"]
        end = utterance["end"]
        duration = utterance["duration"]

        if duration > SEGMENT_SECONDS:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_start = None
            groups.append([utterance])
            continue

        if current_start is None:
            current_start = start
            current_group = [utterance]
            continue

        time_gap = start - current_group[-1]["end"]
        total_span = end - current_start

        if total_span <= SEGMENT_SECONDS and time_gap <= MAX_SPEAKER_GAP_SECONDS:
            current_group.append(utterance)
        else:
            groups.append(current_group)
            current_group = [utterance]
            current_start = start

    if current_group:
        groups.append(current_group)

    return groups


def process_record_mono(row, split, audio, sample_rate, entries, audio_dir, long_utterances_report):
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)

    total_duration = len(audio) / sample_rate
    utterances = extract_utterances(entries, total_duration)
    groups = group_utterances_mono(utterances)

    segments = []
    for group_index, group in enumerate(groups, start=1):
        start_sec = group[0]["start"]
        end_sec = group[-1]["end"]
        duration = end_sec - start_sec

        if duration > SEGMENT_SECONDS and len(group) == 1:
            long_utterances_report.append({
                "split": split,
                "record_id": row["id"],
                "start": start_sec,
                "end": end_sec,
                "duration": duration,
                "text": group[0]["text"],
            })

        segment_text = " ".join(item["text"] for item in group).strip()
        if not segment_text:
            continue

        record_name = f"{row['id']}_part_{group_index:03d}"
        output_audio = audio_dir / split / f"{record_name}.wav"

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
            "audio": output_audio.relative_to(PROJECT_ROOT).as_posix(),
            "text": unicodedata.normalize("NFC", segment_text),
            "start": start_sec,
            "end": end_sec,
            "duration": round(duration, 3),
        })

    return segments


def process_record_dual(row, split, audio, sample_rate, entries, audio_dir, long_utterances_report):
    is_stereo = getattr(audio, "ndim", 1) == 2
    if is_stereo:
        channels = {
            1: ("agent", audio[:, 0]),
            2: ("customer", audio[:, 1]),
        }
    else:
        channels = {
            1: ("agent", audio),
            2: ("customer", audio),
        }

    total_duration = len(audio) / sample_rate
    utterances = extract_utterances(entries, total_duration)

    segments = []
    for spk_id, (spk_name, channel_audio) in channels.items():
        spk_utterances = [u for u in utterances if u["speaker"] == spk_id]
        if not spk_utterances:
            continue

        groups = group_utterances_dual(spk_utterances)

        for group_index, group in enumerate(groups, start=1):
            start_sec = group[0]["start"]
            end_sec = group[-1]["end"]
            duration = end_sec - start_sec

            if duration > SEGMENT_SECONDS and len(group) == 1:
                long_utterances_report.append({
                    "split": split,
                    "record_id": f"{row['id']}_{spk_name}",
                    "start": start_sec,
                    "end": end_sec,
                    "duration": duration,
                    "text": group[0]["text"],
                })

            segment_text = " ".join(item["text"] for item in group).strip()
            if not segment_text:
                continue

            record_name = f"{row['id']}_{spk_name}_part_{group_index:03d}"
            output_audio = audio_dir / split / f"{record_name}.wav"

            success = write_audio_segment(
                audio=channel_audio,
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
                "audio": output_audio.relative_to(PROJECT_ROOT).as_posix(),
                "text": unicodedata.normalize("NFC", segment_text),
                "start": start_sec,
                "end": end_sec,
                "duration": round(duration, 3),
            })

    return segments


def main():
    parser = argparse.ArgumentParser(description="Whisper Segmentasyon Hattı")
    parser.add_argument(
        "--mode",
        choices=["mono", "dual"],
        default="mono",
        help="Segmentasyon türü: 'mono' (v3 miks) veya 'dual' (v4 kanal ayrık). Varsayılan: mono",
    )
    args = parser.parse_args()

    if args.mode == "dual":
        output_dir = PROJECT_ROOT / "training" / "segments_dual_mono"
    else:
        output_dir = PROJECT_ROOT / "training" / "segments"

    audio_dir = output_dir / "audio"

    print("=" * 80)
    print(f"WHISPER SEGMENT DATASET [MOD: {args.mode.upper()}]")
    print(f"Hedef Dizin: {output_dir}")
    print("=" * 80)

    if output_dir.exists():
        print("\nEski segmentler temizleniyor...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments = {"train": [], "validation": [], "test": []}
    long_utterances_report = []

    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"[UYARI] Manifest bulunamadı: {input_file}")
            continue

        split = input_file.stem
        with input_file.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        print(f"\n{split.upper()} orijinal kayıt: {len(rows)}")

        for row in rows:
            audio_path = PROJECT_ROOT / Path(row["audio"].replace("\\", "/"))
            transcript_path = PROJECT_ROOT / Path(row["transcript"].replace("\\", "/"))

            if not audio_path.exists() or not transcript_path.exists():
                continue

            transcript = transcript_path.read_text(encoding="utf-8")
            entries = parse_transcript(transcript)
            if not entries:
                continue

            audio, sample_rate = sf.read(audio_path)

            if args.mode == "dual":
                segments = process_record_dual(
                    row=row,
                    split=split,
                    audio=audio,
                    sample_rate=sample_rate,
                    entries=entries,
                    audio_dir=audio_dir,
                    long_utterances_report=long_utterances_report,
                )
            else:
                segments = process_record_mono(
                    row=row,
                    split=split,
                    audio=audio,
                    sample_rate=sample_rate,
                    entries=entries,
                    audio_dir=audio_dir,
                    long_utterances_report=long_utterances_report,
                )

            all_segments[split].extend(segments)

    # Manifestleri kaydet
    fieldnames = ["id", "original_id", "split", "audio", "text", "start", "end", "duration"]

    for split, records in all_segments.items():
        output_file = output_dir / f"{split}_segments.csv"
        with output_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"{split.upper()} segment sayısı: {len(records)}")

        too_long = [r for r in records if float(r["duration"]) > SEGMENT_SECONDS]
        if too_long:
            print(f"[UYARI] {split}: {len(too_long)} segment 30 saniyeden uzun.")

    # Uzun segment raporu
    report_file = output_dir / "long_utterances.csv"
    with report_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "record_id", "start", "end", "duration", "text"]
        )
        writer.writeheader()
        writer.writerows(long_utterances_report)

    print()
    print("=" * 80)
    print("UZUN UTTERANCE RAPORU")
    print("=" * 80)

    if long_utterances_report:
        print(f"Toplam: {len(long_utterances_report)}")
        for item in long_utterances_report:
            print()
            print(f"[{item['split']}] {item['record_id']}")
            print(f"  zaman: {item['start']:.2f} - {item['end']:.2f}")
            print(f"  süre:  {item['duration']:.2f} s")
            print(f"  text:  {item['text']}")
    else:
        print("30 saniyeden uzun utterance yok.")

    print()
    print(f"Rapor: {report_file}")
    print()
    print("=" * 80)
    print("SEGMENTATION TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()