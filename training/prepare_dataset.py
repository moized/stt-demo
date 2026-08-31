from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "training" / "manifests"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def clean_transcript(text: str) -> str:
    """
    Transcript içindeki metadata/talimat satırlarını temizler.

    Örneğin:

    # taslak: gemini...
    # duzelt: ...

    veya:

    Model: gemini...
    Prompt-SHA256: ...
    --- TRANSKRIPT ---

    gibi satırları kaldırır.
    """

    lines = text.splitlines()

    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Boş satır
        if not stripped:
            continue

        # Gemini metadata
        if stripped.startswith("# taslak:"):
            continue

        if stripped.startswith("# duzelt:"):
            continue

        if stripped.startswith("Model:"):
            continue

        if stripped.startswith("Prompt-SHA256:"):
            continue

        if stripped == "--- TRANSKRIPT ---":
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned)


def find_audio(folder: Path):
    wav_files = list(folder.glob("*.wav"))

    if len(wav_files) == 1:
        return wav_files[0]

    if len(wav_files) == 0:
        return None

    print(f"[UYARI] Birden fazla WAV bulundu: {folder}")
    return wav_files[0]


def find_transcript(folder: Path):
    transcript = folder / "transcript.txt"

    if transcript.exists():
        return transcript

    return None


def process_folder(root: Path, category: str, transcript_type: str):
    records = []

    for folder in sorted(root.iterdir()):

        if not folder.is_dir():
            continue

        audio = find_audio(folder)
        transcript = find_transcript(folder)

        if audio is None:
            print(f"[UYARI] Audio bulunamadı: {folder}")
            continue

        if transcript is None:
            print(f"[UYARI] Transcript bulunamadı: {folder}")
            continue

        raw_text = read_text(transcript)
        cleaned_text = clean_transcript(raw_text)

        records.append(
            {
                "id": folder.name,
                "category": category,
                "transcript_type": transcript_type,
                "audio": str(audio.relative_to(PROJECT_ROOT)),
                "transcript": str(transcript.relative_to(PROJECT_ROOT)),
                "text": cleaned_text,
            }
        )

    return records


def main():

    all_records = []

    print("=" * 80)
    print("DATASET MANIFEST HAZIRLAMA")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 01 - Kendi düzelttiğin kayıtlar
    # ------------------------------------------------------------------

    folder = DATA_DIR / "01_duzeltilecek"

    if folder.exists():

        records = process_folder(
            folder,
            category="01_duzeltilecek",
            transcript_type="human_corrected",
        )

        all_records.extend(records)

        print(f"01_duzeltilecek : {len(records)} kayıt")

    # ------------------------------------------------------------------
    # 02 - Kontrol edilmiş kayıtlar
    # ------------------------------------------------------------------

    folder = DATA_DIR / "02_kontrol_edilecek"

    if folder.exists():

        records = process_folder(
            folder,
            category="02_kontrol_edilecek",
            transcript_type="human_checked",
        )

        all_records.extend(records)

        print(f"02_kontrol_edilecek : {len(records)} kayıt")

    # ------------------------------------------------------------------
    # 50 kayıt
    # ------------------------------------------------------------------

    folder = DATA_DIR / "kayitlar"

    if folder.exists():

        records = process_folder(
            folder,
            category="kayitlar",
            transcript_type="draft_or_human_corrected",
        )

        all_records.extend(records)

        print(f"kayitlar : {len(records)} kayıt")

    # ------------------------------------------------------------------
    # Kör seti AYRI tutuyoruz
    # ------------------------------------------------------------------

    blind_folder = DATA_DIR / "03_kor_set"

    blind_records = []

    if blind_folder.exists():

        blind_records = process_folder(
            blind_folder,
            category="03_kor_set",
            transcript_type="blind_evaluation",
        )

        print(f"03_kor_set : {len(blind_records)} kayıt")

    # ------------------------------------------------------------------
    # Training manifest
    # ------------------------------------------------------------------

    training_csv = OUTPUT_DIR / "all_training_candidates.csv"

    with training_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "transcript_type",
                "audio",
                "transcript",
                "text",
            ],
        )

        writer.writeheader()
        writer.writerows(all_records)

    # ------------------------------------------------------------------
    # Blind evaluation manifest
    # ------------------------------------------------------------------

    blind_csv = OUTPUT_DIR / "blind_evaluation.csv"

    with blind_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "transcript_type",
                "audio",
                "transcript",
                "text",
            ],
        )

        writer.writeheader()
        writer.writerows(blind_records)

    # ------------------------------------------------------------------
    # Özet
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SONUÇ")
    print("=" * 80)

    print(f"Training adayları : {len(all_records)}")
    print(f"Kör evaluation    : {len(blind_records)}")

    print()
    print(f"Oluşturuldu:")
    print(f"  {training_csv}")
    print(f"  {blind_csv}")


if __name__ == "__main__":
    main()