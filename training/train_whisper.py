from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_DIR = PROJECT_ROOT / "training" / "manifests"

TRAIN_FILE = MANIFEST_DIR / "train.csv"
VALIDATION_FILE = MANIFEST_DIR / "validation.csv"

BASE_MODEL = "openai/whisper-small"

OUTPUT_DIR = PROJECT_ROOT / "training" / "output"


def read_manifest(path: Path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(csv.DictReader(f))


def check_manifest(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {path}"
        )

    records = read_manifest(path)

    required_columns = {
        "id",
        "category",
        "transcript_type",
        "audio",
        "transcript",
        "text",
    }

    if not records:
        raise ValueError(
            f"Manifest boş: {path}"
        )

    missing = required_columns - set(records[0].keys())

    if missing:
        raise ValueError(
            f"Eksik kolonlar: {missing}"
        )

    return records


def main():

    print("=" * 80)
    print("WHISPER FINE-TUNING DATA CHECK")
    print("=" * 80)

    print()
    print(f"Base model : {BASE_MODEL}")
    print()

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    train_records = check_manifest(TRAIN_FILE)

    print(f"Train kayıtları      : {len(train_records)}")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    validation_records = check_manifest(
        VALIDATION_FILE
    )

    print(
        f"Validation kayıtları : "
        f"{len(validation_records)}"
    )

    # ------------------------------------------------------------
    # Text kontrolü
    # ------------------------------------------------------------

    empty_train = [
        r for r in train_records
        if not r["text"].strip()
    ]

    empty_validation = [
        r for r in validation_records
        if not r["text"].strip()
    ]

    print()
    print(f"Boş train transcript      : {len(empty_train)}")
    print(
        f"Boş validation transcript : "
        f"{len(empty_validation)}"
    )

    # ------------------------------------------------------------
    # Audio path kontrolü
    # ------------------------------------------------------------

    missing_train_audio = []

    for record in train_records:

        audio_path = PROJECT_ROOT / record["audio"]

        if not audio_path.exists():
            missing_train_audio.append(
                str(audio_path)
            )

    missing_validation_audio = []

    for record in validation_records:

        audio_path = PROJECT_ROOT / record["audio"]

        if not audio_path.exists():
            missing_validation_audio.append(
                str(audio_path)
            )

    print()
    print(
        f"Bulunamayan train audio      : "
        f"{len(missing_train_audio)}"
    )

    print(
        f"Bulunamayan validation audio : "
        f"{len(missing_validation_audio)}"
    )

    # ------------------------------------------------------------
    # İlk örnek
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("İLK TRAIN ÖRNEĞİ")
    print("-" * 80)

    first = train_records[0]

    print(f"ID         : {first['id']}")
    print(f"Category   : {first['category']}")
    print(f"Audio      : {first['audio']}")
    print(f"Transcript : {first['transcript']}")
    print(f"Text       : {first['text'][:300]}")

    # ------------------------------------------------------------
    # Sonuç
    # ------------------------------------------------------------

    print()
    print("=" * 80)

    if (
        empty_train
        or empty_validation
        or missing_train_audio
        or missing_validation_audio
    ):

        print("DATA CHECK: BAŞARISIZ")
        print("Yukarıdaki sorunları düzelt.")

    else:

        print("DATA CHECK: BAŞARILI")
        print("Whisper training için manifest yapısı hazır.")

    print("=" * 80)


if __name__ == "__main__":
    main()