from pathlib import Path
import csv
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "training" / "manifests"


INPUT_FILE = MANIFEST_DIR / "all_training_candidates.csv"

TRAIN_FILE = MANIFEST_DIR / "train.csv"
VALIDATION_FILE = MANIFEST_DIR / "validation.csv"
TEST_FILE = MANIFEST_DIR / "test.csv"


SEED = 42


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(csv.DictReader(f))


def write_csv(path: Path, rows):

    if not rows:
        return

    fieldnames = rows[0].keys()

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 80)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {INPUT_FILE}"
        )

    records = read_csv(INPUT_FILE)

    print(f"Toplam kayıt: {len(records)}")

    if len(records) != 80:
        raise ValueError(
            f"Beklenen 80 kayıt yerine {len(records)} kayıt bulundu."
        )

    # ------------------------------------------------------------
    # Deterministic shuffle
    # ------------------------------------------------------------

    random.seed(SEED)

    random.shuffle(records)

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    train = records[:70]
    validation = records[70:75]
    test = records[75:80]

    # ------------------------------------------------------------
    # CSV yaz
    # ------------------------------------------------------------

    write_csv(TRAIN_FILE, train)
    write_csv(VALIDATION_FILE, validation)
    write_csv(TEST_FILE, test)

    # ------------------------------------------------------------
    # Sonuç
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("SONUÇ")
    print("=" * 80)

    print(f"TRAIN       : {len(train)}")
    print(f"VALIDATION  : {len(validation)}")
    print(f"TEST        : {len(test)}")

    print()
    print("Oluşturuldu:")

    print(f"  {TRAIN_FILE}")
    print(f"  {VALIDATION_FILE}")
    print(f"  {TEST_FILE}")

    print()
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()