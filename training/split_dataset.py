import csv
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "training" / "manifests"

TRAIN_CANDIDATES_FILE = MANIFEST_DIR / "all_training_candidates.csv"
BLIND_FILE = MANIFEST_DIR / "blind_evaluation.csv"

TRAIN_FILE = MANIFEST_DIR / "train.csv"
VALIDATION_FILE = MANIFEST_DIR / "validation.csv"
TEST_FILE = MANIFEST_DIR / "test.csv"

SEED = 42
VAL_COUNT = 5  # Kör setten 5 kayıt Validation
# Kalan 6 kayıt Test olacak


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 80)
    print("TRAIN / VALIDATION / TEST SPLIT (YENİ STRATEJİ)")
    print("=" * 80)

    if not TRAIN_CANDIDATES_FILE.exists() or not BLIND_FILE.exists():
        raise FileNotFoundError("Manifest CSV dosyalarından biri bulunamadı!")

    train_candidates = read_csv(TRAIN_CANDIDATES_FILE)
    blind_records = read_csv(BLIND_FILE)

    print(f"Toplam Train Adayı (80/80 Train setine gidiyor) : {len(train_candidates)}")
    print(f"Toplam Kör Kayıt (Val/Test kümesine bölünüyor)   : {len(blind_records)}")

    if len(blind_records) < (VAL_COUNT + 1):
        raise ValueError(f"Kör set en az {VAL_COUNT + 1} kayıt olmalıdır.")

    # 1. Train kümesi = 80 aday kaydın tamamı
    train_records = train_candidates.copy()

    # 2. Kör seti karıştırıp Val (5) ve Test (6) olarak bölme
    random.seed(SEED)
    shuffled_blind = blind_records.copy()
    random.shuffle(shuffled_blind)

    val_records = shuffled_blind[:VAL_COUNT]
    test_records = shuffled_blind[VAL_COUNT:]

    # 3. CSV Kayıtları
    write_csv(TRAIN_FILE, train_records)
    write_csv(VALIDATION_FILE, val_records)
    write_csv(TEST_FILE, test_records)

    print("\n" + "=" * 80)
    print("SONUÇ")
    print("=" * 80)
    print(f"TRAIN (Adayların Tamamı) : {len(train_records)}")
    print(f"VALIDATION (Kör Set)     : {len(val_records)}")
    print(f"TEST (Kör Set)           : {len(test_records)}")
    print(f"\nOluşturuldu:\n  {TRAIN_FILE}\n  {VALIDATION_FILE}\n  {TEST_FILE}")
    print(f"Seed                     : {SEED}")


if __name__ == "__main__":
    main()