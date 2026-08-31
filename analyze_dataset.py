from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FOLDERS = [
    "kayitlar",
    "01_duzeltilecek",
    "02_kontrol_edilecek",
    "03_kor_set",
]


def print_separator():
    print("\n" + "=" * 80)


def analyze_transcript(path):
    print(f"\n--- {path.relative_to(BASE_DIR)} ---")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except Exception as e:
            print(f"OKUMA HATASI: {e}")
            return

    lines = text.splitlines()

    print(f"Toplam karakter : {len(text)}")
    print(f"Toplam satır    : {len(lines)}")
    print(f"Boş olmayan satır: {sum(bool(x.strip()) for x in lines)}")

    print("\nİLK 15 SATIR:")
    for i, line in enumerate(lines[:15], start=1):
        print(f"{i:03}: {line[:300]}")

    print("\nSON 5 SATIR:")
    start = max(0, len(lines) - 5)
    for i, line in enumerate(lines[start:], start=start + 1):
        print(f"{i:03}: {line[:300]}")


def analyze_folder(folder_name):
    folder = DATA_DIR / folder_name

    print_separator()
    print(f"KLASÖR: {folder_name}")

    if not folder.exists():
        print("KLASÖR BULUNAMADI!")
        return

    files = [p for p in folder.rglob("*") if p.is_file()]

    print(f"Toplam dosya: {len(files)}")

    extensions = {}
    for file in files:
        ext = file.suffix.lower() or "[uzantısız]"
        extensions[ext] = extensions.get(ext, 0) + 1

    print("\nDOSYA TÜRLERİ:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext:15} {count}")

    print("\nDOSYA LİSTESİ:")
    for file in sorted(files):
        relative = file.relative_to(folder)
        print(f"  {relative}")

    transcript_files = [
        p for p in files
        if p.suffix.lower() in [".txt", ".md"]
    ]

    print("\nTRANSCRIPT DOSYALARI:")
    print(f"Toplam: {len(transcript_files)}")

    # Çok fazla dosya varsa bütün transcriptleri yazdırmıyoruz.
    # İlk birkaç tanesinin yapısını inceliyoruz.
    for path in sorted(transcript_files)[:5]:
        analyze_transcript(path)


def analyze_csv():
    csv_files = list(DATA_DIR.glob("*.csv"))

    print_separator()
    print("CSV DOSYALARI")

    if not csv_files:
        print("data/ içinde CSV bulunamadı.")
        return

    for path in csv_files:
        print(f"\nCSV: {path.name}")

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                sample = f.read(5000)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel

                reader = csv.reader(f, dialect)

                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 10:
                        break

                print(f"İlk {len(rows)} satır:")

                for i, row in enumerate(rows, start=1):
                    print(f"{i:03}: {row}")

        except Exception as e:
            print(f"CSV OKUMA HATASI: {e}")


def main():
    print("=" * 80)
    print("STT DATASET ANALİZİ")
    print("=" * 80)

    print(f"\nProje klasörü: {BASE_DIR}")
    print(f"Data klasörü : {DATA_DIR}")

    if not DATA_DIR.exists():
        print("\nHATA: data klasörü bulunamadı!")
        print("Script'i stt-demo klasörünün içine koyduğundan emin ol.")
        return

    for folder in FOLDERS:
        analyze_folder(folder)

    analyze_csv()

    print_separator()
    print("ANALİZ TAMAMLANDI")
    print("=" * 80)


if __name__ == "__main__":
    main()