from pathlib import Path
import ctranslate2

# Dizin Yolları
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_MODEL_DIR = (
    PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned"
)
DST_MODEL_DIR = (
    PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned-ct2"
)


def convert_model():
    if not SRC_MODEL_DIR.exists():
        print(f"[HATA] Kaynak model bulunamadı: {SRC_MODEL_DIR}")
        return

    print("=" * 80)
    print("CTRANSLATE2 MODEL DÖNÜŞTÜRÜCÜ")
    print("=" * 80)
    print(f"Kaynak Model (HF PyTorch) : {SRC_MODEL_DIR}")
    print(f"Hedef Model (CTranslate2) : {DST_MODEL_DIR}")
    print("Dönüştürme işlemi başlatılıyor...\n")

    converter = ctranslate2.converters.TransformersConverter(
        str(SRC_MODEL_DIR)
    )
    converter.convert(
        output_dir=str(DST_MODEL_DIR),
        quantization="float16",  # Hem GPU hem CPU ile uyumlu yüksek performanslı kuantizasyon
        force=True,
    )

    print("\n" + "=" * 80)
    print("✓ Model dönüşümü başarıyla tamamlandı!")
    print(f"✓ CTranslate2 Model Konumu: {DST_MODEL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    convert_model()