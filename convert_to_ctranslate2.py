import argparse
from pathlib import Path
import ctranslate2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "training" / "models"


def main():
    parser = argparse.ArgumentParser(description="Whisper to CTranslate2 Converter")
    parser.add_argument(
        "--model_name",
        default="whisper-small-finetuned-v4",
        help="Dönüştürülecek model klasörünün adı (training/models altındaki)",
    )
    parser.add_argument(
        "--quantization",
        choices=["float16", "int8_float16", "int8"],
        default="float16",
        help="Kuantizasyon formatı. Varsayılan: float16 (~480 MB)",
    )
    args = parser.parse_args()

    src_path = MODELS_DIR / args.model_name
    dst_path = MODELS_DIR / f"{args.model_name}-ct2"

    print("=" * 80)
    print("CTRANSLATE2 MODEL CONVERSION")
    print("=" * 80)
    print(f"Kaynak Model : {src_path}")
    print(f"Hedef Dizin  : {dst_path}")
    print(f"Kuantizasyon : {args.quantization}\n")

    if not src_path.exists():
        raise FileNotFoundError(f"Kaynak model bulunamadı: {src_path}")

    # Gerekli dosya kontrolü
    required_file = src_path / "model.safetensors"
    if not required_file.exists():
        required_file = src_path / "pytorch_model.bin"
        if not required_file.exists():
            raise FileNotFoundError(f"Model ağırlık dosyası bulunamadı: {src_path}")

    print("Dönüştürme işlemi başladı (Optimize ediliyor)...")
    converter = ctranslate2.converters.TransformersConverter(str(src_path))
    converter.convert(
        output_dir=str(dst_path),
        quantization=args.quantization,
        force=True,
    )

    print("\n" + "=" * 80)
    print("✓ CTranslate2 DÖNÜŞÜMÜ TAMAMLANDI")
    print(f"Konum: {dst_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()