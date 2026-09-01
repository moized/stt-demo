from pathlib import Path
import re
import ctranslate2

# Dizin Yolu
PROJECT_ROOT = Path("/content/stt-demo")
MODELS_DIR = PROJECT_ROOT / "training" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def find_source_models() -> list[Path]:
    """CT2 olmayan eğitilmiş modelleri listeler."""
    return [
        d
        for d in MODELS_DIR.iterdir()
        if d.is_dir()
        and not d.name.endswith("-ct2")
        and (d / "config.json").exists()
    ]


def suggest_next_version(base_name: str) -> str:
    """Mevcut CT2 klasörlerini tarayarak otomatik bir sonraki versiyon adını önerir."""
    existing_ct2 = [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
    version = 2
    while f"{base_name}-v{version}-ct2" in existing_ct2:
        version += 1
    return f"{base_name}-v{version}-ct2"


def main():
    print("=" * 80)
    print("CTRANSLATE2 İNTERAKTİF MODEL DÖNÜŞTÜRÜCÜ")
    print("=" * 80)

    # 1. Kaynak Modeli Seçme
    source_models = find_source_models()

    if not source_models:
        default_src = MODELS_DIR / "whisper-small-finetuned"
        print(f"[BİLGİ] Otomatik kaynak model dizini: {default_src}")
        src_path = default_src
    elif len(source_models) == 1:
        src_path = source_models[0]
        print(f"✓ Kaynak Model Tespit Edildi: {src_path.name}")
    else:
        print("\nDönüştürmek istediğiniz kaynak modeli seçin:")
        for idx, model in enumerate(source_models, start=1):
            print(f"  [{idx}] {model.name}")
        choice = int(input("\nSeçiminiz (Numara): ").strip())
        src_path = source_models[choice - 1]

    if not src_path.exists():
        print(f"[HATA] Kaynak model yolu bulunamadı: {src_path}")
        return

    # 2. Hedef CT2 Model İsmi Belirleme
    suggested_name = suggest_next_version("whisper-small-finetuned")
    default_base_name = f"{src_path.name}-ct2"

    print("\n" + "-" * 80)
    print("Hedef CTranslate2 Model Kayıt Seçenekleri:")
    print(f"  [1] Önerilen Yeni Versiyon : {suggested_name}")
    print(f"  [2] Standart İsim          : {default_base_name}")
    print(f"  [3] Özel İsim Gir (Kendi belirleyeceğin bir ad)")
    print("-" * 80)

    target_choice = input("Tercihiniz [1 / 2 / 3] (Varsayılan 1): ").strip()

    if target_choice == "2":
        target_name = default_base_name
    elif target_choice == "3":
        custom_name = input("Yeni model klasör adı: ").strip()
        target_name = (
            custom_name
            if custom_name.endswith("-ct2")
            else f"{custom_name}-ct2"
        )
    else:
        target_name = suggested_name

    dst_path = MODELS_DIR / target_name

    # 3. Üzerine Yazma Kontrolü
    if dst_path.exists():
        print(f"\n[UYARI] '{target_name}' klasörü zaten mevcut!")
        confirm = (
            input(
                "Mevcut modelin üzerine yazılsın mı (replace)? [e / H]: "
            )
            .strip()
            .lower()
        )
        if confirm not in ["e", "evet", "y", "yes"]:
            print("İşlem iptal edildi. Eski modellere dokunulmadı.")
            return

    # 4. Dönüştürme İşlemi
    print("\n" + "=" * 80)
    print(f"Dönüştürülüyor: {src_path.name} -> {target_name}")
    print("=" * 80)

    converter = ctranslate2.converters.TransformersConverter(str(src_path))
    converter.convert(
        output_dir=str(dst_path),
        quantization="float16",
        force=True,
    )

    print("\n" + "=" * 80)
    print("✓ Model başarıyla dönüştürüldü ve Drive'a kaydedildi!")
    print(f"✓ Konum: {dst_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()