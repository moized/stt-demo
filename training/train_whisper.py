import argparse
import csv
import json
from pathlib import Path
import random
import unicodedata

from datasets import Audio, Dataset
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "openai/whisper-small"
MAX_LABEL_LENGTH = 448


# ----------------------------------------------------------------------
# Veri Yükleme ve Doğrulama
# ----------------------------------------------------------------------
def load_and_validate_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Segment manifest bulunamadı: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))

    if not records:
        raise ValueError(f"Manifest dosyası boş: {path}")

    validated = []
    missing_audio_count = 0

    for row in records:
        text = unicodedata.normalize("NFC", row.get("text", "").strip())
        if not text:
            continue

        audio_rel_path = row["audio"].replace("\\", "/")
        audio_path = PROJECT_ROOT / Path(audio_rel_path)

        if not audio_path.exists():
            missing_audio_count += 1
            continue

        validated.append({"audio": str(audio_path), "text": text})

    if missing_audio_count > 0:
        print(f"[UYARI] {path.name}: {missing_audio_count} ses dosyası bulunamadı!")

    return validated


def sample_nested_subset(records: list[dict], ratio: float, seed: int = 42) -> list[dict]:
    """
    İç içe (nested) alt kümeleme:
    Sabit seed ile permütasyon yapılarak %25'lik verinin %50'nin,
    %50'lik verinin de %100'ün kesin alt kümesi kalması garanti edilir.
    """
    if ratio >= 1.0:
        return records

    n_total = len(records)
    n_sample = max(1, int(round(n_total * ratio)))

    rng = random.Random(seed)
    shuffled_indices = list(range(n_total))
    rng.shuffle(shuffled_indices)

    selected_indices = sorted(shuffled_indices[:n_sample])
    return [records[i] for i in selected_indices]


def create_hf_dataset(records: list[dict]) -> Dataset:
    audio_paths = [r["audio"] for r in records]
    texts = [r["text"] for r in records]

    ds = Dataset.from_dict({"audio": audio_paths, "text": texts})
    return ds.cast_column("audio", Audio(sampling_rate=16000))


# ----------------------------------------------------------------------
# Preprocessing & Data Collator
# ----------------------------------------------------------------------
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language="turkish", task="transcribe"
)


def prepare_example(example: dict) -> dict:
    audio = example["audio"]
    audio_array = audio["array"]

    # Stereo sesleri mono kanala indirgeme (güvenlik fallback'i)
    if getattr(audio_array, "ndim", 1) == 2:
        if audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.mean(axis=0)
        else:
            audio_array = audio_array.mean(axis=1)

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    label_ids = processor.tokenizer(
        example["text"], truncation=False
    ).input_ids

    if len(label_ids) > MAX_LABEL_LENGTH:
        raise ValueError(
            f"Metin Whisper {MAX_LABEL_LENGTH} token sınırını aşıyor ({len(label_ids)} token):\n{example['text']}"
        )

    return {"input_features": input_features, "labels": label_ids}


class DataCollatorSpeechSeq2Seq:

    def __init__(self, proc):
        self.processor = proc
        self.decoder_start_token_id = proc.tokenizer.bos_token_id

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        if (
            labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def sanitize_generation_config_file(output_dir: Path) -> None:
    """Kaydedilen generation_config.json içindeki eos_token_id listesini int yapar."""
    cfg_file = output_dir / "generation_config.json"
    if not cfg_file.exists():
        return
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("eos_token_id"), (list, tuple)):
            data["eos_token_id"] = int(data["eos_token_id"][0])
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[ONARILDI] {cfg_file.name} içindeki eos_token_id int olarak sabitlendi.")
    except Exception as exc:
        print(f"[UYARI] generation_config.json güncellenemedi: {exc}")


# ----------------------------------------------------------------------
# Main Training Loop
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning Eğitimi")
    parser.add_argument(
        "--mode",
        choices=["mono", "dual"],
        default="mono",
        help="Eğitim veri kümesi: 'mono' (v3) veya 'dual' (v4 kanal ayrık). Varsayılan: mono",
    )
    parser.add_argument(
        "--data_ratio",
        type=float,
        default=1.0,
        help="Eğitim verisi oranı: 0.25, 0.50 veya 1.0. Varsayılan: 1.0",
    )
    args = parser.parse_args()

    # Dizinleri moda ve veri oranına göre dinamik belirle
    ratio_tag = f"-p{int(args.data_ratio * 100)}" if args.data_ratio < 1.0 else ""

    if args.mode == "dual":
        segment_dir = PROJECT_ROOT / "training" / "segments_dual_mono"
        folder_name = f"whisper-small-dual{ratio_tag}" if ratio_tag else "whisper-small-finetuned-v4"
    else:
        segment_dir = PROJECT_ROOT / "training" / "segments"
        folder_name = f"whisper-small-mono{ratio_tag}" if ratio_tag else "whisper-small-finetuned"

    output_dir = PROJECT_ROOT / "training" / "models" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = segment_dir / "train_segments.csv"
    val_file = segment_dir / "validation_segments.csv"

    print("=" * 80)
    print(f"WHISPER FINE-TUNING [MOD: {args.mode.upper()} | ORAN: %{int(args.data_ratio * 100)}]")
    print("=" * 80)
    print(f"Base Model : {MODEL_NAME}")
    print(f"Veri Yolu  : {segment_dir}")
    print(f"Çıktı Yolu : {output_dir}\n")

    # 1. Veri Yükleme ve Doğrulama
    all_train_records = load_and_validate_manifest(train_file)
    val_records = load_and_validate_manifest(val_file)

    # İç içe alt kümeleme uygula (Validation ASLA filtrelenmez)
    train_records = sample_nested_subset(all_train_records, args.data_ratio, seed=42)

    print(f"Toplam Havuzdaki Train Segmenti : {len(all_train_records)}")
    print(f"Kullanılan Train Segment Sayısı  : {len(train_records)} (%{int(args.data_ratio * 100)})")
    print(f"Sabit Validation Segment Sayısı  : {len(val_records)} (%100 Kilitli)")

    if not train_records or not val_records:
        raise ValueError("Eğitim için yeterli segment verisi bulunamadı!")

    # 2. Dataset Dönüşümü
    print("\nDataset hazırlanıyor...")
    train_ds = create_hf_dataset(train_records)
    val_ds = create_hf_dataset(val_records)

    train_ds = train_ds.map(
        prepare_example,
        remove_columns=["audio", "text"],
        desc="Train ön işleme",
    )
    val_ds = val_ds.map(
        prepare_example,
        remove_columns=["audio", "text"],
        desc="Validation ön işleme",
    )

    # 3. Model Yükleme
    print("\nWhisper modeli yükleniyor...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # eos_token_id tip uyumsuzluğu koruması
    if isinstance(model.generation_config.eos_token_id, (list, tuple)):
        model.generation_config.eos_token_id = int(model.generation_config.eos_token_id[0])

    # 4. Eğitim Hiperparametreleri
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=3,
        warmup_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorSpeechSeq2Seq(processor),
        processing_class=processor,
    )

    # 5. Eğitimi Başlatma
    print("\n" + "=" * 80)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 80)
    trainer.train()

    # 6. Kaydetme ve Konfigürasyon Onarımı
    print("\nModel ve Processor kaydediliyor...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    sanitize_generation_config_file(output_dir)

    print("\n" + "=" * 80)
    print("EĞİTİM BAŞARIYLA TAMAMLANDI")
    print(f"Model Kayıt Yeri: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()