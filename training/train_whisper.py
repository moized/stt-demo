import csv
from pathlib import Path
import unicodedata

from datasets import Audio, Dataset
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ----------------------------------------------------------------------
# Yol ve Model Yapılandırması
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_DIR = PROJECT_ROOT / "training" / "segments"

TRAIN_FILE = SEGMENT_DIR / "train_segments.csv"
VALIDATION_FILE = SEGMENT_DIR / "validation_segments.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned"
)

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

    # Stereo sesleri mono kanala indirgeme
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


# ----------------------------------------------------------------------
# Main Training Loop
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("WHISPER FINE-TUNING")
    print("=" * 80)
    print(f"Base Model : {MODEL_NAME}")
    print(f"Çıktı Yolu : {OUTPUT_DIR}\n")

    # 1. Veri Yükleme ve Doğrulama
    train_records = load_and_validate_manifest(TRAIN_FILE)
    val_records = load_and_validate_manifest(VALIDATION_FILE)

    print(f"Geçerli Train Segment Sayısı      : {len(train_records)}")
    print(f"Geçerli Validation Segment Sayısı : {len(val_records)}")

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

    # 4. Eğitim Bağımsız Değişkenleri
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
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

    # 6. Kaydetme
    print("\nModel ve Processor kaydediliyor...")
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 80)
    print("EĞİTİM BAŞARIYLA TAMAMLANDI")
    print(f"Model Kayıt Yeri: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()