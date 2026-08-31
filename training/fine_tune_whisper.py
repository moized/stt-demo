from pathlib import Path
import csv

import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "training" / "manifests"

TRAIN_FILE = MANIFEST_DIR / "train.csv"
VALIDATION_FILE = MANIFEST_DIR / "validation.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "models"
    / "whisper-small-finetuned"
)

MODEL_NAME = "openai/whisper-small"


# ============================================================
# CSV
# ============================================================

def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


# ============================================================
# DATASET
# ============================================================

def create_dataset(records):

    audio_paths = []
    texts = []

    for row in records:

        audio_path = PROJECT_ROOT / row["audio"]

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio bulunamadı: {audio_path}"
            )

        audio_paths.append(str(audio_path))
        texts.append(row["text"])

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": texts,
    })

    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )

    return dataset


# ============================================================
# PROCESSOR
# ============================================================

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="turkish",
    task="transcribe",
)


# ============================================================
# PREPROCESS
# ============================================================

def prepare_example(example):

    audio = example["audio"]

    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    labels = processor.tokenizer(
        example["text"]
    ).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


# ============================================================
# DATA COLLATOR
# ============================================================

class DataCollatorSpeechSeq2Seq:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):

        input_features = [
            feature["input_features"]
            for feature in features
        ]

        label_features = [
            feature["labels"]
            for feature in features
        ]

        batch = {
            "input_features": torch.tensor(
                input_features,
                dtype=torch.float32
            )
        }

        labels_batch = self.processor.tokenizer.pad(
            {
                "input_ids": label_features
            },
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"]

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        batch["labels"] = labels

        return batch


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("WHISPER FINE-TUNING")
    print("=" * 80)

    print(f"Model       : {MODEL_NAME}")
    print(f"Train       : {TRAIN_FILE}")
    print(f"Validation  : {VALIDATION_FILE}")
    print(f"Output      : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    train_records = load_csv(TRAIN_FILE)
    validation_records = load_csv(VALIDATION_FILE)

    print()
    print(f"Train kayıtları      : {len(train_records)}")
    print(f"Validation kayıtları : {len(validation_records)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Dataset oluşturuluyor...")

    train_dataset = create_dataset(train_records)
    validation_dataset = create_dataset(
        validation_records
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print("Audio preprocessing...")

    train_dataset = train_dataset.map(
        prepare_example,
        remove_columns=["audio", "text"],
    )

    validation_dataset = validation_dataset.map(
        prepare_example,
        remove_columns=["audio", "text"],
    )

    print("Dataset hazır.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Whisper modeli yükleniyor...")

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.generation_config.language = "turkish"
    model.generation_config.task = "transcribe"

    model.config.forced_decoder_ids = None

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = Seq2SeqTrainingArguments(

        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=1,

        learning_rate=1e-5,

        num_train_epochs=3,

        warmup_steps=50,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=10,

        predict_with_generate=True,

        fp16=torch.cuda.is_available(),

        save_total_limit=2,

        load_best_model_at_end=True,

        report_to="none",

        seed=42,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    data_collator = DataCollatorSpeechSeq2Seq(
        processor
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING BAŞLIYOR")
    print("=" * 80)

    trainer.train()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("Model kaydediliyor...")

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    processor.save_pretrained(
        str(OUTPUT_DIR)
    )

    print()
    print("=" * 80)
    print("TRAINING TAMAMLANDI")
    print("=" * 80)

    print(f"Model: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()