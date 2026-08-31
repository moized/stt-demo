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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEGMENT_DIR = (
    PROJECT_ROOT
    / "training"
    / "segments"
)

TRAIN_FILE = (
    SEGMENT_DIR
    / "train_segments.csv"
)

VALIDATION_FILE = (
    SEGMENT_DIR
    / "validation_segments.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "models"
    / "whisper-small-finetuned"
)

MODEL_NAME = "openai/whisper-small"

MAX_LABEL_LENGTH = 448


# ============================================================
# CSV
# ============================================================

def load_csv(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# Dataset
# ============================================================

def create_dataset(records):

    audio_paths = []
    texts = []

    for row in records:

        audio_path = (
            PROJECT_ROOT
            / Path(
                row["audio"].replace(
                    "\\",
                    "/"
                )
            )
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio bulunamadı: "
                f"{audio_path}"
            )

        audio_paths.append(
            str(audio_path)
        )

        texts.append(
            row["text"].strip()
        )

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": texts,
    })

    dataset = dataset.cast_column(
        "audio",
        Audio(
            sampling_rate=16000
        )
    )

    return dataset


# ============================================================
# Processor
# ============================================================

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="turkish",
    task="transcribe",
)


# ============================================================
# Preprocessing
# ============================================================

def prepare_example(example):

    audio = example["audio"]

    audio_array = audio["array"]

    # Stereo güvenliği
    if getattr(
        audio_array,
        "ndim",
        1
    ) == 2:

        # datasets bazı durumlarda
        # [samples, channels],
        # bazı durumlarda [channels, samples]
        # verebilir.
        #
        # En büyük ekseni sample olarak kabul ediyoruz.

        if audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.mean(
                axis=0
            )
        else:
            audio_array = audio_array.mean(
                axis=1
            )

    input_features = (
        processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
    )

    label_ids = processor.tokenizer(
        example["text"],
        truncation=False,
    ).input_ids

    # --------------------------------------------------------
    # Whisper 448 token sınırı
    # --------------------------------------------------------

    if len(label_ids) > MAX_LABEL_LENGTH:

        raise ValueError(
            "Transcript 448 token sınırını aşıyor.\n"
            f"Token sayısı: {len(label_ids)}\n"
            f"Text: {example['text']}"
        )

    return {
        "input_features": input_features,
        "labels": label_ids,
    }


# ============================================================
# Data Collator
# ============================================================

class DataCollatorSpeechSeq2Seq:

    def __init__(self, processor):

        self.processor = processor

        self.decoder_start_token_id = (
            processor.tokenizer.bos_token_id
        )

    def __call__(self, features):

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        input_features = [
            {
                "input_features":
                feature["input_features"]
            }
            for feature in features
        ]

        batch = (
            self.processor
            .feature_extractor
            .pad(
                input_features,
                return_tensors="pt",
            )
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        label_features = [
            {
                "input_ids":
                feature["labels"]
            }
            for feature in features
        ]

        labels_batch = (
            self.processor
            .tokenizer
            .pad(
                label_features,
                return_tensors="pt",
            )
        )

        labels = labels_batch[
            "input_ids"
        ]

        labels = labels.masked_fill(
            labels_batch[
                "attention_mask"
            ].ne(1),
            -100,
        )

        # ----------------------------------------------------
        # Whisper BOS token
        # ----------------------------------------------------

        if (
            labels.shape[1] > 0
            and (
                labels[:, 0]
                == self.decoder_start_token_id
            ).all().cpu().item()
        ):

            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("WHISPER FINE-TUNING")
    print("=" * 80)

    print(
        f"Model      : {MODEL_NAME}"
    )

    print(
        f"Train      : {TRAIN_FILE}"
    )

    print(
        f"Validation : {VALIDATION_FILE}"
    )

    print(
        f"Output     : {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_records = load_csv(
        TRAIN_FILE
    )

    validation_records = load_csv(
        VALIDATION_FILE
    )

    print()
    print(
        f"Train segments      : "
        f"{len(train_records)}"
    )

    print(
        f"Validation segments : "
        f"{len(validation_records)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Dataset oluşturuluyor..."
    )

    train_dataset = create_dataset(
        train_records
    )

    validation_dataset = (
        create_dataset(
            validation_records
        )
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print(
        "Audio preprocessing..."
    )

    train_dataset = train_dataset.map(
        prepare_example,
        remove_columns=[
            "audio",
            "text",
        ],
        desc="Train preprocessing",
    )

    validation_dataset = (
        validation_dataset.map(
            prepare_example,
            remove_columns=[
                "audio",
                "text",
            ],
            desc="Validation preprocessing",
        )
    )

    print(
        "Dataset hazır."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Whisper modeli yükleniyor..."
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.generation_config.language = (
        "turkish"
    )

    model.generation_config.task = (
        "transcribe"
    )

    model.generation_config.forced_decoder_ids = (
        None
    )

    model.config.forced_decoder_ids = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_args = (
        Seq2SeqTrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

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
    )

    # --------------------------------------------------------
    # Collator
    # --------------------------------------------------------

    data_collator = (
        DataCollatorSpeechSeq2Seq(
            processor
        )
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Seq2SeqTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        data_collator=data_collator,

        processing_class=processor,
    )

    # --------------------------------------------------------
    # Train
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
    print(
        "Model kaydediliyor..."
    )

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

    print(
        f"Model: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()