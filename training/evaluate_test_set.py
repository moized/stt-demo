import argparse
import csv
from pathlib import Path
import re
import unicodedata

import jiwer
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_NAME = "openai/whisper-small"


def normalize_turkish_asr(text: str) -> str:
    """ASR değerlendirmesi için Türkçe Unicode NFC ve harf standardizasyonu."""
    if not isinstance(text, str) or not text:
        return ""

    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    # Noktalama işaretlerini kaldırma
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )
    # Fazla boşlukları temizleme
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Test CSV dosyası bulunamadı: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def transcribe_audio(model, processor, audio_path: str, device: str) -> str:
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(
        speech, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            language="turkish",
            task="transcribe",
            max_new_tokens=444,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return transcription.strip()


def main():
    parser = argparse.ArgumentParser(description="Whisper Test Seti Değerlendirme")
    parser.add_argument(
        "--mode",
        choices=["mono", "dual"],
        default="mono",
        help="Değerlendirme modu: 'mono' (v3) veya 'dual' (v4 kanal ayrık). Varsayılan: mono",
    )
    args = parser.parse_args()

    # Yolları moda göre dinamik eşle
    if args.mode == "dual":
        test_csv_path = PROJECT_ROOT / "training" / "segments_dual_mono" / "test_segments.csv"
        finetuned_model_path = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned-v4"
    else:
        test_csv_path = PROJECT_ROOT / "training" / "segments" / "test_segments.csv"
        finetuned_model_path = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"WHISPER BENCHMARK & EVALUATION [MOD: {args.mode.upper()}]")
    print("=" * 80)
    print(f"Çalıştırılan Cihaz : {device}")
    print(f"Test Manifest Yolu : {test_csv_path}")
    print(f"Değerlendirilen FT : {finetuned_model_path}\n")

    test_records = load_manifest(test_csv_path)
    print(f"Toplam Test Segment Sayısı: {len(test_records)}\n")

    # Modelleri Yükleme
    print("[1/2] Base Whisper yükleniyor...")
    base_processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_NAME, language="turkish", task="transcribe"
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME
    ).to(device)
    base_model.eval()

    print("[2/2] Fine-Tuned Whisper yükleniyor...")
    ft_processor = WhisperProcessor.from_pretrained(str(finetuned_model_path))
    ft_model = WhisperForConditionalGeneration.from_pretrained(
        str(finetuned_model_path)
    ).to(device)
    ft_model.eval()

    results = []
    all_refs, all_base_preds, all_ft_preds = [], [], []

    print("\n" + "=" * 80)
    print("TEST İNFİRANS VE KARŞILAŞTIRMA")
    print("=" * 80)

    for idx, row in enumerate(test_records, start=1):
        audio_rel_path = row["audio"].replace("\\", "/")
        audio_full_path = str(PROJECT_ROOT / audio_rel_path)
        ref_text = row["text"]

        base_pred = transcribe_audio(
            base_model, base_processor, audio_full_path, device
        )
        ft_pred = transcribe_audio(
            ft_model, ft_processor, audio_full_path, device
        )

        ref_norm = normalize_turkish_asr(ref_text)
        base_norm = normalize_turkish_asr(base_pred)
        ft_norm = normalize_turkish_asr(ft_pred)

        base_wer = jiwer.wer(ref_norm, base_norm)
        ft_wer = jiwer.wer(ref_norm, ft_norm)
        base_cer = jiwer.cer(ref_norm, base_norm)
        ft_cer = jiwer.cer(ref_norm, ft_norm)

        all_refs.append(ref_norm)
        all_base_preds.append(base_norm)
        all_ft_preds.append(ft_norm)

        results.append({
            "id": row["id"],
            "base_wer": base_wer,
            "ft_wer": ft_wer,
            "base_cer": base_cer,
            "ft_cer": ft_cer,
        })

        print(f"\n[{idx}/{len(test_records)}] Segment: {row['id']}")
        print(f"REF : {ref_norm}")
        print(f"BASE: {base_norm} (WER: {base_wer:.2%})")
        print(f"FT  : {ft_norm} (WER: {ft_wer:.2%})")

    # Metrikleri Hesaplama
    micro_base_wer = jiwer.wer(all_refs, all_base_preds)
    micro_ft_wer = jiwer.wer(all_refs, all_ft_preds)
    micro_base_cer = jiwer.cer(all_refs, all_base_preds)
    micro_ft_cer = jiwer.cer(all_refs, all_ft_preds)

    macro_base_wer = sum(r["base_wer"] for r in results) / len(results)
    macro_ft_wer = sum(r["ft_wer"] for r in results) / len(results)
    macro_base_cer = sum(r["base_cer"] for r in results) / len(results)
    macro_ft_cer = sum(r["ft_cer"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print("NİHAİ SONUÇ TABLOSU")
    print("=" * 80)
    print(
        f"MICRO WER  -> Base: %{micro_base_wer * 100:.2f} | Fine-Tuned: %{micro_ft_wer * 100:.2f}"
    )
    print(
        f"MICRO CER  -> Base: %{micro_base_cer * 100:.2f} | Fine-Tuned: %{micro_ft_cer * 100:.2f}"
    )
    print("-" * 80)
    print(
        f"MACRO WER  -> Base: %{macro_base_wer * 100:.2f} | Fine-Tuned: %{macro_ft_wer * 100:.2f}"
    )
    print(
        f"MACRO CER  -> Base: %{macro_base_cer * 100:.2f} | Fine-Tuned: %{macro_ft_cer * 100:.2f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()