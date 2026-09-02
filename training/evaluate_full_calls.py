import argparse
import csv
from pathlib import Path
import re
import unicodedata

import jiwer
import librosa
import soundfile as sf
import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MANIFEST = PROJECT_ROOT / "training" / "manifests" / "test.csv"
V3_MODEL_PATH = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned"
V4_MODEL_PATH = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned-v4"


def normalize_turkish_asr(text: str) -> str:
    """ASR değerlendirmesi için Türkçe Unicode ve noktalama standardizasyonu."""
    if not isinstance(text, str) or not text:
        return ""

    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_full_reference_text(transcript_path: Path) -> str:
    """Orijinal txt transkriptindeki metadata satırlarını ve etiketleri temizler."""
    raw_text = transcript_path.read_text(encoding="utf-8")
    raw_text = unicodedata.normalize("NFC", raw_text)
    
    clean_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (
            line.startswith("#")
            or line.startswith("Model:")
            or line.startswith("Prompt-SHA256:")
            or line == "--- TRANSKRIPT ---"
        ):
            continue

        # Zaman damgası ve konuşmacı etiketini temizle: "[00:15] Konuşmacı 1: Alo" -> "Alo"
        line = re.sub(r"^\[\d{2}:\d{2}(?::\d{2})?\]\s*", "", line)
        line = re.sub(r"^Konuşmacı\s+\d+\s*:\s*", "", line, flags=re.IGNORECASE)

        if line.lower() in {"[sessizlik]", "[anlaşılmıyor]"}:
            continue

        line = re.sub(r"\[anlaşılmıyor\]", "", line, flags=re.IGNORECASE).strip()
        if line:
            clean_lines.append(line)

    return " ".join(clean_lines)


def run_v3_mono_inference(pipe, audio_path: Path) -> str:
    """v3 için mono ses üzerinde kayan pencereli tam çağrı transkripsiyonu."""
    audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    result = pipe(
        audio_array,
        chunk_length_s=30,
        stride_length_s=4,
        return_timestamps=False,
    )
    return result["text"].strip()


def run_v4_dual_inference(pipe, audio_path: Path) -> str:
    """v4 için stereo kanalları ayrı deşifre edip zaman damgalı birleştirme."""
    audio_data, sr = sf.read(str(audio_path))

    # Stereo değilse mono fallback uygula
    if getattr(audio_data, "ndim", 1) < 2:
        left_channel = audio_data
        right_channel = audio_data
    else:
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]

    # 16kHz resample
    left_16k = librosa.resample(left_channel.astype(float), orig_sr=sr, target_sr=16000)
    right_16k = librosa.resample(right_channel.astype(float), orig_sr=sr, target_sr=16000)

    # İki kanalı bağımsız transkribe et (zaman damgalarıyla)
    left_res = pipe(left_16k, chunk_length_s=30, stride_length_s=4, return_timestamps=True)
    right_res = pipe(right_16k, chunk_length_s=30, stride_length_s=4, return_timestamps=True)

    timeline_chunks = []

    # Sol kanal parçaları (Ajan)
    for ch in left_res.get("chunks", []):
        text = ch["text"].strip()
        if text and ch["timestamp"][0] is not None:
            timeline_chunks.append({"start": ch["timestamp"][0], "text": text})

    # Sağ kanal parçaları (Müşteri)
    for ch in right_res.get("chunks", []):
        text = ch["text"].strip()
        if text and ch["timestamp"][0] is not None:
            timeline_chunks.append({"start": ch["timestamp"][0], "text": text})

    # Kronolojik olarak sıraya diz ve birleştir
    timeline_chunks.sort(key=lambda x: x["start"])
    merged_text = " ".join([c["text"] for c in timeline_chunks])
    
    return merged_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Tam Çağrı (Full Call) A/B Benchmark")
    parser.add_argument(
        "--v3_path",
        default=str(V3_MODEL_PATH),
        help="v3 Model dizini",
    )
    parser.add_argument(
        "--v4_path",
        default=str(V4_MODEL_PATH),
        help="v4 Model dizini",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1

    print("=" * 80)
    print("TAM ÇAĞRI (FULL CALL) A/B BENCHMARK: v3 (Mono) vs v4 (Dual-Mono)")
    print("=" * 80)
    print(f"Test Manifesti : {TEST_MANIFEST}")
    print(f"Cihaz          : {'CUDA' if device == 0 else 'CPU'}\n")

    with TEST_MANIFEST.open("r", encoding="utf-8-sig") as f:
        test_calls = list(csv.DictReader(f))

    print(f"Toplam Test Çağrısı: {len(test_calls)}\n")

    # Pipeline yüklemeleri
    print("Pipeline modelleri yükleniyor...")
    pipe_v3 = pipeline(
        "automatic-speech-recognition",
        model=args.v3_path,
        device=device,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )
    pipe_v4 = pipeline(
        "automatic-speech-recognition",
        model=args.v4_path,
        device=device,
        generate_kwargs={"language": "turkish", "task": "transcribe"},
    )

    all_refs = []
    all_v3_preds = []
    all_v4_preds = []
    results = []

    print("\n" + "=" * 80)
    print("ÇAĞRI BAZLI İNFİRANS BAŞLADI")
    print("=" * 80)

    for idx, call in enumerate(test_calls, start=1):
        audio_path = PROJECT_ROOT / Path(call["audio"].replace("\\", "/"))
        transcript_path = PROJECT_ROOT / Path(call["transcript"].replace("\\", "/"))

        ref_raw = extract_full_reference_text(transcript_path)
        ref_norm = normalize_turkish_asr(ref_raw)

        # İnfirans
        v3_raw = run_v3_mono_inference(pipe_v3, audio_path)
        v4_raw = run_v4_dual_inference(pipe_v4, audio_path)

        v3_norm = normalize_turkish_asr(v3_raw)
        v4_norm = normalize_turkish_asr(v4_raw)

        v3_wer = jiwer.wer(ref_norm, v3_norm)
        v4_wer = jiwer.wer(ref_norm, v4_norm)
        v3_cer = jiwer.cer(ref_norm, v3_norm)
        v4_cer = jiwer.cer(ref_norm, v4_norm)

        all_refs.append(ref_norm)
        all_v3_preds.append(v3_norm)
        all_v4_preds.append(v4_norm)

        results.append({
            "id": call["id"],
            "v3_wer": v3_wer,
            "v4_wer": v4_wer,
            "v3_cer": v3_cer,
            "v4_cer": v4_cer,
        })

        print(f"\n[{idx}/{len(test_calls)}] Çağrı: {call['id']}")
        print(f"v3 WER : %{v3_wer * 100:.2f} (CER: %{v3_cer * 100:.2f})")
        print(f"v4 WER : %{v4_wer * 100:.2f} (CER: %{v4_cer * 100:.2f})")
        diff = (v3_wer - v4_wer) * 100
        print(f"Kazanç : {'+' if diff > 0 else ''}{diff:.2f}% (Pozitif: v4 daha iyi)")

    # Genel Metrikler
    v3_micro_wer = jiwer.wer(all_refs, all_v3_preds)
    v4_micro_wer = jiwer.wer(all_refs, all_v4_preds)
    v3_micro_cer = jiwer.cer(all_refs, all_v3_preds)
    v4_micro_cer = jiwer.cer(all_refs, all_v4_preds)

    print("\n" + "=" * 80)
    print("NİHAİ A/B KARŞILAŞTIRMA TABLOSU (TAM ÇAĞRILAR)")
    print("=" * 80)
    print(f"MICRO WER  -> v3 (Mono): %{v3_micro_wer * 100:.2f} | v4 (Dual): %{v4_micro_wer * 100:.2f}")
    print(f"MICRO CER  -> v3 (Mono): %{v3_micro_cer * 100:.2f} | v4 (Dual): %{v4_micro_cer * 100:.2f}")
    print("-" * 80)
    net_wer_gain = (v3_micro_wer - v4_micro_wer) * 100
    print(f"Net WER İyileşmesi: {net_wer_gain:+.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()