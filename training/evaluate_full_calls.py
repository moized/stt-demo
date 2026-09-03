import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MANIFEST = PROJECT_ROOT / "training" / "manifests" / "test.csv"

BASE_MODEL_NAME = "openai/whisper-small"
V3_MODEL_PATH = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned"
V4_MODEL_PATH = PROJECT_ROOT / "training" / "models" / "whisper-small-finetuned-v4"


def fix_eos_token_on_disk(model_dir: Path | str) -> None:
    """Diskteki generation_config.json dosyasında eos_token_id listeyse int yapar."""
    cfg_path = Path(model_dir) / "generation_config.json"
    if not cfg_path.exists():
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("eos_token_id"), (list, tuple)):
            data["eos_token_id"] = int(data["eos_token_id"][0])
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except Exception:
        pass


def sanitize_pipeline(pipe):
    """Bellekteki generation_config nesnesindeki token uyumsuzluklarını giderir."""
    if pipe is not None and hasattr(pipe, "model"):
        for target in [getattr(pipe.model, "generation_config", None), getattr(pipe.model, "config", None)]:
            if target is not None and hasattr(target, "eos_token_id"):
                eos = getattr(target, "eos_token_id")
                if isinstance(eos, (list, tuple)):
                    target.eos_token_id = int(eos[0])
    return pipe


def normalize_turkish_asr(text: str) -> str:
    """benchmark.py standartlarına uygun tam Türkçe ASR normalizasyonu."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = unicodedata.normalize("NFC", text)

    # Zaman damgalarını süz ([00:12], [01:10:05])
    text = re.sub(r"\[?\d{2}:\d{2}(?::\d{2})?\]?", " ", text)

    # Konuşmacı etiketlerini süz
    text = re.sub(
        r"(konuşmacı\s+\d+|agent|customer)\s*:\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Özel gürültü / sessizlik etiketlerini temizle
    text = re.sub(r"\[(sessizlik|anlaşılmıyor|müzik)\]", " ", text, flags=re.IGNORECASE)

    # Noktalama işaretlerini kaldır
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_full_reference_text(transcript_path: Path) -> str:
    """Orijinal txt transkriptindeki metadata satırlarını ayıklar."""
    raw_text = transcript_path.read_text(encoding="utf-8")
    raw_text = unicodedata.normalize("NFC", raw_text)

    clean_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("Model:")
            or line.startswith("Prompt-SHA256:")
            or line == "--- TRANSKRIPT ---"
        ):
            continue
        clean_lines.append(line)

    return normalize_turkish_asr(" ".join(clean_lines))


def get_speech_intervals(audio_16k: np.ndarray, top_db: int = 25, min_length_s: float = 0.4):
    """Sesteki sessizlikleri filtreleyip konuşma olan aralıkları döner."""
    intervals = librosa.effects.split(audio_16k, top_db=top_db, frame_length=2048, hop_length=512)
    valid_chunks = []
    for start_idx, end_idx in intervals:
        dur = (end_idx - start_idx) / 16000.0
        if dur >= min_length_s:
            pad = int(0.15 * 16000)
            s_padded = max(0, start_idx - pad)
            e_padded = min(len(audio_16k), end_idx + pad)
            valid_chunks.append({
                "start_s": start_idx / 16000.0,
                "audio": audio_16k[s_padded:e_padded]
            })
    return valid_chunks


def run_mono_inference(pipe, audio_path: Path) -> str:
    """Mono ses üzerinde standart kayan pencereli çıkarım (Base ve v3 için)."""
    audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    result = pipe(
        audio_array,
        chunk_length_s=30,
        stride_length_s=4,
        return_timestamps=False,
    )
    return result.get("text", "").strip()


def run_v4_dual_inference(pipe, audio_path: Path) -> str:
    """v4 için stereo kanalları akustik VAD ile filtreleyip kronolojik birleştirir."""
    audio_data, sr = sf.read(str(audio_path))

    if getattr(audio_data, "ndim", 1) < 2:
        left_channel = audio_data
        right_channel = audio_data
    else:
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]

    left_16k = librosa.resample(left_channel.astype(float), orig_sr=sr, target_sr=16000)
    right_16k = librosa.resample(right_channel.astype(float), orig_sr=sr, target_sr=16000)

    # İki kanalın konuşma aralıklarını bul
    left_segments = get_speech_intervals(left_16k, top_db=25)
    right_segments = get_speech_intervals(right_16k, top_db=25)

    timeline = []

    for seg in left_segments:
        res = pipe(seg["audio"], return_timestamps=False)
        text = res.get("text", "").strip()
        if text:
            timeline.append({"start": seg["start_s"], "text": text})

    for seg in right_segments:
        res = pipe(seg["audio"], return_timestamps=False)
        text = res.get("text", "").strip()
        if text:
            timeline.append({"start": seg["start_s"], "text": text})

    if not timeline:
        return ""

    timeline.sort(key=lambda x: x["start"])
    return " ".join(item["text"] for item in timeline).strip()


def main():
    parser = argparse.ArgumentParser(description="Nihai Kilitli Test Seti Benchmark: Base vs v3 vs v4")
    parser.add_argument("--base_model", default=BASE_MODEL_NAME, help="Base Whisper model adı")
    parser.add_argument("--v3_path", default=str(V3_MODEL_PATH), help="v3 model dizini")
    parser.add_argument("--v4_path", default=str(V4_MODEL_PATH), help="v4 model dizini")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1

    print("=" * 85)
    print("NİHAİ KİLİTLİ TEST SETİ BENCHMARK: Base vs v3 (Mono) vs v4 (Dual-Mono)")
    print("=" * 85)
    print(f"Test Manifesti : {TEST_MANIFEST}")
    print(f"Cihaz          : {'CUDA (GPU)' if device == 0 else 'CPU'}\n")

    fix_eos_token_on_disk(args.v3_path)
    fix_eos_token_on_disk(args.v4_path)

    with TEST_MANIFEST.open("r", encoding="utf-8-sig") as f:
        test_calls = list(csv.DictReader(f))

    print(f"Kilitli Test Çağrısı Sayısı: {len(test_calls)}\n")

    gen_kwargs = {
        "language": "turkish",
        "task": "transcribe",
        "condition_on_previous_text": False,
    }

    print(">>> Modeller yükleniyor...")
    print(f"1. Base Whisper Small ({args.base_model})...")
    pipe_base = pipeline("automatic-speech-recognition", model=args.base_model, device=device, generate_kwargs=gen_kwargs)
    pipe_base = sanitize_pipeline(pipe_base)

    print(f"2. Final v3 Mono ({Path(args.v3_path).name})...")
    pipe_v3 = pipeline("automatic-speech-recognition", model=args.v3_path, device=device, generate_kwargs=gen_kwargs)
    pipe_v3 = sanitize_pipeline(pipe_v3)

    print(f"3. Final v4 Dual-Mono ({Path(args.v4_path).name})...")
    pipe_v4 = pipeline("automatic-speech-recognition", model=args.v4_path, device=device, generate_kwargs=gen_kwargs)
    pipe_v4 = sanitize_pipeline(pipe_v4)

    all_refs = []
    all_base_preds = []
    all_v3_preds = []
    all_v4_preds = []

    print("\n" + "=" * 85)
    print("TEST SETİ İNFİRANSI BAŞLADI")
    print("=" * 85)

    for idx, call in enumerate(test_calls, start=1):
        audio_path = PROJECT_ROOT / Path(call["audio"].replace("\\", "/"))
        transcript_path = PROJECT_ROOT / Path(call["transcript"].replace("\\", "/"))

        ref_norm = extract_full_reference_text(transcript_path)

        # 3 model için çıkarım
        base_raw = run_mono_inference(pipe_base, audio_path)
        v3_raw = run_mono_inference(pipe_v3, audio_path)
        v4_raw = run_v4_dual_inference(pipe_v4, audio_path)

        base_norm = normalize_turkish_asr(base_raw)
        v3_norm = normalize_turkish_asr(v3_raw)
        v4_norm = normalize_turkish_asr(v4_raw)

        b_wer = jiwer.wer(ref_norm, base_norm) * 100.0
        v3_wer = jiwer.wer(ref_norm, v3_norm) * 100.0
        v4_wer = jiwer.wer(ref_norm, v4_norm) * 100.0

        all_refs.append(ref_norm)
        all_base_preds.append(base_norm)
        all_v3_preds.append(v3_norm)
        all_v4_preds.append(v4_norm)

        print(f"[{idx}/{len(test_calls)}] Çağrı: {call['id']}")
        print(f"  Base Whisper WER : %{b_wer:.2f}")
        print(f"  v3 (Mono) WER    : %{v3_wer:.2f} (Base'e Göre: {b_wer - v3_wer:+.2f}%)")
        print(f"  v4 (Dual) WER    : %{v4_wer:.2f} (v3'e Göre: {v3_wer - v4_wer:+.2f}%)")

    # Global Micro Değerleri
    base_micro_wer = jiwer.wer(all_refs, all_base_preds) * 100.0
    v3_micro_wer = jiwer.wer(all_refs, all_v3_preds) * 100.0
    v4_micro_wer = jiwer.wer(all_refs, all_v4_preds) * 100.0

    base_micro_cer = jiwer.cer(all_refs, all_base_preds) * 100.0
    v3_micro_cer = jiwer.cer(all_refs, all_v3_preds) * 100.0
    v4_micro_cer = jiwer.cer(all_refs, all_v4_preds) * 100.0

    domain_gain = base_micro_wer - v3_micro_wer
    spatial_gain = v3_micro_wer - v4_micro_wer
    total_gain = base_micro_wer - v4_micro_wer

    print("\n" + "=" * 85)
    print("NİHAİ A/B/C KARŞILAŞTIRMA KARNESİ (KİLİTLİ TEST ÇAĞRILARI)")
    print("=" * 85)
    print(f"{'Model Seviyesi':<25} | {'Micro WER (%)':<15} | {'Micro CER (%)':<15} | {'Katkı / Rol':<25}")
    print("-" * 85)
    print(f"{'Base Whisper Small':<25} | %{base_micro_wer:<14.2f} | %{base_micro_cer:<14.2f} | Ham Taban Başarımı")
    print(f"{'Final v3 (Mono-Mix)':<25} | %{v3_micro_wer:<14.2f} | %{v3_micro_cer:<14.2f} | Sektörel / Akustik Adaptasyon")
    print(f"{'Final v4 (Dual-Mono)':<25} | %{v4_micro_wer:<14.2f} | %{v4_micro_cer:<14.2f} | Kanal Ayrık Boru Hattı")
    print("-" * 85)
    print(f"1. Domain Adaptasyon Kazancı (Base -> v3) : {domain_gain:+.2f}% WER")
    print(f"2. Kanal Ayrıştırma Kazancı (v3 -> v4)    : {spatial_gain:+.2f}% WER")
    print(f"3. Toplam Proje Kazancı (Base -> v4)      : {total_gain:+.2f}% WER")
    print("=" * 85)


if __name__ == "__main__":
    main()