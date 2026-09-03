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

    # 1. benchmark.py kuralı: Zaman damgalarını süz ([00:12], [01:10:05])
    text = re.sub(r"\[?\d{2}:\d{2}(?::\d{2})?\]?", " ", text)

    # 2. benchmark.py kuralı: Konuşmacı etiketlerini süz
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
        if not line or line.startswith("#") or line.startswith("Model:") or line.startswith("Prompt-SHA256:") or line == "--- TRANSKRIPT ---":
            continue
        clean_lines.append(line)

    return normalize_turkish_asr(" ".join(clean_lines))


def get_speech_intervals(audio_16k: np.ndarray, top_db: int = 25, min_length_s: float = 0.4):
    """Sesteki sessizlikleri atıp sadece konuşma olan aralıkları [başlangıç, bitiş] döner."""
    intervals = librosa.effects.split(audio_16k, top_db=top_db, frame_length=2048, hop_length=512)
    valid_chunks = []
    
    for start_idx, end_idx in intervals:
        dur = (end_idx - start_idx) / 16000.0
        if dur >= min_length_s:
            # Kelimenin başı/sonu kesilmesin diye 0.15s pay (padding) ekle
            pad = int(0.15 * 16000)
            s_padded = max(0, start_idx - pad)
            e_padded = min(len(audio_16k), end_idx + pad)
            valid_chunks.append({
                "start_s": start_idx / 16000.0,
                "audio": audio_16k[s_padded:e_padded]
            })
    return valid_chunks


def run_v3_mono_inference(pipe, audio_path: Path) -> str:
    """v3 için mono ses üzerinde halüsinasyonsuz transkripsiyon."""
    audio_array, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    result = pipe(
        audio_array,
        chunk_length_s=30,
        stride_length_s=4,
        return_timestamps=False,
    )
    return result["text"].strip()


def run_v4_dual_inference(pipe, audio_path: Path) -> str:
    """v4 için stereo kanalları akustik VAD ile süzüp kronolojik birleştirme."""
    audio_data, sr = sf.read(str(audio_path))

    if getattr(audio_data, "ndim", 1) < 2:
        left_channel = audio_data
        right_channel = audio_data
    else:
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]

    left_16k = librosa.resample(left_channel.astype(float), orig_sr=sr, target_sr=16000)
    right_16k = librosa.resample(right_channel.astype(float), orig_sr=sr, target_sr=16000)

    # İki kanalın konuşma aralıklarını ayrı ayrı bul (sessizlikler doğrudan elenir)
    left_segments = get_speech_intervals(left_16k, top_db=25)
    right_segments = get_speech_intervals(right_16k, top_db=25)

    timeline = []

    # Sol kanal (Ajan)
    for seg in left_segments:
        res = pipe(seg["audio"], return_timestamps=False)
        text = res["text"].strip()
        if text:
            timeline.append({"start": seg["start_s"], "text": text})

    # Sağ kanal (Müşteri)
    for seg in right_segments:
        res = pipe(seg["audio"], return_timestamps=False)
        text = res["text"].strip()
        if text:
            timeline.append({"start": seg["start_s"], "text": text})

    if not timeline:
        return ""

    # Konuşmaları gerçek başlama zamanına göre sıraya diz ve birleştir
    timeline.sort(key=lambda x: x["start"])
    return " ".join(item["text"] for item in timeline).strip()


def main():
    parser = argparse.ArgumentParser(description="Tam Çağrı (Full Call) A/B Benchmark")
    parser.add_argument("--v3_path", default=str(V3_MODEL_PATH), help="v3 Model dizini")
    parser.add_argument("--v4_path", default=str(V4_MODEL_PATH), help="v4 Model dizini")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1

    print("=" * 80)
    print("TAM ÇAĞRI (FULL CALL) A/B BENCHMARK: v3 (Mono) vs v4 (Dual-Mono)")
    print("=" * 80)
    print(f"Test Manifesti : {TEST_MANIFEST}")
    print(f"Cihaz          : {'CUDA' if device == 0 else 'CPU'}\n")

    fix_eos_token_on_disk(args.v3_path)
    fix_eos_token_on_disk(args.v4_path)

    with TEST_MANIFEST.open("r", encoding="utf-8-sig") as f:
        test_calls = list(csv.DictReader(f))

    print(f"Toplam Test Çağrısı: {len(test_calls)}\n")

    # generate_kwargs içine condition_on_previous_text=False ekleyerek döngü halüsinasyonlarını engelliyoruz
    gen_kwargs = {
        "language": "turkish",
        "task": "transcribe",
        "condition_on_previous_text": False,
    }

    print("Pipeline modelleri yükleniyor...")
    pipe_v3 = pipeline(
        "automatic-speech-recognition",
        model=args.v3_path,
        device=device,
        generate_kwargs=gen_kwargs,
    )
    pipe_v3 = sanitize_pipeline(pipe_v3)

    pipe_v4 = pipeline(
        "automatic-speech-recognition",
        model=args.v4_path,
        device=device,
        generate_kwargs=gen_kwargs,
    )
    pipe_v4 = sanitize_pipeline(pipe_v4)

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

        ref_norm = extract_full_reference_text(transcript_path)

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