import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import jiwer
import matplotlib.pyplot as plt
import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "training" / "models"
OUTPUT_PLOT_PATH = PROJECT_ROOT / "training" / "scaling_curves.png"

# Manifest yolları
MONO_VAL_MANIFEST = PROJECT_ROOT / "training" / "segments" / "validation_segments.csv"
DUAL_VAL_MANIFEST = PROJECT_ROOT / "training" / "segments_dual_mono" / "validation_segments.csv"

# Değerlendirilecek model mimarileri ve alt küme yolları
EXPERIMENT_CONFIG = {
    "Mono-Mix (v3)": {
        "manifest": MONO_VAL_MANIFEST,
        "color": "#D9534F",  # Kırmızımsı / Turuncu
        "marker": "s",
        "runs": [
            {"ratio": 25, "path": MODELS_DIR / "whisper-small-mono-p25"},
            {"ratio": 50, "path": MODELS_DIR / "whisper-small-mono-p50"},
            {"ratio": 100, "path": MODELS_DIR / "whisper-small-finetuned"},
        ],
    },
    "Dual-Mono (v4)": {
        "manifest": DUAL_VAL_MANIFEST,
        "color": "#0275D8",  # Mavi
        "marker": "o",
        "runs": [
            {"ratio": 25, "path": MODELS_DIR / "whisper-small-dual-p25"},
            {"ratio": 50, "path": MODELS_DIR / "whisper-small-dual-p50"},
            {"ratio": 100, "path": MODELS_DIR / "whisper-small-finetuned-v4"},
        ],
    },
}


def sanitize_pipeline(pipe):
    """Pipeline modelindeki eos_token_id listeyse int yapar."""
    if pipe is not None and hasattr(pipe, "model"):
        for target in [getattr(pipe.model, "generation_config", None), getattr(pipe.model, "config", None)]:
            if target is not None and hasattr(target, "eos_token_id"):
                eos = getattr(target, "eos_token_id")
                if isinstance(eos, (list, tuple)):
                    target.eos_token_id = int(eos[0])
    return pipe


def normalize_turkish_asr(text: str) -> str:
    """ASR karşılaştırması için benchmark uyumlu standart Türkçe normalizasyonu."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = unicodedata.normalize("NFC", text)

    # Zaman damgaları ve konuşmacı etiketlerini temizle
    text = re.sub(r"\[?\d{2}:\d{2}(?::\d{2})?\]?", " ", text)
    text = re.sub(r"(konuşmacı\s+\d+|agent|customer)\s*:\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(sessizlik|anlaşılmıyor|müzik)\]", " ", text, flags=re.IGNORECASE)

    # Noktalama işaretlerini kaldır
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def evaluate_manifest_on_model(model_path: Path, manifest_path: Path, device: int) -> float:
    """Belirtilen modeli manifestteki tüm segmentler üzerinde test edip Micro WER hesaplar."""
    if not model_path.exists():
        print(f"[UYARI] Model dizini bulunamadı: {model_path.name} (Atlanıyor)")
        return None

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        device=device,
        generate_kwargs={
            "language": "turkish",
            "task": "transcribe",
            "condition_on_previous_text": False,
        },
    )
    pipe = sanitize_pipeline(pipe)

    refs = []
    hyps = []

    for r in records:
        audio_rel = r["audio"].replace("\\", "/")
        audio_full = PROJECT_ROOT / audio_rel
        if not audio_full.exists():
            continue

        ref_clean = normalize_turkish_asr(r["text"])
        if not ref_clean:
            continue

        res = pipe(str(audio_full), return_timestamps=False)
        hyp_clean = normalize_turkish_asr(res.get("text", ""))

        refs.append(ref_clean)
        hyps.append(hyp_clean)

    if not refs:
        return 0.0

    return jiwer.wer(refs, hyps) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Validation Seti Veri Ölçekleme Eğrisi")
    parser.add_argument("--device", type=int, default=0 if torch.cuda.is_available() else -1)
    args = parser.parse_args()

    print("=" * 80)
    print("VALIDATION SETİ VERİ ÖLÇEKLEME (SCALING CURVE) DEĞERLENDİRMESİ")
    print("=" * 80)
    print(f"Cihaz: {'CUDA (GPU)' if args.device == 0 else 'CPU'}\n")

    results = {}

    for mode_name, cfg in EXPERIMENT_CONFIG.items():
        print(f"\n>>> Mimarisi Değerlendiriliyor: {mode_name}")
        print(f"Validasyon Manifesti: {cfg['manifest'].name}")
        results[mode_name] = []

        for run in cfg["runs"]:
            ratio = run["ratio"]
            model_path = run["path"]
            print(f"  - %{ratio} Veri Modeli ({model_path.name})...", end=" ", flush=True)

            wer_score = evaluate_manifest_on_model(model_path, cfg["manifest"], args.device)
            if wer_score is not None:
                print(f"WER: %{wer_score:.2f}")
                results[mode_name].append((ratio, wer_score))
            else:
                print("Bulunamadı!")

    # 1. Terminal Karşılaştırma Tablosu
    print("\n" + "=" * 80)
    print("VALIDATION WER ÖLÇEKLEME TABLOSU")
    print("=" * 80)
    print(f"{'Veri Oranı':<12} | {'Mono-Mix (v3) WER':<20} | {'Dual-Mono (v4) WER':<20} | {'Fark (Kazanç)':<15}")
    print("-" * 80)

    ratios = [25, 50, 100]
    mono_dict = dict(results.get("Mono-Mix (v3)", []))
    dual_dict = dict(results.get("Dual-Mono (v4)", []))

    for r in ratios:
        m_val = mono_dict.get(r)
        d_val = dual_dict.get(r)
        m_str = f"%{m_val:.2f}" if m_val is not None else "N/A"
        d_str = f"%{d_val:.2f}" if d_val is not None else "N/A"

        diff_str = "N/A"
        if m_val is not None and d_val is not None:
            diff = m_val - d_val
            diff_str = f"{'+' if diff > 0 else ''}{diff:.2f}%"

        print(f"%{r:<11} | {m_str:<20} | {d_str:<20} | {diff_str:<15}")

    print("=" * 80)

    # 2. Matplotlib Scaling Curve Grafiği
    plt.figure(figsize=(9, 6), dpi=300)

    for mode_name, cfg in EXPERIMENT_CONFIG.items():
        data_points = results.get(mode_name, [])
        if not data_points:
            continue
        xs = [p[0] for p in data_points]
        ys = [p[1] for p in data_points]

        plt.plot(
            xs,
            ys,
            marker=cfg["marker"],
            color=cfg["color"],
            linewidth=2.5,
            markersize=8,
            label=mode_name,
        )

        for x, y in zip(xs, ys):
            plt.annotate(
                f"%{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    plt.title("Veri Ölçekleme Eğrisi (Validation WER vs Eğitim Verisi Hacmi)", fontsize=13, pad=15)
    plt.xlabel("Eğitim Verisi Oranı (%)", fontsize=11, labelpad=10)
    plt.ylabel("Validation WER (%) [Düşük Olan Daha İyi]", fontsize=11, labelpad=10)
    plt.xticks([25, 50, 100], ["%25", "%50", "%100"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"\n✓ Çift çizgili ölçekleme grafiği kaydedildi: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()