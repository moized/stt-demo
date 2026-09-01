from pathlib import Path
import time

from audio_utils import get_audio_info, load_stereo_audio, save_channel
from benchmark import evaluate
from config import (
    DEFAULT_MODEL_CHOICE,
    MODEL_OPTIONS,
    OUTPUT_DIR,
    TEMP_DIR,
)
from export_utils import export_to_csv, export_to_json
import gradio as gr
from stt_engine import STTEngine

ENGINE = None
CURRENT_MODEL = None


def get_engine(model_key: str):
    global ENGINE, CURRENT_MODEL

    model_path_or_name = MODEL_OPTIONS.get(model_key, model_key)

    if ENGINE is None or CURRENT_MODEL != model_key:
        ENGINE = STTEngine(model_size_or_path=model_path_or_name)
        CURRENT_MODEL = model_key

    return ENGINE


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def format_transcript(results: list[dict]) -> str:
    lines = []
    for item in results:
        timestamp = format_timestamp(item["start"])
        speaker_number = "1" if item["speaker"] == "Agent" else "2"
        lines.append(
            f"[{timestamp}] Konuşmacı {speaker_number}: {item['text']}"
        )
    return "\n".join(lines)


def merge_results(
    agent_results: list[dict], customer_results: list[dict]
) -> list[dict]:
    combined = agent_results + customer_results
    combined.sort(key=lambda x: x["start"])
    return combined


def transcribe_call(audio_file: str, model_choice: str):
    if audio_file is None:
        return "Lütfen bir ses dosyası yükleyin.", "", "", None, None

    start_total = time.perf_counter()

    info = get_audio_info(audio_file)
    if info["channels"] < 2:
        return (
            "HATA: Ses kaydı stereo (2 kanal) olmalıdır.",
            "",
            "",
            None,
            None,
        )

    sample_rate, left, right = load_stereo_audio(audio_file)

    agent_path = TEMP_DIR / "agent.wav"
    customer_path = TEMP_DIR / "customer.wav"

    save_channel(left, sample_rate, agent_path)
    save_channel(right, sample_rate, customer_path)

    engine = get_engine(model_choice)

    agent_results, _, agent_time = engine.transcribe(
        str(agent_path), speaker="Agent"
    )
    customer_results, _, customer_time = engine.transcribe(
        str(customer_path), speaker="Customer"
    )

    results = merge_results(agent_results, customer_results)
    transcript = format_transcript(results)

    total_time = time.perf_counter() - start_total
    duration = info["duration"]
    realtime_factor = (total_time / duration) if duration > 0 else 0

    statistics = f"""
### Ses Kaydı Bilgileri
- **Süre:** {duration:.1f} saniye
- **Örnekleme Hızı:** {sample_rate} Hz
- **Kanal Sayısı:** {info['channels']} (Stereo)

### Performans ve İşlem
- **Kullanılan Model:** `{model_choice}`
- **Müşteri Temsilcisi (Sol):** {agent_time:.2f} sn
- **Müşteri (Sağ):** {customer_time:.2f} sn
- **Toplam İşlem Süresi:** {total_time:.2f} sn
- **Gerçek Zaman Katsayısı (RTF):** {realtime_factor:.2f}×
"""

    txt_path = OUTPUT_DIR / "latest_transcript.txt"
    txt_path.write_text(transcript, encoding="utf-8")

    json_path = export_to_json(results, "latest_transcript.json")

    return (
        transcript,
        statistics,
        "Transkripsiyon tamamlandı.",
        str(txt_path),
        str(json_path),
    )


def compare_transcript(reference_text: str, hypothesis_text: str):
    if not reference_text.strip():
        return "Lütfen manuel referans metni girin."

    if not hypothesis_text.strip():
        return "Lütfen önce transkripsiyonu çalıştırın."

    results = evaluate(reference=reference_text, hypothesis=hypothesis_text)

    return f"""
## 📊 Karşılaştırma Sonucu (Benchmark)

| Metrik | Değer |
|---|---:|
| **WER (Kelime Hata Oranı)** | **%{results['WER'] * 100:.2f}** |
| **CER (Karakter Hata Oranı)** | **%{results['CER'] * 100:.2f}** |

*Türkçe ASR Unicode NFC normalizasyonu uygulanmıştır.*
"""


with gr.Blocks(title="Türkçe Çağrı ASR Demo") as demo:
    gr.Markdown(
        """
    # 🇹🇷 Türkçe Çağrı Merkezi STT Arayüzü
    Stereo ses kayıtlarını Sol (Agent) ve Sağ (Customer) kanallarına ayırarak zaman damgalı transkripsiyon üretir.
    """
    )

    with gr.Row():
        audio_input = gr.Audio(label="Çağrı Kaydı (.wav)", type="filepath")
        model_input = gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value=DEFAULT_MODEL_CHOICE,
            label="Whisper Modeli Seçin",
        )

    transcribe_button = gr.Button("🎙️ TRANSKRİPSİYONU BAŞLAT", variant="primary")

    with gr.Row():
        transcript_output = gr.Textbox(
            label="Model Çıktısı (Transkript)", lines=18
        )
        stats_output = gr.Markdown(label="Performans Metrikleri")

    status_output = gr.Markdown()

    with gr.Row():
        download_txt = gr.File(label="TXT İndir")
        download_json = gr.File(label="JSON İndir")

    gr.Markdown("---")
    gr.Markdown("### 📊 Doğrulama & Benchmark")

    reference_input = gr.Textbox(
        label="Manuel Referans Transkript (Ground Truth)",
        placeholder="Doğrulanmış metni buraya yapıştırın...",
        lines=8,
    )

    compare_button = gr.Button(
        "📊 REFERANS İLE KARŞILAŞTIR (WER/CER)", variant="secondary"
    )
    benchmark_output = gr.Markdown()

    transcribe_button.click(
        fn=transcribe_call,
        inputs=[audio_input, model_input],
        outputs=[
            transcript_output,
            stats_output,
            status_output,
            download_txt,
            download_json,
        ],
    )

    compare_button.click(
        fn=compare_transcript,
        inputs=[reference_input, transcript_output],
        outputs=[benchmark_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)