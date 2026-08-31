import os
import time
from pathlib import Path

import gradio as gr

from audio_utils import (
    get_audio_info,
    load_stereo_audio,
    save_channel,
)

from stt_engine import STTEngine

from benchmark import evaluate


OUTPUT_DIR = Path("outputs")
TEMP_DIR = OUTPUT_DIR / "channels"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load model once when application starts
# ---------------------------------------------------------

ENGINE = None
CURRENT_MODEL = None


def get_engine(model_size):

    global ENGINE
    global CURRENT_MODEL

    if ENGINE is None or CURRENT_MODEL != model_size:

        ENGINE = STTEngine(
            model_size=model_size,
            device="cpu",
            compute_type="int8"
        )

        CURRENT_MODEL = model_size

    return ENGINE


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def format_timestamp(seconds):

    total_seconds = int(seconds)

    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


def format_transcript(results):

    lines = []

    for item in results:

        timestamp = format_timestamp(
            item["start"]
        )

        speaker_number = (
            "1"
            if item["speaker"] == "Agent"
            else "2"
        )

        lines.append(
            f"[{timestamp}] "
            f"Konuşmacı {speaker_number}: "
            f"{item['text']}"
        )

    return "\n".join(lines)


def merge_results(agent_results, customer_results):

    combined = (
        agent_results +
        customer_results
    )

    combined.sort(
        key=lambda x: x["start"]
    )

    return combined


# ---------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------

def transcribe_call(audio_file, model_size):

    if audio_file is None:

        return (
            "Please upload an audio file.",
            "",
            "",
            None
        )

    start_total = time.perf_counter()

    audio_path = audio_file

    info = get_audio_info(
        audio_path
    )

    if info["channels"] < 2:

        return (
            "ERROR: The recording must be stereo.",
            "",
            "",
            None
        )

    sample_rate, left, right = (
        load_stereo_audio(
            audio_path
        )
    )

    # -------------------------------------------------
    # Save channels
    # -------------------------------------------------

    agent_path = (
        TEMP_DIR / "agent.wav"
    )

    customer_path = (
        TEMP_DIR / "customer.wav"
    )

    save_channel(
        left,
        sample_rate,
        agent_path
    )

    save_channel(
        right,
        sample_rate,
        customer_path
    )

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------

    engine = get_engine(
        model_size
    )

    # -------------------------------------------------
    # Agent transcription
    # -------------------------------------------------

    agent_results, _, agent_time = (
        engine.transcribe(
            str(agent_path),
            speaker="Agent"
        )
    )

    # -------------------------------------------------
    # Customer transcription
    # -------------------------------------------------

    customer_results, _, customer_time = (
        engine.transcribe(
            str(customer_path),
            speaker="Customer"
        )
    )

    # -------------------------------------------------
    # Merge
    # -------------------------------------------------

    results = merge_results(
        agent_results,
        customer_results
    )

    transcript = format_transcript(
        results
    )

    total_time = (
        time.perf_counter()
        - start_total
    )

    duration = info["duration"]

    if duration > 0:

        realtime_factor = (
            total_time / duration
        )

    else:

        realtime_factor = 0

    statistics = f"""
### Recording

**Duration:** {duration:.1f} seconds

**Sample rate:** {sample_rate} Hz

**Channels:** {info["channels"]}

### Processing

**Model:** Whisper / faster-whisper `{model_size}`

**Agent processing:** {agent_time:.2f} sec

**Customer processing:** {customer_time:.2f} sec

**Total processing:** {total_time:.2f} sec

**Real-time factor:** {realtime_factor:.2f}×

### Speakers

🧑 Agent = LEFT channel

👤 Customer = RIGHT channel
"""

    # -------------------------------------------------
    # Save transcript
    # -------------------------------------------------

    output_file = (
        OUTPUT_DIR / "latest_transcript.txt"
    )

    output_file.write_text(
        transcript,
        encoding="utf-8"
    )

    return (
        transcript,
        statistics,
        "Transcription completed.",
        str(output_file)
    )

# ---------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------

def compare_transcript(reference_text, hypothesis_text):

    if not reference_text.strip():
        return "Please enter the manual/reference transcript."

    if not hypothesis_text.strip():
        return "Please run the STT transcription first."

    results = evaluate(
        reference=reference_text,
        hypothesis=hypothesis_text
    )

    return f"""
## 📊 Benchmark Result

| Metric | Result |
|---|---:|
| **WER** | {results["WER"]:.2%} |
| **CER** | {results["CER"]:.2%} |

### What this means

**WER (Word Error Rate)** measures how many word-level
errors the STT model made compared with the reference.

**CER (Character Error Rate)** measures character-level
differences between the reference and the model output.

Lower is better.
"""

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

with gr.Blocks(
    title="Turkish STT Demo"
) as demo:

    gr.Markdown(
        """
# 🇹🇷 Turkish Call STT Demo

### Veribase-style stereo call transcription

This demo:

1. Reads the stereo call
2. Separates LEFT and RIGHT channels
3. Sends each channel to Whisper
4. Assigns Agent / Customer labels
5. Merges the results by timestamp
6. Shows the final transcript
"""
    )

    with gr.Row():

        audio_input = gr.Audio(
            label="Call recording",
            type="filepath"
        )

        model_input = gr.Dropdown(
            choices=[
                "tiny",
                "base",
                "small",
                "medium"
            ],
            value="small",
            label="Whisper model"
        )

    transcribe_button = gr.Button(
        "🎙️ TRANSCRIBE",
        variant="primary"
    )
    reference_input = gr.Textbox(
    label="Manual Reference Transcript",
    placeholder="Paste your manually corrected transcript here...",
    lines=15
    )

    with gr.Row():

        transcript_output = gr.Textbox(
            label="Transcript",
            lines=25,
        )

        stats_output = gr.Markdown(
            label="Statistics"
        )

    status_output = gr.Markdown()

    compare_button = gr.Button(
    "📊 COMPARE WITH REFERENCE",
    variant="secondary"
)

    benchmark_output = gr.Markdown()

    download_output = gr.File(
        label="Download transcript"
    )

    transcribe_button.click(
        fn=transcribe_call,
        inputs=[
            audio_input,
            model_input
        ],
        outputs=[
            transcript_output,
            stats_output,
            status_output,
            download_output
        ]
    )

    compare_button.click(
    fn=compare_transcript,
    inputs=[
        reference_input,
        transcript_output
    ],
    outputs=[
        benchmark_output
    ]
)


if __name__ == "__main__":

    demo.launch(
        inbrowser=True
    )