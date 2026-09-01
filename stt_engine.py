from pathlib import Path
import time
from typing import Dict, List, Tuple, Union

from config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE
from faster_whisper import WhisperModel


class STTEngine:

    def __init__(
        self,
        model_size_or_path: str = "small",
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ):
        print(f"[STTEngine] Model yükleniyor: {model_size_or_path}")
        print(f"[STTEngine] Cihaz: {device} | Kuantizasyon: {compute_type}")

        self.model = WhisperModel(
            model_size_or_path, device=device, compute_type=compute_type
        )

    def transcribe(
        self, audio_path: str, speaker: str, language: str = "tr"
    ) -> Tuple[List[Dict[str, Union[str, float]]], object, float]:
        start_time = time.perf_counter()

        segments, whisper_info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        results = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            print(
                f"[DEBUG] {speaker} | start={segment.start:.3f} | end={segment.end:.3f} | text={text}"
            )

            results.append({
                "speaker": speaker,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": text,
            })

        processing_time = time.perf_counter() - start_time
        return results, whisper_info, processing_time