import time

from faster_whisper import WhisperModel


class STTEngine:

    def __init__(
        self,
        model_size="small",
        device="cpu",
        compute_type="int8"
    ):
        print(f"Loading model: {model_size}")
        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def transcribe(
        self,
        audio_path,
        speaker,
        language="tr"
    ):
        """
        Transcribe one audio channel.

        speaker:
            Agent
            Customer
        """

        start_time = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True
        )

        results = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            print(
                    f"[DEBUG] {speaker} | "
                    f"start={segment.start:.3f} | "
                    f"end={segment.end:.3f} | "
                    f"text={text}"
                )
            results.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
                "text": text,
            })

        processing_time = (
            time.perf_counter() - start_time
        )

        return results, info, processing_time