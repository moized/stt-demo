from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf


def load_stereo_audio(audio_path: Union[str, Path]):
    """WAV dosyasını yükler ve Sol (Agent) / Sağ (Customer) kanallarını döner."""
    audio_path = str(audio_path)
    audio, sample_rate = sf.read(audio_path, always_2d=True)

    if audio.shape[1] < 2:
        raise ValueError(
            "Bu proje stereo (2 kanallı) ses kayıtları beklemektedir."
        )

    return sample_rate, audio[:, 0], audio[:, 1]


def save_channel(
    audio: np.ndarray, sample_rate: int, output_path: Union[str, Path]
):
    """Tek bir ses kanalını mono WAV olarak kaydeder."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)


def get_audio_info(audio_path: Union[str, Path]) -> dict:
    """Ses kaydı hakkında süre ve kanal bilgilerini döner."""
    audio_path = str(audio_path)
    info = sf.info(audio_path)
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.frames / info.samplerate,
    }