from pathlib import Path

import numpy as np
import soundfile as sf


def load_stereo_audio(audio_path: str):
    """
    Load a WAV file and return:
        sample_rate
        left_channel
        right_channel

    Our project convention:
        LEFT  = Agent
        RIGHT = Customer
    """

    audio, sample_rate = sf.read(audio_path, always_2d=True)

    if audio.shape[1] < 2:
        raise ValueError(
            "This project expects stereo audio with two channels."
        )

    left = audio[:, 0]
    right = audio[:, 1]

    return sample_rate, left, right


def save_channel(audio: np.ndarray, sample_rate: int, output_path: str):
    """
    Save one channel as a mono WAV file.
    """

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        audio,
        sample_rate
    )


def get_audio_info(audio_path: str):
    """
    Return basic information about the recording.
    """

    info = sf.info(audio_path)

    duration = info.frames / info.samplerate

    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": duration,
    }