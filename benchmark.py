import re
import unicodedata
from jiwer import cer, wer


def normalize_turkish_asr(text: str) -> str:
    """ASR değerlendirmesi için Unicode NFC ve Türkçe harf normalizasyonu."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = unicodedata.normalize("NFC", text)

    # Zaman damgalarını süzme ([00:12], [01:10:05])
    text = re.sub(r"\[?\d{2}:\d{2}(?::\d{2})?\]?", " ", text)

    # Konuşmacı etiketlerini süzme (Konuşmacı 1:, Agent:, Customer:)
    text = re.sub(
        r"(konuşmacı\s+\d+|agent|customer)\s*:\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Noktalama işaretlerini kaldırma ve boşluk düzenleme
    text = "".join(
        char
        for char in text
        if not unicodedata.category(char).startswith("P")
    )
    return re.sub(r"\s+", " ", text).strip()


def evaluate(reference: str, hypothesis: str) -> dict[str, float]:
    """İki metin arasındaki WER ve CER oranlarını hesaplar."""
    ref_clean = normalize_turkish_asr(reference)
    hyp_clean = normalize_turkish_asr(hypothesis)

    if not ref_clean:
        return {"WER": 0.0, "CER": 0.0}

    return {
        "WER": wer(ref_clean, hyp_clean),
        "CER": cer(ref_clean, hyp_clean),
    }