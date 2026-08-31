import re

from jiwer import (
    wer,
    cer,
)


def normalize_text(text):

    text = text.replace("İ", "i").replace("I", "ı").lower()
    # Remove timestamps such as:
    # [00:13]
    # [01:24]
    # [02:05]
    text = re.sub(
        r"\[\d{2}:\d{2}\]",
        " ",
        text
    )

    # Remove speaker labels such as:
    # Konuşmacı 1:
    # Konuşmacı 2:
    text = re.sub(
        r"konuşmacı\s+[12]\s*:",
        " ",
        text
    )

    # Keep Turkish letters, numbers and spaces.
    # Everything else becomes a space.
    text = re.sub(
        r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]",
        " ",
        text
    )

    # Collapse multiple spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def evaluate(
    reference,
    hypothesis
):

    reference_clean = normalize_text(
        reference
    )

    hypothesis_clean = normalize_text(
        hypothesis
    )

    word_error_rate = wer(
        reference_clean,
        hypothesis_clean
    )

    character_error_rate = cer(
        reference_clean,
        hypothesis_clean
    )

    return {
        "WER": word_error_rate,
        "CER": character_error_rate,
    }


if __name__ == "__main__":

    reference = """
    [00:13] Konuşmacı 1: Merhaba, Veribase İlaç şirketinden arıyorum.
    [00:20] Konuşmacı 2: Telefonu kapatmak istiyorum.
    [00:24] Konuşmacı 1: Elbette, görüşmeyi burada sonlandırabilirsiniz.
    """

    hypothesis = """
    [00:14] Konuşmacı 1: Merhaba, Velideyiz İlerç Şirketi'nden arıyorum.
    [00:20] Konuşmacı 1: Elbette. Görüşmeyi burada sonlandırabiliriz.
    [00:20] Konuşmacı 2: telefonu kapatma şiştürüm
    """

    print("REFERENCE AFTER NORMALIZATION:")
    print(normalize_text(reference))

    print()
    print("HYPOTHESIS AFTER NORMALIZATION:")
    print(normalize_text(hypothesis))

    print()

    results = evaluate(
        reference,
        hypothesis
    )

    print(
        f"WER: {results['WER']:.2%}"
    )

    print(
        f"CER: {results['CER']:.2%}"
    )