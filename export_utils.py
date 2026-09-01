import json
from pathlib import Path

from config import EXPORTS_DIR
import pandas as pd


def export_to_json(
    results: list[dict], filename: str = "latest_transcript.json"
) -> Path:
    """Zaman damgalı konuşma dizisini JSON formatında dışa aktarır."""
    output_path = EXPORTS_DIR / filename
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def export_to_csv(
    results: list[dict], filename: str = "latest_transcript.csv"
) -> Path:
    """Konuşma dizisini CSV formatında dışa aktarır."""
    output_path = EXPORTS_DIR / filename
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path