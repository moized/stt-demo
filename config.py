from pathlib import Path
import ctranslate2

# ============================================================
# 1. DIZIN YAPILANDIRMASI
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = OUTPUT_DIR / "channels"
EXPORTS_DIR = OUTPUT_DIR / "exports"

MODELS_DIR = PROJECT_ROOT / "training" / "models"
CTRANSLATE2_MODEL_DIR = MODELS_DIR / "whisper-small-finetuned-ct2"

# Gerekli tüm klasörlerin otomatik oluşturulması
for directory in [OUTPUT_DIR, TEMP_DIR, EXPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DONANIM VE MOTOR AYARLARI (PyTorch Olmadan)
# ============================================================
# CTranslate2 üzerinden GPU tespiti yapıyoruz
has_gpu = ctranslate2.get_cuda_device_count() > 0
DEFAULT_DEVICE = "cuda" if has_gpu else "cpu"
DEFAULT_COMPUTE_TYPE = "float16" if has_gpu else "int8"

# Gradio arayüzünde görüntülenecek model seçenekleri
MODEL_OPTIONS = {
    "Fine-Tuned Small CT2 (Bizim Model)": str(CTRANSLATE2_MODEL_DIR),
    "Whisper Small (Base)": "small",
    "Whisper Medium": "medium",
    "Whisper Base": "base",
    "Whisper Tiny": "tiny",
}

# Varsayılan model seçimi
DEFAULT_MODEL_CHOICE = (
    "Fine-Tuned Small CT2 (Bizim Model)"
    if CTRANSLATE2_MODEL_DIR.exists()
    else "Whisper Small (Base)"
)


# ============================================================
# 3. SES VE ASR PARAMETRELERI
# ============================================================
SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "tr"