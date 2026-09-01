import os
import re
import time
from pathlib import Path
from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

BASE_DIR = Path(r"C:\Users\moham\Projects\stt-demo\data\kayitlar")
START_FOLDER_NUM = 33
END_FOLDER_NUM = 50

MODEL_ID = "gemini-3.6-flash"

# Benchmark Line Format şeması
class Utterance(BaseModel):
    timestamp: str  # [MM:SS] formatı
    speaker: str    # "Konuşmacı 1" veya "Konuşmacı 2"
    text: str       # Harfiyen metin

class TranscriptionResponse(BaseModel):
    segments: List[Utterance]

def extract_benchmark_vocabulary(base_dir: Path) -> List[str]:
    """01-32 arasındaki düzeltilmiş transcript.txt dosyalarından terim havuzu oluşturur."""
    text_corpus = []
    all_folders = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    for folder in all_folders:
        prefix = folder.name.split("_")[0]
        if prefix.isdigit() and 1 <= int(prefix) <= 32:
            txt_file = folder / "transcript.txt"
            if txt_file.exists():
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if not line.startswith("#")]
                    text_corpus.append(" ".join(lines))

    combined_text = " ".join(text_corpus)
    words = re.findall(r'\b[A-ZÇĞİÖŞÜa-zçğıöşü0-9\-]+\b', combined_text)
    
    # Zorunlu terimler ve korpus özel isimleri
    special_terms = {"Veribase", "Demomix", "Alo", "saşe", "kapsül", "mg", "ml"}
    for w in words:
        if w[0].isupper() and len(w) > 2 and w not in {"Konuşmacı", "Left", "Right", "Agent", "Customer", "Call", "Channel"}:
            special_terms.add(w)

    return sorted(list(special_terms))

def build_system_prompt(custom_vocab: List[str]) -> str:
    vocab_str = ", ".join(custom_vocab)
    return f"""
Sen profesyonel bir Türkçe Speech-to-Text (STT) benchmark transkripsiyon asistanısın.
Sana verilen stereo ses kaydını aşağıdaki KESİN KURALLARA (STRICT RULES) göre transkribe et:

1. KANAL & KONUŞMACI ETİKETİ (KRİTİK):
   - Sol Kanal (LEFT channel) = Agent (Müşteri Temsilcisi) -> Her zaman "Konuşmacı 1"
   - Sağ Kanal (RIGHT channel) = Customer (Müşteri) -> Her zaman "Konuşmacı 2"
   - Konuşmacı etiketlerini karıştırma.

2. HARFİYEN YAZIM (VERBATIM):
   - Ağızdan çıkan her dolgu kelimesini (ııı, şey, hani, yani vb.) ve konuşma dilindeki gramer hatalarını AYNEN KORU. Sakın düzeltme/temizleme yapma.
   - İlaç ve Şirket isimlerini birebir doğru yaz: Veribase, Demomix.
   - Tanınan ek terimler: [{vocab_str}]

3. BELİRSİZLİK & SESSİZLİK:
   - Ses duyuluyor ama ne söylendiği anlaşılmıyorsa sadece: [anlaşılmıyor]
   - Sessizlik süresi 3 saniyeden büyük olduğunda (yani en az 4 saniye veya daha uzun süren sessizliklerde) sadece: [sessizlik]

4. ZAMAN DAMGASI (TIMESTAMP):
   - Format kesinlikle [MM:SS] olmalıdır (Örn: [00:19]). Zaman damgaları ±2 sn hassasiyetinde olabilir.

5. EKSİK SATIR BIRAKMA:
   - Sesteki hiçbir konuşmayı atlama. Omitted speech (atlanan konuşma) en büyük hatadır.
"""

def generate_with_smart_retry(uploaded_audio, folder_name: str, system_prompt: str):
    """429 (Rate Limit / 2 dk) ve 503 (High Demand / 10 sn) hatalarını yöneten döngü."""
    attempt = 1
    while True:
        try:
            print(f"⏳ [{MODEL_ID}] Yanıt bekleniyor (Deneme #{attempt})...")
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    uploaded_audio,
                    f"Lütfen bu ses kaydını '{folder_name}' çağrısı için verilen kurallara ve JSON şemasına %100 uygun şekilde transkribe et."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=TranscriptionResponse,
                )
            )
            print("✅ Model yanıtı başarıyla alındı!")
            return response.parsed

        except APIError as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print("⚠️ Kota Aşıldı (429 Rate Limit). 2 dakika (120 saniye) bekleniyor...")
                time.sleep(120)
                attempt += 1
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                print("⚠️ Sunucu Yoğun (503 High Demand). 10 saniye sonra tekrar deneniyor...")
                time.sleep(10)
                attempt += 1
            else:
                print(f"❌ Kritik API Hatası: {e}")
                raise e
        except Exception as e:
            print(f"❌ Beklenmeyen Kod Hatası: {e}")
            raise e

def process_benchmark_range():
    # 1. Terim Havuzunu Oluştur
    vocab = extract_benchmark_vocabulary(BASE_DIR)
    print(f"✅ Çıkarılan Kelime Havuzu ({len(vocab)} adet): {vocab}\n")
    system_prompt = build_system_prompt(vocab)

    # 2. 33 ile 50 Arasındaki Klasörleri Filtrele
    all_folders = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    target_folders = []
    
    for f in all_folders:
        prefix = f.name.split("_")[0]
        if prefix.isdigit() and START_FOLDER_NUM <= int(prefix) <= END_FOLDER_NUM:
            target_folders.append(f)

    print(f"🚀 Toplam {len(target_folders)} klasör işlenecek ({START_FOLDER_NUM} - {END_FOLDER_NUM} arası)\n")

    # 3. Klasörleri Sırayla İşle
    for idx, folder in enumerate(target_folders, start=1):
        transcript_file = folder / "transcript.txt"
        audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))

        if not audio_files:
            print(f"[{idx}/{len(target_folders)}] ⚠️ Ses dosyası bulunamadı: {folder.name}")
            continue

        audio_path = audio_files[0]
        uploaded_audio = None

        try:
            print(f"[{idx}/{len(target_folders)}] 📁 {folder.name} -> Ses yükleniyor...")
            uploaded_audio = client.files.upload(file=audio_path)
            
            while uploaded_audio.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_audio = client.files.get(name=uploaded_audio.name)

            # Transkripsiyon Al (Otomatik Döngülü)
            res_data = generate_with_smart_retry(uploaded_audio, folder.name, system_prompt)

            # Benchmark Çıktı Formatı (Header Kuralı)
            output_lines = [
                f"# Call ID: {folder.name}",
                f"# Channel: Left=Agent (Konuşmacı 1), Right=Customer (Konuşmacı 2)"
            ]

            for seg in res_data.segments:
                ts = seg.timestamp if seg.timestamp.startswith("[") else f"[{seg.timestamp}]"
                line = f"{ts} {seg.speaker}: {seg.text}"
                output_lines.append(line)

            # transcript.txt Olarak Kaydet
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines) + "\n")

            print(f"[{idx}/{len(target_folders)}] ✅ Kaydedildi -> {transcript_file.name}\n")

        except Exception as e:
            print(f"[{idx}/{len(target_folders)}] ❌ İşlem durduruldu ({folder.name}): {e}\n")
            break  # Kritik bir hatada sonraki klasörlere geçmeden durur
        finally:
            if uploaded_audio:
                try:
                    client.files.delete(name=uploaded_audio.name)
                except Exception:
                    pass

if __name__ == "__main__":
    process_benchmark_range()