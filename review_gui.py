import os
import sys
from pathlib import Path

# --- TCL/TK ORTAM DEĞİŞKENLERİ DÜZELTMESİ (Python 3.13 .venv) ---
base_python_dir = Path(sys.base_prefix)
tcl_dir = base_python_dir / "tcl" / "tcl8.6"
tk_dir = base_python_dir / "tcl" / "tk8.6"

if tcl_dir.exists():
    os.environ["TCL_LIBRARY"] = str(tcl_dir)
if tk_dir.exists():
    os.environ["TK_LIBRARY"] = str(tk_dir)

import time
import tkinter as tk
from tkinter import ttk, messagebox
import pygame

# --- AYARLAR VE PATİKALAR ---
BASE_DIR = Path(r"C:\Users\moham\Projects\stt-demo\data\kayitlar")
START_FOLDER_NUM = 33
END_FOLDER_NUM = 50

class STTReviewApp:
    def __init__(self, root, target_folders):
        self.root = root
        self.root.title("STT Benchmark - Manuel İnceleme ve Düzeltme Arayüzü")
        self.root.geometry("1100x860")

        self.target_folders = target_folders
        self.current_idx = 0
        self.start_time = None

        # Ses süre takibi
        self.current_pos_sec = 0.0
        self.audio_duration_sec = 0.0
        self.is_playing = False
        self.last_play_time = 0.0

        # Pygame Ses Oynatıcı
        pygame.mixer.init()

        # Arayüz Elemanları
        self.create_widgets()

        if self.target_folders:
            self.load_folder(self.current_idx)
        else:
            messagebox.showerror("Hata", "Belirtilen aralıkta klasör bulunamadı!")

    def create_widgets(self):
        # Üst Panel: Navigasyon ve Durum
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        self.lbl_status = ttk.Label(top_frame, text="", font=("Arial", 11, "bold"))
        self.lbl_status.pack(side=tk.LEFT)

        self.btn_next = ttk.Button(top_frame, text="Sonraki (Alt+Sağ) >>", command=self.next_folder)
        self.btn_next.pack(side=tk.RIGHT, padx=5)

        self.btn_prev = ttk.Button(top_frame, text="<< Önceki (Alt+Sol)", command=self.prev_folder)
        self.btn_prev.pack(side=tk.RIGHT, padx=5)

        # Ses Kontrol Paneli
        audio_frame = ttk.LabelFrame(self.root, text=" Ses Durumu & Çalma Kontrolleri ", padding=10)
        audio_frame.pack(fill=tk.X, padx=10, pady=5)

        # Kontrol Butonları Satırı
        ctrl_box = ttk.Frame(audio_frame)
        ctrl_box.pack(fill=tk.X, pady=2)

        self.btn_play = ttk.Button(ctrl_box, text="▶ / ⏸ (Tuş: 0)", command=self.toggle_audio)
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.btn_restart = ttk.Button(ctrl_box, text="↺ Başa Sar (Tuş: R)", command=self.restart_audio)
        self.btn_restart.pack(side=tk.LEFT, padx=3)

        self.btn_rewind = ttk.Button(ctrl_box, text="⏪ 3s Geri (←)", command=lambda: self.seek_audio(-3))
        self.btn_rewind.pack(side=tk.LEFT, padx=3)

        self.btn_forward = ttk.Button(ctrl_box, text="⏩ 3s İleri (→)", command=lambda: self.seek_audio(3))
        self.btn_forward.pack(side=tk.LEFT, padx=3)

        # Ses Süresi Göstergesi (Büyük ve Net)
        self.lbl_audio_time = ttk.Label(
            ctrl_box, 
            text="Ses Konumu: 00:00 / 00:00", 
            font=("Consolas", 11, "bold"), 
            foreground="#D9534F"
        )
        self.lbl_audio_time.pack(side=tk.LEFT, padx=15)

        self.lbl_timer = ttk.Label(ctrl_box, text="Harcanan: 00:00", font=("Consolas", 10, "bold"), foreground="navy")
        self.lbl_timer.pack(side=tk.RIGHT, padx=5)

        # İlerleme Çubuğu (Progress Slider)
        self.slider_pos = tk.DoubleVar()
        self.scale_progress = ttk.Scale(
            audio_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            variable=self.slider_pos,
            command=self.on_slider_move
        )
        self.scale_progress.pack(fill=tk.X, padx=5, pady=5)

        # Hızlı Metin Snippet Butonları
        snippet_frame = ttk.LabelFrame(self.root, text=" Hızlı Etiketler ", padding=5)
        snippet_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Button(snippet_frame, text="+ [anlaşılmıyor]", command=lambda: self.insert_text("[anlaşılmıyor] ")).pack(side=tk.LEFT, padx=3)
        ttk.Button(snippet_frame, text="+ [sessizlik]", command=lambda: self.insert_text("[sessizlik] ")).pack(side=tk.LEFT, padx=3)
        ttk.Button(snippet_frame, text="+ Konuşmacı 1:", command=lambda: self.insert_text("Konuşmacı 1: ")).pack(side=tk.LEFT, padx=3)
        ttk.Button(snippet_frame, text="+ Konuşmacı 2:", command=lambda: self.insert_text("Konuşmacı 2: ")).pack(side=tk.LEFT, padx=3)

        # Metin Editörü
        editor_frame = ttk.LabelFrame(self.root, text=" Transcript Editor ", padding=10)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.txt_editor = tk.Text(editor_frame, wrap=tk.WORD, font=("Consolas", 11), undo=True)
        scrollbar = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.txt_editor.yview)
        self.txt_editor.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Alt Bilgi Paneli
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)

        self.lbl_rules = ttk.Label(
            bottom_frame, 
            text="[←: 3s Geri] | [→: 3s İleri] | [0: Oynat/Durdur] | [R: Başa Sar] | [Enter: Yeni Satır + Timestamp] | [Ctrl+S: Kaydet]", 
            font=("Consolas", 9, "bold"), 
            foreground="darkgreen"
        )
        self.lbl_rules.pack(side=tk.LEFT)

        self.btn_save = ttk.Button(bottom_frame, text="💾 Kaydet (Ctrl+S)", command=self.save_transcript)
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # Tuş Bağlantıları
        self.txt_editor.bind('<Left>', self.handle_left_arrow)
        self.txt_editor.bind('<Right>', self.handle_right_arrow)
        self.txt_editor.bind('0', self.handle_zero_key)
        self.txt_editor.bind('r', self.handle_restart_key)
        self.txt_editor.bind('R', self.handle_restart_key)
        self.txt_editor.bind('<Return>', self.handle_enter_timestamp)
        self.txt_editor.bind('<Shift-Return>', self.handle_shift_enter)

        # Genel Kısayollar
        self.root.bind('<Control-s>', lambda e: self.save_transcript())
        self.root.bind('<Alt-Right>', lambda e: self.next_folder())
        self.root.bind('<Alt-Left>', lambda e: self.prev_folder())

        # Sürekli Güncelleme Döngüsü
        self.update_ui_loop()

    # --- ÖZEL TUŞ HANDLERLARI ---

    def handle_left_arrow(self, event):
        self.seek_audio(-3)
        return "break"

    def handle_right_arrow(self, event):
        self.seek_audio(3)
        return "break"

    def handle_zero_key(self, event):
        self.toggle_audio()
        return "break"

    def handle_restart_key(self, event):
        self.restart_audio()
        return "break"

    def handle_enter_timestamp(self, event):
        current_sec = self.get_current_audio_time()
        mins = int(current_sec // 60)
        secs = int(current_sec % 60)
        ts_str = f"\n[{mins:02d}:{secs:02d}] "
        self.txt_editor.insert(tk.INSERT, ts_str)
        self.txt_editor.see(tk.INSERT)
        return "break"

    def handle_shift_enter(self, event):
        self.txt_editor.insert(tk.INSERT, "\n")
        self.txt_editor.see(tk.INSERT)
        return "break"

    # --- SES VE ÇALMA MANTIĞI ---

    def get_current_audio_time(self) -> float:
        if self.is_playing:
            cur = max(0.0, self.current_pos_sec + (time.time() - self.last_play_time))
            if self.audio_duration_sec > 0 and cur > self.audio_duration_sec:
                return self.audio_duration_sec
            return cur
        return self.current_pos_sec

    def toggle_audio(self):
        if not hasattr(self, 'current_audio') or not self.current_audio:
            return

        if self.is_playing:
            self.current_pos_sec += time.time() - self.last_play_time
            pygame.mixer.music.pause()
            self.is_playing = False
        else:
            if pygame.mixer.music.get_pos() == -1:
                pygame.mixer.music.load(str(self.current_audio))
                pygame.mixer.music.play(start=max(0.0, self.current_pos_sec))
            else:
                pygame.mixer.music.unpause()
            self.last_play_time = time.time()
            self.is_playing = True

    def restart_audio(self):
        """Sesi sıfırlar ve en baştan başlatır."""
        if not hasattr(self, 'current_audio') or not self.current_audio:
            return
        self.current_pos_sec = 0.0
        pygame.mixer.music.load(str(self.current_audio))
        pygame.mixer.music.play(start=0.0)
        self.last_play_time = time.time()
        self.is_playing = True

    def seek_audio(self, offset_seconds: float):
        if not hasattr(self, 'current_audio') or not self.current_audio:
            return

        new_pos = max(0.0, self.get_current_audio_time() + offset_seconds)
        if self.audio_duration_sec > 0:
            new_pos = min(new_pos, self.audio_duration_sec)

        self.current_pos_sec = new_pos
        pygame.mixer.music.load(str(self.current_audio))

        if self.is_playing:
            pygame.mixer.music.play(start=new_pos)
            self.last_play_time = time.time()
        else:
            pygame.mixer.music.play(start=new_pos)
            pygame.mixer.music.pause()

    def on_slider_move(self, value):
        """İlerleme çubuğuna tıklandığında oraya atlar."""
        target_sec = float(value)
        if abs(target_sec - self.get_current_audio_time()) > 1.5:
            self.current_pos_sec = target_sec
            if hasattr(self, 'current_audio') and self.current_audio:
                pygame.mixer.music.load(str(self.current_audio))
                if self.is_playing:
                    pygame.mixer.music.play(start=target_sec)
                    self.last_play_time = time.time()
                else:
                    pygame.mixer.music.play(start=target_sec)
                    pygame.mixer.music.pause()

    def update_ui_loop(self):
        # 1. Kronometre güncellemesi
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.lbl_timer.config(text=f"Harcanan: {mins:02d}:{secs:02d}")

        # 2. Ses zamanı göstergesi ve slider güncellemesi
        cur_sec = self.get_current_audio_time()
        cur_min_str = f"{int(cur_sec // 60):02d}:{int(cur_sec % 60):02d}"
        dur_min_str = f"{int(self.audio_duration_sec // 60):02d}:{int(self.audio_duration_sec % 60):02d}"
        
        self.lbl_audio_time.config(text=f"Ses Konumu: {cur_min_str} / {dur_min_str}")

        if not self.scale_progress.instate(['pressed']):
            self.slider_pos.set(cur_sec)

        self.root.after(200, self.update_ui_loop)

    def load_folder(self, idx):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        self.is_playing = False
        self.current_pos_sec = 0.0

        folder_path = self.target_folders[idx]
        self.current_idx = idx
        self.start_time = time.time()

        self.lbl_status.config(text=f"Çağrı [{idx + 1}/{len(self.target_folders)}]: {folder_path.name}")

        # Ses dosyasını hazırla ve süresini al
        audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
        if audio_files:
            self.current_audio = audio_files[0]
            try:
                sound = pygame.mixer.Sound(str(self.current_audio))
                self.audio_duration_sec = sound.get_length()
            except Exception:
                self.audio_duration_sec = 0.0

            self.scale_progress.config(to=self.audio_duration_sec if self.audio_duration_sec > 0 else 100)
            pygame.mixer.music.load(str(self.current_audio))
        else:
            self.current_audio = None
            self.audio_duration_sec = 0.0

        # Transkripti yükle
        self.current_transcript_path = folder_path / "transcript.txt"
        self.txt_editor.delete("1.0", tk.END)

        if self.current_transcript_path.exists():
            with open(self.current_transcript_path, "r", encoding="utf-8") as f:
                self.txt_editor.insert(tk.END, f.read())

        self.txt_editor.focus_set()

    def save_transcript(self):
        if hasattr(self, 'current_transcript_path'):
            content = self.txt_editor.get("1.0", tk.END).strip()
            with open(self.current_transcript_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            
            elapsed_min = round((time.time() - self.start_time) / 60, 2)
            messagebox.showinfo("Kaydedildi", f"{self.current_transcript_path.name} kaydedildi.\nSüre: {elapsed_min} dk")

    def insert_text(self, text_to_insert):
        self.txt_editor.insert(tk.INSERT, text_to_insert)
        self.txt_editor.focus_set()

    def next_folder(self):
        if self.current_idx < len(self.target_folders) - 1:
            self.load_folder(self.current_idx + 1)

    def prev_folder(self):
        if self.current_idx > 0:
            self.load_folder(self.current_idx - 1)

def filter_benchmark_folders(base_dir, start_num, end_num):
    all_folders = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    selected = []
    for f in all_folders:
        prefix = f.name.split("_")[0]
        if prefix.isdigit() and start_num <= int(prefix) <= end_num:
            selected.append(f)
    return selected

if __name__ == "__main__":
    if not BASE_DIR.exists():
        print(f"Hata: Dizin bulunamadı -> {BASE_DIR}")
    else:
        target_folders = filter_benchmark_folders(BASE_DIR, START_FOLDER_NUM, END_FOLDER_NUM)
        root = tk.Tk()
        app = STTReviewApp(root, target_folders)
        root.mainloop()