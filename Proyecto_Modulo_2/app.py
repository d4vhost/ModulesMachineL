import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import translators as ts
from gtts import gTTS
from playsound import playsound
import threading
import os
import time
import glob
import atexit 
import cv2
import numpy as np
import warnings

# Suprimir advertencias
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports para TrOCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
    print("TrOCR disponible (modelo avanzado)")
except ImportError:
    TROCR_AVAILABLE = False
    print("TrOCR no disponible, usando EasyOCR")
    import easyocr

COLOR_BG = "#F7F3E8"       
COLOR_CARD = "#FFFFFF"     
COLOR_FG = "#5D4037"       
COLOR_ACCENT = "#80CBC4"   
COLOR_ACCENT_2 = "#FFCC80" 
COLOR_SUCCESS = "#66BB6A"  
COLOR_ERROR = "#EF5350"    

FONT_TITLE = ("Garamond", 24, "bold") 
FONT_BODY = ("Helvetica Neue", 11)
FONT_BODY_BOLD = ("Helvetica Neue", 11, "bold")

# 2. DICCIONARIO DE IDIOMAS 
SUPPORTED_LANGUAGES = {
    "Inglés": "en",
    "Español": "es",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Portugués": "pt",
    "Japonés": "ja",
    "Chino (Simplificado)": "zh-CN",
    "Ruso": "ru",
    "Coreano": "ko"
}


class OCRTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Reconocimiento de caracteres manuscritos")
        
        window_width = 950
        window_height = 720
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int((screen_width / 2) - (window_width / 2))
        center_y = int((screen_height / 2) - (window_height / 2))
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        self.image_path = None
        self.original_text = ""
        self.current_audio_file = None
        
        # Limpiar audios viejos al iniciar
        self.cleanup_old_files()

        # CARGAR MODELO
        if TROCR_AVAILABLE:
            print("Cargando TrOCR para manuscritos...")
            try:
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                print(f"TrOCR cargado en {self.device}")
                self.use_trocr = True
            except Exception as e:
                print(f"Error cargando TrOCR: {e}, usando EasyOCR")
                self.ocr_reader = easyocr.Reader(['es'], gpu=False, verbose=False)
                self.use_trocr = False
        else:
            print("Cargando EasyOCR...")
            self.ocr_reader = easyocr.Reader(['es'], gpu=False, verbose=False)
            self.use_trocr = False

        self.setup_styles()
        self.create_widgets()
        atexit.register(self.cleanup_old_files) 

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="ridge", borderwidth=1)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT, foreground="#FFFFFF", borderwidth=0, padding=(15, 10))
        self.style.map("TButton", background=[("active", "#4DB6AC"), ("disabled", "#D7CCC8")])
        self.style.configure("Action.TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT_2, foreground="#5D4037", borderwidth=0, padding=(15, 10))
        self.style.map("Action.TButton", background=[("active", "#FFB74D"), ("disabled", "#D7CCC8")])
        self.style.configure("TCombobox", fieldbackground=COLOR_CARD, background=COLOR_CARD, foreground=COLOR_FG, selectbackground=COLOR_ACCENT_2)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Reconocimiento de caracteres manuscritos", style="Title.TLabel").pack(pady=(0, 20))

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=10)

        self.load_button = ttk.Button(top_frame, text="1. Cargar Imagen", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.ocr_button = ttk.Button(top_frame, text="2. Leer Manuscrito", command=self.recognize_text, state="disabled")
        self.ocr_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        image_frame = ttk.Frame(content_frame, style="Card.TFrame", padding=10)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="📷 Esperando imagen...", style="Card.TLabel", anchor="center")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        text_frame = ttk.Frame(content_frame, style="Card.TFrame", padding=10)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        ttk.Label(text_frame, text="✍️ Texto Original:", style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(0, 5))
        
        scroll_orig = ttk.Scrollbar(text_frame)
        scroll_orig.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.original_text_box = tk.Text(text_frame, height=10, width=35, bg=COLOR_CARD, fg=COLOR_FG, font=("Consolas", 10), wrap="word", padx=5, pady=5, bd=0, yscrollcommand=scroll_orig.set)
        self.original_text_box.pack(fill=tk.BOTH, expand=True)
        scroll_orig.config(command=self.original_text_box.yview)

        bottom_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=15)
        bottom_frame.pack(fill=tk.X, pady=10)
        bottom_frame.columnconfigure(1, weight=1)

        ttk.Label(bottom_frame, text="🌐 Traducir / Leer en:", style="Card.TLabel").grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        self.lang_combo = ttk.Combobox(bottom_frame, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.set("Inglés")
        self.lang_combo.grid(row=0, column=1, sticky="ew")

        self.translate_button = ttk.Button(bottom_frame, text="3. Traducir y Escuchar", style="Action.TButton", command=self.run_translation_in_thread, state="disabled")
        self.translate_button.grid(row=0, column=2, padx=(10, 0))
        
        ttk.Label(main_frame, text="💬 Resultado:", style="TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(5, 0))
        self.translated_text_box = tk.Text(main_frame, height=5, bg=COLOR_CARD, fg=COLOR_SUCCESS, font=("Consolas", 11, "bold"), wrap="word", padx=10, pady=10, bd=0)
        self.translated_text_box.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(main_frame, text="💡 Listo - Carga una imagen manuscrita", anchor="w", foreground="#795548")
        self.status_label.pack(fill=tk.X)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")])
        if not path: return
        self.image_path = path
        try:
            img_pil = Image.open(path)
            img_pil.thumbnail((400, 300)) 
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk 
            self.status_label.config(text=f"Imagen cargada: {os.path.basename(path)}")
            self.ocr_button.config(state="normal")
            self.original_text_box.delete("1.0", tk.END)
            self.translated_text_box.delete("1.0", tk.END)
            self.translate_button.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        if img.mode != 'RGB': img = img.convert('RGB')
        width, height = img.size
        scale = 2000 / max(width, height)
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        return img

    def segment_lines(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//50, 1))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20 and w > 50: lines.append((y, x, w, h))
        lines.sort(key=lambda l: l[0])
        return lines, img

    def recognize_text(self):
        if not self.image_path: return
        self.status_label.config(text="🔍 Analizando manuscrito...")
        self.update_idletasks()
        try:
            if self.use_trocr: self.recognize_with_trocr()
            else: self.recognize_with_easyocr()
        except Exception as e:
            messagebox.showerror("Error", f"Error en reconocimiento:\n{str(e)}")
            self.status_label.config(text="Error de OCR")

    def recognize_with_trocr(self):
        lines, original_img = self.segment_lines(self.image_path)
        all_text = []
        if len(lines) == 0: lines = [(0, 0, original_img.shape[1], original_img.shape[0])]
        for idx, (y, x, w, h) in enumerate(lines):
            self.status_label.config(text=f"🔍 Procesando línea {idx+1}/{len(lines)}...")
            self.update_idletasks()
            line_img = original_img[y:y+h, x:x+w]
            line_pil = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)).convert('RGB')
            pixel_values = self.processor(line_pil, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_text.append(text)
        self.original_text = "\n".join(all_text)
        self.display_result()

    def recognize_with_easyocr(self):
        img = self.preprocess_image(self.image_path)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = self.ocr_reader.readtext(img_cv, detail=1, paragraph=False)
        if results:
            results.sort(key=lambda r: r[0][0][1])
            self.original_text = "\n".join([r[1] for r in results if len(r) > 1])
        else:
            self.original_text = ""
        self.display_result()

    def display_result(self):
        if not self.original_text: self.original_text = "[No se detectó texto]"
        else: self.original_text = self.original_text.strip()
        self.original_text_box.delete("1.0", tk.END)
        self.original_text_box.insert("1.0", self.original_text)
        self.status_label.config(text="¡Reconocimiento completado!")
        self.translate_button.config(state="normal")

    def run_translation_in_thread(self):
        self.translate_button.config(state="disabled")
        threading.Thread(target=self.translate_and_speak, daemon=True).start()

    # FUNCIÓN BLINDADA PARA TRADUCCIÓN
    def robust_translate(self, text, target_lang):
        """Intenta traducir usando múltiples motores hasta que uno funcione de verdad"""
        
        # Motores a probar en orden
        providers = ['google', 'bing', 'alibaba', 'tencent']
        
        for provider in providers:
            try:
                print(f"Intentando traducir con {provider} a {target_lang}...")
                result = ts.translate_text(text, 
                                         provider=provider, 
                                         from_language='es',
                                         to_language=target_lang)
                
                # Si el resultado es idéntico al original.
                # También limpiamos espacios para comparar.
                if target_lang != 'es' and result.strip() == text.strip():
                    print(f"{provider} devolvió el mismo texto (fallo silencioso).")
                    continue 
                
                return result
                
            except Exception as e:
                print(f"Error con {provider}: {e}")
                continue 
        
        raise Exception("Todos los motores de traducción fallaron.")

    def translate_and_speak(self):
        try:
            text_to_translate = self.original_text
            if not text_to_translate or "[❌" in text_to_translate:
                self.status_label.config(text="⚠️ No hay texto para procesar.")
                return

            selected_lang_name = self.lang_combo.get()
            target_lang_code = SUPPORTED_LANGUAGES.get(selected_lang_name, "en")
            
            # Limpiar el texto
            text_clean = text_to_translate.replace("\n", " ").replace("\r", "").strip()
            
            self.status_label.config(text=f"🌐 Procesando a {selected_lang_name}...")
            
            translated_text = ""

            if target_lang_code == 'es': 
                 translated_text = text_clean
            else:
                translated_text = self.robust_translate(text_clean, target_lang_code)

            self.after(0, self.update_translated_text, translated_text)
            
            # AUDIO
            self.status_label.config(text="🔊 Generando audio...")
            filename = f"audio_{int(time.time())}.mp3"
            self.current_audio_file = filename
            
            tts = gTTS(text=translated_text, lang=target_lang_code)
            tts.save(filename)
            
            self.status_label.config(text="▶️ Reproduciendo...")
            playsound(filename)
            
            self.status_label.config(text="✅ ¡Completado!")

        except Exception as e:
            self.after(0, self.show_translation_error, e)
        
        finally:
            self.after(0, self.reactivate_translate_button)

    def update_translated_text(self, text):
        self.translated_text_box.delete("1.0", tk.END)
        self.translated_text_box.insert("1.0", text)
    
    def show_translation_error(self, e):
        messagebox.showerror("Error", f"Ocurrió un error en proceso:\n{e}")
        self.status_label.config(text="❌ Error")
        
    def reactivate_translate_button(self):
        self.translate_button.config(state="normal")
        
    def cleanup_old_files(self):
        for file in glob.glob("audio_*.mp3"):
            try: os.remove(file)
            except: pass

if __name__ == "__main__":
    app = OCRTranslatorApp()
    app.mainloop()