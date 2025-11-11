import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import translators as ts
from gtts import gTTS
from playsound import playsound
import threading
import os
import atexit 
import cv2
import numpy as np
import warnings

# Suprimir advertencias
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports para TrOCR (modelo avanzado de manuscritos)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
    print("✅ TrOCR disponible (modelo avanzado)")
except ImportError:
    TROCR_AVAILABLE = False
    print("⚠️ TrOCR no disponible, usando EasyOCR")
    import easyocr

# --- 1. CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")

# --- 2. DICCIONARIO DE IDIOMAS ---
SUPPORTED_LANGUAGES = {
    "Inglés": "en",
    "Español": "es",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Portugués": "pt",
    "Japonés": "ja",
    "Chino (Simplificado)": "zh-CN",
    "Ruso": "ru"
}

# --- 3. CLASE PRINCIPAL ---

class OCRTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Módulo 2: OCR Manuscrito Avanzado")
        self.geometry("950x700")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        self.image_path = None
        self.original_text = ""
        self.temp_audio_file = "temp_audio.mp3"
        
        # ✅ CARGAR MODELO SEGÚN DISPONIBILIDAD
        if TROCR_AVAILABLE:
            print("🔄 Cargando TrOCR para manuscritos (puede tardar la primera vez)...")
            try:
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                print(f"✅ TrOCR cargado en {self.device}")
                self.use_trocr = True
            except Exception as e:
                print(f"⚠️ Error cargando TrOCR: {e}, usando EasyOCR")
                self.ocr_reader = easyocr.Reader(['es'], gpu=False, verbose=False)
                self.use_trocr = False
        else:
            print("🔄 Cargando EasyOCR...")
            self.ocr_reader = easyocr.Reader(['es'], gpu=False, verbose=False)
            self.use_trocr = False
            print("✅ EasyOCR listo")

        self.setup_styles()
        self.create_widgets()
        atexit.register(self.cleanup)

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT, foreground=COLOR_FG, borderwidth=0, padding=(15, 10))
        self.style.map("TButton", background=[("active", "#0056b3")])
        self.style.configure("TCombobox", fieldbackground=COLOR_CARD, background=COLOR_CARD, foreground=COLOR_FG, arrowcolor=COLOR_FG)
        self.style.map('TCombobox', fieldbackground=[('readonly', COLOR_CARD)])

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        model_name = "TrOCR (Microsoft)" if self.use_trocr else "EasyOCR"
        ttk.Label(main_frame, text=f"📝 Reconocimiento Manuscrito ({model_name})", style="Title.TLabel").pack(pady=(0, 20))

        # Botones principales
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=10)

        self.load_button = ttk.Button(top_frame, text="1. Cargar Imagen", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.ocr_button = ttk.Button(top_frame, text="2. Reconocer Manuscrito", command=self.recognize_text, state="disabled")
        self.ocr_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Área de contenido
        content_frame = ttk.Frame(main_frame, style="Card.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Imagen
        image_frame = ttk.Frame(content_frame, style="Card.TFrame")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.image_label = ttk.Label(image_frame, text="📷 Carga una imagen manuscrita", style="Card.TLabel", anchor="center")
        self.image_label.pack(expand=True)

        # Texto reconocido
        text_frame = ttk.Frame(content_frame, style="Card.TFrame")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20), pady=20)
        
        ttk.Label(text_frame, text="✍️ Texto Reconocido:", style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(0, 10))
        self.original_text_box = tk.Text(text_frame, height=10, width=40, bg=COLOR_BG, 
                                         fg=COLOR_FG, font=FONT_BODY, wrap="word",
                                         padx=10, pady=10, bd=0, highlightthickness=0)
        self.original_text_box.pack(fill=tk.BOTH, expand=True)

        # Traducción
        bottom_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        bottom_frame.pack(fill=tk.X, pady=10)
        bottom_frame.columnconfigure(1, weight=1)

        ttk.Label(bottom_frame, text="🌐 Traducir a:", style="Card.TLabel").grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        self.lang_combo = ttk.Combobox(bottom_frame, values=list(SUPPORTED_LANGUAGES.keys()), state="readonly")
        self.lang_combo.set("Inglés")
        self.lang_combo.grid(row=0, column=1, sticky="ew")

        self.translate_button = ttk.Button(bottom_frame, text="3. Traducir y Hablar", 
                                           command=self.run_translation_in_thread, state="disabled")
        self.translate_button.grid(row=0, column=2, padx=(10, 0))
        
        self.translated_text_box = tk.Text(main_frame, height=5, width=60, bg=COLOR_CARD, 
                                           fg=COLOR_SUCCESS, font=FONT_BODY, wrap="word",
                                           padx=10, pady=10, bd=0, highlightthickness=0)
        self.translated_text_box.pack(fill=tk.X, pady=(5, 10))

        self.status_label = ttk.Label(main_frame, text="💡 Listo - Carga una imagen", anchor="w")
        self.status_label.pack(fill=tk.X)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        
        try:
            img_pil = Image.open(path)
            img_pil.thumbnail((350, 350)) 
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk 
            
            self.status_label.config(text=f"✅ Imagen cargada: {os.path.basename(path)}")
            self.ocr_button.config(state="normal")
            self.original_text_box.delete("1.0", tk.END)
            self.translated_text_box.delete("1.0", tk.END)
            self.translate_button.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    def preprocess_image(self, image_path):
        """Preprocesamiento optimizado"""
        img = Image.open(image_path)
        
        # Convertir a RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Escalar a tamaño óptimo
        width, height = img.size
        scale = 2000 / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Mejorar contraste automáticamente
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Mejorar nitidez
        img = img.filter(ImageFilter.SHARPEN)
        
        return img

    def segment_lines(self, image_path):
        """Segmentar imagen en líneas de texto"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarizar
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Encontrar contornos (líneas de texto)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//50, 1))
        dilated = cv2.dilate(binary, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar líneas de arriba a abajo
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 20 and w > 50:  # Filtrar ruido
                lines.append((y, x, w, h))
        
        lines.sort(key=lambda l: l[0])  # Ordenar por Y
        return lines, img

    def recognize_text(self):
        """Reconocimiento con el modelo cargado"""
        if not self.image_path:
            return
            
        self.status_label.config(text="🔍 Analizando manuscrito...")
        self.update_idletasks()

        try:
            if self.use_trocr:
                self.recognize_with_trocr()
            else:
                self.recognize_with_easyocr()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en reconocimiento:\n{str(e)}")
            self.status_label.config(text="❌ Error de OCR")

    def recognize_with_trocr(self):
        """Reconocimiento con TrOCR (más preciso para manuscritos)"""
        lines, original_img = self.segment_lines(self.image_path)
        
        all_text = []
        total_lines = len(lines)
        
        for idx, (y, x, w, h) in enumerate(lines):
            self.status_label.config(text=f"🔍 Procesando línea {idx+1}/{total_lines}...")
            self.update_idletasks()
            
            # Extraer línea
            line_img = original_img[y:y+h, x:x+w]
            line_pil = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
            
            # Preprocesar
            line_pil = line_pil.convert('RGB')
            
            # Reconocer con TrOCR
            pixel_values = self.processor(line_pil, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            all_text.append(text)
        
        self.original_text = "\n".join(all_text)
        self.display_result()

    def recognize_with_easyocr(self):
        """Reconocimiento con EasyOCR mejorado"""
        img = self.preprocess_image(self.image_path)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        results = self.ocr_reader.readtext(
            img_cv,
            detail=1,
            paragraph=False,
            decoder='greedy',
            batch_size=1
        )
        
        if results:
            # Ordenar por posición Y
            results.sort(key=lambda r: r[0][0][1])
            text_lines = [r[1] for r in results if len(r) > 1]
            self.original_text = "\n".join(text_lines)
        else:
            self.original_text = ""
        
        self.display_result()

    def display_result(self):
        """Mostrar resultado"""
        if not self.original_text:
            self.original_text = "[❌ No se detectó texto]"
        else:
            self.original_text = self.original_text.strip()
            
        self.original_text_box.delete("1.0", tk.END)
        self.original_text_box.insert("1.0", self.original_text)
        
        self.status_label.config(text="✅ ¡Reconocimiento completado!")
        self.translate_button.config(state="normal")

    def run_translation_in_thread(self):
        self.translate_button.config(state="disabled")
        threading.Thread(target=self.translate_and_speak, daemon=True).start()

    def translate_and_speak(self):
        try:
            text_to_translate = self.original_text
            if not text_to_translate or "[❌" in text_to_translate:
                self.status_label.config(text="⚠️ No hay texto para traducir.")
                self.translate_button.config(state="normal") 
                return
            
            selected_lang_name = self.lang_combo.get()
            target_lang_code = SUPPORTED_LANGUAGES.get(selected_lang_name, "en")

            self.status_label.config(text=f"🌐 Traduciendo a {selected_lang_name}...")
            
            translated_text = ts.translate_text(text_to_translate, 
                                                from_language='es', 
                                                to_language=target_lang_code)
            
            self.after(0, self.update_translated_text, translated_text)
            
            self.status_label.config(text="🔊 Generando audio...")
            
            tts = gTTS(text=translated_text, lang=target_lang_code)
            tts.save(self.temp_audio_file)
            
            self.status_label.config(text="▶️ Reproduciendo...")
            playsound(self.temp_audio_file)
            
            self.status_label.config(text="✅ ¡Completado!")

        except Exception as e:
            self.after(0, self.show_translation_error, e)
        
        finally:
            self.after(0, self.reactivate_translate_button)

    def update_translated_text(self, text):
        self.translated_text_box.delete("1.0", tk.END)
        self.translated_text_box.insert("1.0", text)
    
    def show_translation_error(self, e):
        messagebox.showerror("Error de Traducción", f"Ocurrió un error:\n{e}")
        self.status_label.config(text="❌ Error en la traducción")
        
    def reactivate_translate_button(self):
        self.translate_button.config(state="normal")
        
    def cleanup(self):
        if os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass

if __name__ == "__main__":
    app = OCRTranslatorApp()
    app.mainloop()