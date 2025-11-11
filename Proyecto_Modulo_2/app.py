import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import easyocr
import translators as ts
from gtts import gTTS
from playsound import playsound
import threading
import os
import atexit 
import cv2
import numpy as np  # <--- ¡IMPORTANTE! AÑADE ESTA LÍNEA

# --- 1. CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"
COLOR_PLACEHOLDER = "#888888"

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

# --- 3. CLASE PRINCIPAL DE LA APLICACIÓN ---

class OCRTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Módulo 2: OCR, Traducción y Voz")
        self.geometry("900x700")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        self.image_path = None
        self.original_text = ""
        self.temp_audio_file = "temp_audio.mp3"
        
        print("Cargando EasyOCR... (Puede tardar la primera vez)")
        # --- MEJORA --- Asegúrate de incluir 'es' para español.
        self.ocr_reader = easyocr.Reader(['es', 'en'], gpu=True) 
        print("EasyOCR listo.")

        # --- Estilos TTK ---
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
        
        self.create_widgets()
        atexit.register(self.cleanup)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Reconocimiento y Traducción", style="Title.TLabel").pack(pady=(0, 20))

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=10)

        self.load_button = ttk.Button(top_frame, text="1. Cargar Imagen", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.ocr_button = ttk.Button(top_frame, text="2. Reconocer Texto", command=self.recognize_text, state="disabled")
        self.ocr_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        content_frame = ttk.Frame(main_frame, style="Card.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        image_frame = ttk.Frame(content_frame, style="Card.TFrame")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.image_label = ttk.Label(image_frame, text="Carga una imagen para empezar", style="Card.TLabel", anchor="center")
        self.image_label.pack(expand=True)

        text_frame = ttk.Frame(content_frame, style="Card.TFrame")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20), pady=20)
        
        ttk.Label(text_frame, text="Texto Reconocido:", style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(0, 10))
        self.original_text_box = tk.Text(text_frame, height=10, width=40, bg=COLOR_BG, 
                                         fg=COLOR_FG, font=FONT_BODY, wrap="word",
                                         padx=10, pady=10, bd=0, highlightthickness=0)
        self.original_text_box.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        bottom_frame.columnconfigure(1, weight=1)

        ttk.Label(bottom_frame, text="Traducir a:", style="Card.TLabel").grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        self.lang_combo = ttk.Combobox(bottom_frame, 
                                       values=list(SUPPORTED_LANGUAGES.keys()), 
                                       state="readonly")
        self.lang_combo.set("Español") # Valor por defecto
        self.lang_combo.grid(row=0, column=1, sticky="ew")

        self.translate_button = ttk.Button(bottom_frame, text="3. Traducir y Hablar", 
                                           command=self.run_translation_in_thread, state="disabled")
        self.translate_button.grid(row=0, column=2, padx=(10, 0))
        
        self.translated_text_box = tk.Text(main_frame, height=5, width=60, bg=COLOR_CARD, 
                                           fg=COLOR_SUCCESS, font=FONT_BODY, wrap="word",
                                           padx=10, pady=10, bd=0, highlightthickness=0)
        self.translated_text_box.pack(fill=tk.X, pady=(5, 10))

        self.status_label = ttk.Label(main_frame, text="Estado: Listo", anchor="w")
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
            
            self.status_label.config(text=f"Imagen cargada: {os.path.basename(path)}")
            self.ocr_button.config(state="normal")
            self.original_text_box.delete("1.0", tk.END)
            self.translated_text_box.delete("1.0", tk.END)
            self.translate_button.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    # ==================================================================
    # === FUNCIÓN MODIFICADA (VERSIÓN 3) ===
    # ==================================================================
    def recognize_text(self):
        """Usa EasyOCR con pre-procesamiento agresivo para fuentes finas."""
        if not self.image_path:
            return
            
        self.status_label.config(text="Procesando OCR (Pre-procesamiento agresivo)...")
        self.update_idletasks() 

        try:
            # --- INICIO DE LA MEJORA DE PROCESAMIENTO ---
            
            # 1. Cargar la imagen con OpenCV
            image = cv2.imread(self.image_path)
            
            # 2. Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 3. Aplicar Umbral Adaptativo (Adaptive Thresholding)
            # Esto es mucho mejor que el umbral global para fuentes complejas.
            # Crea texto BLANCO (255) sobre fondo NEGRO (0).
            adaptive_bw = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Método de "vecindario"
                cv2.THRESH_BINARY_INV,          # Invertir (texto en blanco)
                15, # Tamaño del bloque (ajustar si es necesario)
                4   # Constante C (ajustar si es necesario)
            )

            # 4. "Engordar" el texto (Dilation)
            # ¡Este es el paso más importante para fuentes finas!
            # Creamos un "kernel" (un pequeño cuadrado)
            kernel = np.ones((2, 2), np.uint8) 
            # Aplicamos la dilatación para "engordar" el texto blanco
            dilated_image = cv2.dilate(adaptive_bw, kernel, iterations=1)

            # 5. Invertir la imagen para EasyOCR
            # EasyOCR prefiere texto NEGRO sobre fondo BLANCO.
            final_image = cv2.bitwise_not(dilated_image)
            
            # --- FIN DE LA MEJORA DE PROCESAMIENTO ---
            
            # Pasamos la imagen final (array de numpy) a EasyOCR
            results = self.ocr_reader.readtext(final_image)
            
            self.original_text = " ".join([res[1] for res in results])
            
            if not self.original_text:
                self.original_text = "[No se detectó texto]"
                
            self.original_text_box.delete("1.0", tk.END)
            self.original_text_box.insert("1.0", self.original_text)
            
            self.status_label.config(text="¡Reconocimiento completado!")
            self.translate_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error de OCR", f"Ocurrió un error: {e}")
            self.status_label.config(text="Error de OCR", foreground=COLOR_ERROR)
    # ==================================================================
    # === FIN DE LA MODIFICACIÓN ===
    # ==================================================================

    def run_translation_in_thread(self):
        self.translate_button.config(state="disabled")
        threading.Thread(target=self.translate_and_speak, daemon=True).start()

    def translate_and_speak(self):
        try:
            text_to_translate = self.original_text
            if not text_to_translate or text_to_translate == "[No se detectó texto]":
                self.status_label.config(text="No hay texto para traducir.")
                self.translate_button.config(state="normal") 
                return
            
            selected_lang_name = self.lang_combo.get()
            target_lang_code = SUPPORTED_LANGUAGES.get(selected_lang_name, "en")

            self.status_label.config(text=f"Traduciendo a {selected_lang_name}...")
            
            translated_text = ts.translate_text(text_to_translate, 
                                                from_language='auto', 
                                                to_language=target_lang_code)
            
            self.after(0, self.update_translated_text, translated_text)
            
            self.status_label.config(text="Generando audio...")
            
            tts = gTTS(text=translated_text, lang=target_lang_code)
            tts.save(self.temp_audio_file)
            
            self.status_label.config(text="Reproduciendo...")
            playsound(self.temp_audio_file)
            
            self.status_label.config(text="¡Completado!")

        except Exception as e:
            self.after(0, self.show_translation_error, e)
        
        finally:
            self.after(0, self.reactivate_translate_button)

    def update_translated_text(self, text):
        self.translated_text_box.delete("1.0", tk.END)
        self.translated_text_box.insert("1.0", text)
    
    def show_translation_error(self, e):
        messagebox.showerror("Error de Traducción/Voz", 
                             f"Ocurrió un error. Revisa tu conexión a internet.\nDetalle: {e}")
        self.status_label.config(text="Error en la traducción")
        
    def reactivate_translate_button(self):
        self.translate_button.config(state="normal")
        
    def cleanup(self):
        if os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
                print(f"Archivo temporal '{self.temp_audio_file}' eliminado.")
            except PermissionError:
                print(f"No se pudo eliminar '{self.temp_audio_file}', el archivo está en uso.")

# --- 4. EJECUTAR LA APLICACIÓN ---
if __name__ == "__main__":
    app = OCRTranslatorApp()
    app.mainloop()