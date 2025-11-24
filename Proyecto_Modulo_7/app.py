import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os 
import translators as ts 
import threading

COLOR_BG = "#f5f7fa"
COLOR_CARD = "#ffffff"
COLOR_FG = "#2c3e50"
COLOR_ACCENT = "#3498db"
COLOR_SUCCESS = "#27ae60"
COLOR_ERROR = "#e74c3c"
COLOR_WARNING = "#f39c12"
COLOR_PLACEHOLDER = "#95a5a6"
COLOR_BORDER = "#e1e8ed"
COLOR_BUTTON_HOVER = "#2980b9"

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUBTITLE = ("Segoe UI", 11)
FONT_BODY = ("Segoe UI", 11)
FONT_BODY_BOLD = ("Segoe UI", 11, "bold")
FONT_STATUS = ("Segoe UI", 10)
FONT_RESULT = ("Segoe UI", 14, "bold")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_generos.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "binarizer_generos.joblib")

GENRE_TRANSLATIONS = {
    "Action & Adventure": "Acción y Aventura",
    "Comedies": "Comedias",
    "Dramas": "Dramas",
    "Horror Movies": "Películas de Terror",
    "Thrillers": "Suspenso",
    "Documentaries": "Documentales",
    "Sci-Fi & Fantasy": "Ciencia Ficción y Fantasía",
    "Romantic Movies": "Películas Románticas",
    "Children & Family Movies": "Infantiles y Familiares",
    "Stand-Up Comedy": "Comedia en Vivo",
    "International Movies": "Películas Internacionales",
    "Classic Movies": "Películas Clásicas",
    "Music & Musicals": "Música y Musicales",
    "Anime Features": "Anime",
    "Sports Movies": "Películas de Deporte",
    "LGBTQ Movies": "Películas LGBTQ",
    "Cult Movies": "Películas de Culto",
}

class GenreClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.mlb = None
        self.placeholder_text = "Escribe o pega aquí la sinopsis de tu película..."
        
        self.title("Clasificador de Géneros de Películas")

        window_width = 800
        window_height = 650
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        self.configure_styles()
        self.load_model()
        self.create_widgets()
        
    def configure_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Subtitle.TLabel", font=FONT_SUBTITLE, background=COLOR_BG, foreground=COLOR_PLACEHOLDER)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG, foreground=COLOR_PLACEHOLDER)
        self.style.configure("Accent.TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT, foreground="white", borderwidth=0, padding=(20, 12), relief="flat")
        self.style.map("Accent.TButton", background=[("active", COLOR_BUTTON_HOVER), ("pressed", COLOR_BUTTON_HOVER)])
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MLB_PATH):
            messagebox.showerror("Error", "Modelos no encontrados. Ejecuta 'entrenar.py'.")
            self.destroy()
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.mlb = joblib.load(MLB_PATH)
            print("✓ Modelos cargados.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelos: {e}")
            self.destroy()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=40)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Header
        ttk.Label(main_frame, text="🎬 Clasificador de Géneros", style="Title.TLabel").pack()
        ttk.Label(main_frame, text="Descubre el género con IA", style="Subtitle.TLabel").pack(pady=(5, 20))
        
        # Input Area
        input_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        input_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Borde falso
        border = tk.Frame(input_card, bg=COLOR_BORDER, padx=1, pady=1)
        border.pack(fill=tk.BOTH, expand=True)
        
        text_area = tk.Frame(border, bg=COLOR_CARD)
        text_area.pack(fill=tk.BOTH, expand=True)
        
        self.plot_entry = tk.Text(text_area, height=8, bg=COLOR_CARD, fg=COLOR_FG, font=FONT_BODY,
                                  padx=15, pady=15, bd=0, highlightthickness=0, wrap="word")
        self.plot_entry.pack(fill=tk.BOTH, expand=True)
        
        self.plot_entry.insert("1.0", self.placeholder_text)
        self.plot_entry.config(foreground=COLOR_PLACEHOLDER)
        self.plot_entry.bind("<FocusIn>", self.on_entry_click)
        self.plot_entry.bind("<FocusOut>", self.on_entry_leave)
        
        # Botón
        self.classify_button = ttk.Button(main_frame, text="🔍 Clasificar", command=self.run_classification_thread, style="Accent.TButton")
        self.classify_button.pack(fill=tk.X, pady=(0, 20))
        
        # Resultados
        result_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        result_card.pack(fill=tk.X)
        
        ttk.Label(result_card, text="Resultado:", style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w")
        
        self.result_label = tk.Label(result_card, text="Esperando...", bg=COLOR_CARD, fg=COLOR_PLACEHOLDER,
                                     font=FONT_RESULT, wraplength=700, justify="left", anchor="w")
        self.result_label.pack(fill="x", pady=(5, 0))
        
        self.status_label = ttk.Label(main_frame, text="● Listo", style="Status.TLabel")
        self.status_label.pack(pady=(15, 0))

    def on_entry_click(self, event):
        if self.plot_entry.get("1.0", tk.END).strip() == self.placeholder_text:
            self.plot_entry.delete("1.0", tk.END)
            self.plot_entry.config(foreground=COLOR_FG)

    def on_entry_leave(self, event):
        if not self.plot_entry.get("1.0", tk.END).strip():
            self.plot_entry.insert("1.0", self.placeholder_text)
            self.plot_entry.config(foreground=COLOR_PLACEHOLDER)

    # LÓGICA DE HILOS 
    def run_classification_thread(self):
        plot_text = self.plot_entry.get("1.0", tk.END).strip()
        if not plot_text or plot_text == self.placeholder_text:
            self.status_label.config(text="✖ El campo está vacío", foreground=COLOR_ERROR)
            return
            
        # Deshabilitar botón mientras procesa
        self.classify_button.config(state="disabled")
        self.status_label.config(text="⏳ Procesando...", foreground=COLOR_ACCENT)
        self.result_label.config(text="Traduciendo y analizando...", fg=COLOR_PLACEHOLDER)
        
        threading.Thread(target=self.classify_process, args=(plot_text,), daemon=True).start()

    def classify_process(self, text_es):
        try:
            print("Iniciando traducción")
            text_en = ts.translate_text(text_es, translator='google', from_language='es', to_language='en')
            print(f"Traducción OK: {text_en[:30]}...")
            prediction = self.model.predict([text_en])
            genres = self.mlb.inverse_transform(prediction)
            
            # Llamar a actualización de UI
            self.after(0, self.update_ui_success, genres)
            
        except Exception as e:
            print(f"Error: {e}")
            self.after(0, self.update_ui_error, str(e))

    def update_ui_success(self, genres_tuple):
        self.classify_button.config(state="normal")
        
        if genres_tuple and genres_tuple[0]:
            genres_es = [GENRE_TRANSLATIONS.get(g, g) for g in genres_tuple[0]]
            final_text = "  •  ".join(genres_es)
            
            self.result_label.config(text=final_text, fg=COLOR_SUCCESS)
            self.status_label.config(text="✓ Clasificación exitosa", foreground=COLOR_SUCCESS)
        else:
            self.result_label.config(text="No se pudo determinar el género", fg=COLOR_WARNING)
            self.status_label.config(text="⚠ Intenta con más detalles", foreground=COLOR_WARNING)

    def update_ui_error(self, error_msg):
        self.classify_button.config(state="normal")
        self.status_label.config(text="✖ Error", foreground=COLOR_ERROR)
        self.result_label.config(text="Error de conexión o traducción.", fg=COLOR_ERROR)

if __name__ == "__main__":
    app = GenreClassifierApp()
    app.mainloop()