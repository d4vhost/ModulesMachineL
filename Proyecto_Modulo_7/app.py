import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os 
import translators as ts 

# --- 1. CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"
COLOR_WARNING = "#ff9500"
COLOR_PLACEHOLDER = "#888888" # Color para el texto guía

FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")
FONT_STATUS = ("SF Pro Text", 12, "bold")
FONT_RESULT = ("SF Pro Display", 15, "bold")

# ==================================================================
# --- 2. DICCIONARIO DE TRADUCCIÓN DE GÉNEROS ---
# (Traduce los resultados del modelo al español)
# ==================================================================
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
    # Puedes añadir más si descubres nuevos géneros
}
# ==================================================================

# Rutas a los archivos (sin cambios)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_generos.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "binarizer_generos.joblib")

# --- 3. CLASE PRINCIPAL DE LA APLICACIÓN ---

class GenreClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.mlb = None
        
        # --- 4. MEJORA UI: Texto Guía (Placeholder) ---
        self.placeholder_text = "Pega aquí la sinopsis en español..."
        
        # --- Configuración de la Ventana ---
        self.title("Módulo 7: Clasificador de Géneros de Películas")
        self.geometry("700x540")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        # --- Estilos TTK (sin cambios) ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0, lightcolor=COLOR_CARD, darkcolor=COLOR_CARD)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        self.style.configure("Result.TLabel", font=FONT_RESULT, background=COLOR_CARD, foreground=COLOR_ACCENT)
        self.style.configure("TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT, foreground=COLOR_FG, borderwidth=0, padding=(15, 10), relief="flat")
        self.style.map("TButton", background=[("active", "#0056b3"), ("pressed", "#0056b3")])
        
        self.load_model()
        self.create_widgets()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(MLB_PATH):
            messagebox.showerror("Error de Archivos", 
                                 f"No se encontraron los archivos de modelo en '{MODEL_DIR}'.\n"
                                 "Por favor, ejecuta el script 'entrenar.py' primero.")
            self.after(100, self.destroy)
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.mlb = joblib.load(MLB_PATH)
            print("Modelo y Binarizer cargados correctamente.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudieron cargar los modelos: {e}")
            self.after(100, self.destroy)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Clasificador de Géneros", style="Title.TLabel").pack(pady=(0, 20))
        
        input_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        input_card.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(input_card, text="Pega la sinopsis o trama de la película aquí:", 
                  style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(0, 10))
        
        text_frame = ttk.Frame(input_card, style="Card.TFrame", borderwidth=1, relief="solid")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.plot_entry = tk.Text(text_frame, 
                                  height=10, width=60, bg=COLOR_CARD, 
                                  fg=COLOR_FG, font=FONT_BODY,
                                  padx=10, pady=10, bd=0,
                                  highlightthickness=0,
                                  insertbackground=COLOR_FG,
                                  wrap="word")
        self.plot_entry.pack(fill=tk.BOTH, expand=True)
        
        # --- 5. MEJORA UI: Lógica del Placeholder ---
        self.plot_entry.insert("1.0", self.placeholder_text)
        self.plot_entry.config(foreground=COLOR_PLACEHOLDER)
        self.plot_entry.bind("<FocusIn>", self.on_entry_click)
        self.plot_entry.bind("<FocusOut>", self.on_entry_leave)
        # --- Fin de Mejora UI ---
        
        self.classify_button = ttk.Button(main_frame, text="Clasificar Géneros", command=self.classify_plot)
        self.classify_button.pack(pady=20, fill="x")
        
        result_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        result_card.pack(fill="x")
        
        self.result_label = ttk.Label(result_card, text="Géneros: ...", style="Result.TLabel", wraplength=600)
        self.result_label.pack(fill="x")
        
        self.status_label = ttk.Label(main_frame, text="Listo para clasificar", style="Status.TLabel", anchor="center")
        self.status_label.pack(fill=tk.X, pady=(15, 0))
        
        if self.model is None:
             self.status_label.config(text="Error al cargar modelos", foreground=COLOR_ERROR)
             self.classify_button.config(state="disabled")

    # --- 6. MEJORA UI: Funciones para el Placeholder ---
    def on_entry_click(self, event):
        """Borra el placeholder cuando el usuario hace clic."""
        current_text = self.plot_entry.get("1.0", tk.END).strip()
        if current_text == self.placeholder_text:
            self.plot_entry.delete("1.0", tk.END)
            self.plot_entry.config(foreground=COLOR_FG)

    def on_entry_leave(self, event):
        """Restaura el placeholder si el campo está vacío."""
        current_text = self.plot_entry.get("1.0", tk.END).strip()
        if not current_text:
            self.plot_entry.insert("1.0", self.placeholder_text)
            self.plot_entry.config(foreground=COLOR_PLACEHOLDER)
    # --- Fin de Mejora UI ---

    # --- 7. LÓGICA DE CLASIFICACIÓN ACTUALIZADA ---
    def classify_plot(self):
        """Toma el texto en ESPAÑOL, lo traduce, predice y traduce el resultado."""
        
        plot_text_es = self.plot_entry.get("1.0", tk.END).strip()
        
        # Validación de entrada
        if not plot_text_es or plot_text_es == self.placeholder_text:
            self.status_label.config(text="Error: El campo de texto está vacío", foreground=COLOR_ERROR)
            return
            
        if self.model is None or self.mlb is None:
            self.status_label.config(text="Error: Los modelos no están cargados", foreground=COLOR_ERROR)
            return

        try:
            # --- PASO A: TRADUCIR (ES -> EN) ---
            self.status_label.config(text="Traduciendo sinopsis...", foreground=COLOR_ACCENT)
            self.update_idletasks()
            
            # (Usamos 'google' como motor de traducción)
            plot_text_en = ts.translate_text(plot_text_es, from_language='es', to_language='en')
            
            self.status_label.config(text="Clasificando...", foreground=COLOR_ACCENT)
            self.update_idletasks() 
            
            # --- PASO B: PREDECIR (en inglés) ---
            plot_text_list = [plot_text_en]
            binary_prediction = self.model.predict(plot_text_list)
            genres_tuple_en = self.mlb.inverse_transform(binary_prediction)
            
            if genres_tuple_en and genres_tuple_en[0]:
                
                # --- PASO C: TRADUCIR RESULTADO (EN -> ES) ---
                genres_list_es = []
                for genre_en in genres_tuple_en[0]:
                    # Usamos el diccionario. Si no encuentra, usa el original en inglés.
                    genres_list_es.append(GENRE_TRANSLATIONS.get(genre_en, genre_en))
                
                genres_str_es = " / ".join(genres_list_es)
                self.result_label.config(text=f"Géneros: {genres_str_es}", foreground=COLOR_SUCCESS)
                self.status_label.config(text="¡Clasificación completa!", foreground=COLOR_SUCCESS)
            else:
                self.result_label.config(text="Géneros: No se pudo determinar", foreground=COLOR_WARNING)
                self.status_label.config(text="La sinopsis no arrojó un género claro.", foreground=COLOR_WARNING)

        except Exception as e:
            # Error común: Sin internet para traducir
            print(f"Error en traducción/predicción: {e}")
            self.status_label.config(text="Error: No se pudo traducir. Revisa tu conexión.", foreground=COLOR_ERROR)
            self.result_label.config(text="Géneros: ...", foreground=COLOR_ACCENT)


# --- 4. EJECUTAR LA APLICACIÓN ---
if __name__ == "__main__":
    app = GenreClassifierApp()
    app.mainloop()