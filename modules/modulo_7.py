import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os

# --- 1. CONFIGURACIÓN DE ESTILO (Reutilizado) ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_SUCCESS = "#34c759"
COLOR_ERROR = "#ff3b30"

FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")
FONT_STATUS = ("SF Pro Text", 13, "bold")
FONT_RESULT = ("SF Pro Display", 16, "bold")

# --- 2. CLASE PRINCIPAL DE LA APLICACIÓN ---

class GenreClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.model = None
        self.mlb = None
        
        # --- Configuración de la Ventana ---
        self.title("Módulo 7: Clasificador de Géneros de Películas")
        self.geometry("700x550")
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        
        # --- Estilos TTK ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0, lightcolor=COLOR_CARD, darkcolor=COLOR_CARD)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        self.style.configure("Result.TLabel", font=FONT_RESULT, background=COLOR_CARD, foreground=COLOR_ACCENT, anchor="center")
        self.style.configure("TButton", font=FONT_BODY_BOLD, background=COLOR_ACCENT, foreground=COLOR_FG, borderwidth=0, padding=(15, 10), relief="flat")
        self.style.map("TButton", background=[("active", "#0056b3"), ("pressed", "#0056b3")])
        
        # --- Cargar Modelo ---
        self.load_model()
        
        # --- Creación de Widgets ---
        self.create_widgets()
        
    def load_model(self):
        """Carga el modelo entrenado y el binarizer desde los archivos."""
        model_path = 'modelo_generos.joblib'
        mlb_path = 'binarizer_generos.joblib'
        
        if not os.path.exists(model_path) or not os.path.exists(mlb_path):
            messagebox.showerror("Error de Archivos", 
                                 "No se encontraron 'modelo_generos.joblib' o 'binarizer_generos.joblib'.\n"
                                 "Por favor, ejecuta el script 'entrenar_modelo_7.py' primero.")
            self.after(100, self.destroy)
            return
        
        try:
            self.model = joblib.load(model_path)
            self.mlb = joblib.load(mlb_path)
            print("Modelo y Binarizer cargados correctamente.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudieron cargar los modelos: {e}")
            self.after(100, self.destroy)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Clasificador de Géneros", style="Title.TLabel").pack(pady=(0, 20))
        
        # --- Tarjeta de Entrada ---
        input_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        input_card.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(input_card, text="Pega la sinopsis o trama de la película aquí:", 
                  style="Card.TLabel", font=FONT_BODY_BOLD).pack(anchor="w", pady=(0, 10))
        
        # Frame para el Text widget con borde
        text_frame = ttk.Frame(input_card, style="Card.TFrame", borderwidth=1, relief="solid")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.plot_entry = tk.Text(text_frame, 
                                  height=10, 
                                  width=60, 
                                  bg=COLOR_CARD, 
                                  fg=COLOR_FG, 
                                  font=FONT_BODY,
                                  padx=10, 
                                  pady=10,
                                  bd=0,
                                  highlightthickness=0,
                                  insertbackground=COLOR_FG,
                                  wrap="word")
        self.plot_entry.pack(fill=tk.BOTH, expand=True)
        
        self.classify_button = ttk.Button(main_frame, text="Clasificar Géneros", command=self.classify_plot)
        self.classify_button.pack(pady=20, fill="x")
        
        # --- Tarjeta de Resultado ---
        result_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        result_card.pack(fill="x")
        
        self.result_label = ttk.Label(result_card, text="Géneros: ...", style="Result.TLabel")
        self.result_label.pack(fill="x")
        
        # --- Estado ---
        self.status_label = ttk.Label(main_frame, text="Listo para clasificar", style="Status.TLabel", anchor="center")
        self.status_label.pack(fill=tk.X, pady=(15, 0))
        
        if self.model is None:
             self.status_label.config(text="Error al cargar modelos", foreground=COLOR_ERROR)
             self.classify_button.config(state="disabled")

    def classify_plot(self):
        """Toma el texto del widget, lo predice y muestra el resultado."""
        plot_text = self.plot_entry.get("1.0", tk.END).strip()
        
        if not plot_text:
            self.status_label.config(text="Error: El campo de texto está vacío", foreground=COLOR_ERROR)
            return
            
        if self.model is None or self.mlb is None:
            self.status_label.config(text="Error: Los modelos no están cargados", foreground=COLOR_ERROR)
            return

        try:
            self.status_label.config(text="Clasificando...", foreground=COLOR_ACCENT)
            self.update_idletasks() # Forzar actualización de la GUI
            
            # El modelo espera una lista o iterable, no un solo string
            plot_text_list = [plot_text]
            
            # 1. Predecir (da un resultado binario, ej: [0, 1, 0, 1, ...])
            binary_prediction = self.model.predict(plot_text_list)
            
            # 2. Decodificar (convierte el binario en etiquetas de texto)
            #    Usamos inverse_transform del binarizer que guardamos
            genres_tuple = self.mlb.inverse_transform(binary_prediction)
            
            # El resultado es una tupla con una lista, ej: (['Action', 'Drama'],)
            if genres_tuple and genres_tuple[0]:
                # Unimos los géneros con "/" como en tu ejemplo
                genres_str = " / ".join(genres_tuple[0])
                self.result_label.config(text=f"Géneros: {genres_str}", foreground=COLOR_SUCCESS)
            else:
                self.result_label.config(text="Géneros: No se pudo determinar", foreground=COLOR_FG)

            self.status_label.config(text="¡Clasificación completa!", foreground=COLOR_SUCCESS)

        except Exception as e:
            self.status_label.config(text=f"Error en predicción: {e}", foreground=COLOR_ERROR)


# --- 3. EJECUTAR LA APLICACIÓN ---
if __name__ == "__main__":
    app = GenreClassifierApp()
    app.mainloop()