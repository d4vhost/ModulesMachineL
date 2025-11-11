import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO
import os

# --- 1. CONFIGURACIÓN DE ESTILO (Reutiliza tu estilo del Módulo 3/7) ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_BODY_BOLD = ("SF Pro Text", 12, "bold")

# --- 2. CLASE PRINCIPAL DE LA APLICACIÓN ---

class OlimpiadasApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # --- Configuración de la Ventana ---
        self.title("Módulo 1: Análisis del Medallero Olímpico")
        self.geometry("1000x800") # Más grande para mostrar tablas y gráficos
        self.configure(bg=COLOR_BG)
        
        # --- Variables de Estado ---
        self.df_olympics = None # DataFrame principal
        self.all_years = []
        self.all_countries = []
        
        # --- Ruta de datos ---
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # ¡IMPORTANTE! Asegúrate que tu CSV se llame 'athlete_events.csv'
        # Si lo llamaste 'olympic_history.csv', cambia el nombre aquí.
        self.DATA_PATH = os.path.join(BASE_DIR, "data", "athlete_events.csv")
        
        # --- Estilos TTK ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD, borderwidth=0, lightcolor=COLOR_CARD, darkcolor=COLOR_CARD)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_FG)
        
        self.style.configure("TCombobox",
                             fieldbackground=COLOR_CARD,
                             background=COLOR_CARD,
                             foreground=COLOR_FG,
                             arrowcolor=COLOR_FG,
                             bordercolor=COLOR_BG,
                             lightcolor=COLOR_CARD,
                             darkcolor=COLOR_CARD,
                             insertcolor=COLOR_FG,
                             selectbackground=COLOR_BG,
                             selectforeground=COLOR_FG)
        self.style.map('TCombobox', fieldbackground=[('readonly', COLOR_CARD)],
                       selectbackground=[('readonly', COLOR_BG)],
                       foreground=[('readonly', COLOR_FG)])

        self.style.configure("TButton",
                             font=FONT_BODY_BOLD,
                             background=COLOR_ACCENT,
                             foreground=COLOR_FG,
                             borderwidth=0,
                             padding=(15, 10),
                             relief="flat")
        self.style.map("TButton",
                       background=[("active", "#0056b3"), ("pressed", "#0056b3")])
        
        # Configurar estilo para Notebook y Treeview
        self.style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        self.style.configure("TNotebook.Tab",
                             background=COLOR_BG,
                             foreground=COLOR_FG,
                             padding=[10, 5],
                             font=FONT_BODY_BOLD)
        self.style.map("TNotebook.Tab",
                       background=[("selected", COLOR_CARD)],
                       foreground=[("selected", COLOR_ACCENT)])

        self.style.configure("Treeview",
                             background=COLOR_CARD,
                             foreground=COLOR_FG,
                             fieldbackground=COLOR_CARD,
                             rowheight=25,
                             font=FONT_BODY)
        self.style.configure("Treeview.Heading",
                             background=COLOR_BG,
                             foreground=COLOR_ACCENT,
                             font=FONT_BODY_BOLD,
                             padding=5)
        self.style.map("Treeview.Heading",
                       background=[("active", COLOR_CARD)])


        # --- Cargar datos ---
        self.load_data()
        
        # --- Crear Widgets ---
        if self.df_olympics is not None:
            self.create_widgets()
        else:
            self.show_error_loading()

    def load_data(self):
        """Carga y procesa inicialmente el CSV de olimpiadas."""
        try:
            self.df_olympics = pd.read_csv(self.DATA_PATH)
            
            # --- Limpieza Básica de Datos ---
            # 1. Nos quedamos solo con medallas (excluimos 'NA')
            self.df_olympics = self.df_olympics.dropna(subset=['Medal'])
            
            # 2. Obtenemos listas para los filtros
            self.all_years = sorted(self.df_olympics['Year'].unique())
            
            # === ¡ARREGLO 1! ===
            # Usamos 'NOC' en lugar de 'region'
            self.all_countries = sorted(self.df_olympics['NOC'].dropna().unique())
            
            print("Datos cargados y procesados correctamente.")
            
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en {self.DATA_PATH}")
            messagebox.showerror("Error de Archivo", 
                                 f"No se encontró '{os.path.basename(self.DATA_PATH)}' en la carpeta 'data/'.\n"
                                 "Por favor, descarga el dataset y colócalo allí.")
            self.df_olympics = None
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            messagebox.showerror("Error", f"No se pudieron cargar los datos: {e}")
            self.df_olympics = None

    def show_error_loading(self):
        """Muestra un mensaje si los datos no se cargaron."""
        ttk.Label(self, text="Error al cargar los datos.", style="Title.TLabel").pack(pady=50)

    def create_widgets(self):
        """Crea la interfaz de usuario (filtros y áreas de resultado)."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Análisis del Medallero Olímpico", style="Title.TLabel").pack(pady=(0, 20))
        
        # --- Panel de Filtros ---
        filter_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        filter_card.pack(fill=tk.X, pady=10)
        
        # Centrar filtros
        filter_center = ttk.Frame(filter_card, style="Card.TFrame")
        filter_center.pack()

        # Filtro de Año
        ttk.Label(filter_center, text="Año:", style="Card.TLabel").grid(row=0, column=0, padx=5, sticky="w")
        self.year_combo = ttk.Combobox(filter_center, values=['Todos'] + self.all_years, state="readonly")
        self.year_combo.set('Todos')
        self.year_combo.grid(row=0, column=1, padx=5)
        
        # Filtro de País
        ttk.Label(filter_center, text="País (NOC):", style="Card.TLabel").grid(row=0, column=2, padx=5, sticky="w")
        self.country_combo = ttk.Combobox(filter_center, values=['Todos'] + self.all_countries, state="readonly", width=30)
        self.country_combo.set('Todos')
        self.country_combo.grid(row=0, column=3, padx=5)

        # Filtro de Temporada
        ttk.Label(filter_center, text="Temporada:", style="Card.TLabel").grid(row=0, column=4, padx=5, sticky="w")
        self.season_combo = ttk.Combobox(filter_center, values=['Todos', 'Summer', 'Winter'], state="readonly")
        self.season_combo.set('Todos')
        self.season_combo.grid(row=0, column=5, padx=5)
        
        # Botón de Análisis
        self.analyze_button = ttk.Button(filter_center, text="Analizar", command=self.run_analysis)
        self.analyze_button.grid(row=0, column=6, padx=20)
        
        # --- Panel de Resultados (con Pestañas) ---
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Pestaña 1: Medallero (Tabla)
        self.tab_table = ttk.Frame(self.notebook) # No necesita estilo de tarjeta, ya lo tiene el notebook
        self.notebook.add(self.tab_table, text='Medallero (Tabla)')
        self.create_table_tab()
        
        # Pestaña 2: Evolución (Gráfico)
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text='Evolución (Gráfico)')
        
        # Canvas para el gráfico de Matplotlib
        self.plot_canvas_frame = ttk.Frame(self.tab_plot, style="Card.TFrame")
        self.plot_canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.plot_label = ttk.Label(self.plot_canvas_frame,
                                    text="Selecciona un país para ver su evolución histórica.",
                                    style="Card.TLabel", font=FONT_BODY_BOLD, anchor="center")
        self.plot_label.pack(expand=True)
        self.plot_canvas = None # Para guardar la referencia del canvas de Matplotlib


        # Ejecutar análisis inicial
        self.run_analysis()

    def create_table_tab(self):
        """Crea el widget Treeview (tabla) en la primera pestaña."""
        
        # Frame para contener el Treeview y la Scrollbar
        table_frame = ttk.Frame(self.tab_table)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cols = ('País (NOC)', 'Oro', 'Plata', 'Bronce', 'Total')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        for col in cols:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_treeview(_col, False))
            self.tree.column(col, width=150, anchor=tk.CENTER)
            
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def sort_treeview(self, col, reverse):
        """Ordena la tabla al hacer clic en la cabecera."""
        try:
            # Intenta convertir a número para ordenar
            data = [(float(self.tree.set(child, col)), child) for child in self.tree.get_children('')]
            data.sort(reverse=reverse, key=lambda x: x[0])
        except ValueError:
            # Si falla (ej. nombres de países), ordena como texto
            data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
            data.sort(reverse=reverse)

        for index, (val, child) in enumerate(data):
            self.tree.move(child, '', index)

        # Cambia el comando para ordenar en la dirección opuesta la próxima vez
        self.tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))


    def run_analysis(self):
        """Función principal que filtra datos y actualiza los widgets."""
        
        # 1. Obtener filtros
        year = self.year_combo.get()
        country = self.country_combo.get()
        season = self.season_combo.get()
        
        # 2. Filtrar el DataFrame
        df_filtered = self.df_olympics.copy()
        
        if year != 'Todos':
            df_filtered = df_filtered[df_filtered['Year'] == int(year)]
        
        # === ¡ARREGLO 2! ===
        if country != 'Todos':
            df_filtered = df_filtered[df_filtered['NOC'] == country]
            
        if season != 'Todos':
            df_filtered = df_filtered[df_filtered['Season'] == season]
        
        # 3. Actualizar la tabla
        self.update_table(df_filtered)
        
        # 4. Actualizar el gráfico
        self.update_plot(df_filtered, country)

    def update_table(self, df):
        """Calcula el medallero y lo muestra en el Treeview."""
        
        # Limpiar tabla anterior
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Calcular medallero usando pandas
        # Usamos 'Event' para evitar contar medallas de equipo varias veces
        df_medals = df.drop_duplicates(subset=['Year', 'Event', 'Medal'])
        
        # === ¡ARREGLO 3! ===
        # Agrupar por 'NOC' y tipo de medalla
        medallero = df_medals.groupby(['NOC', 'Medal']).size().unstack(fill_value=0)
        
        # Asegurar que existan las 3 columnas
        for medal in ['Gold', 'Silver', 'Bronze']:
            if medal not in medallero.columns:
                medallero[medal] = 0
        
        medallero['Total'] = medallero['Gold'] + medallero['Silver'] + medallero['Bronze']
        medallero = medallero.sort_values(by=['Total', 'Gold', 'Silver'], ascending=False)
        
        # Insertar en la tabla (Treeview)
        for index, row in medallero.iterrows():
            self.tree.insert("", tk.END, values=(index, row['Gold'], row['Silver'], row['Bronze'], row['Total']))

    def update_plot(self, df, selected_country):
        """Crea un gráfico de evolución y lo muestra en la Pestaña 2."""
        
        # Limpiar el canvas anterior (si existe)
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        
        # Mostrar el label de instrucción si no hay país seleccionado
        if selected_country == 'Todos':
            self.plot_label.pack(expand=True) # Muestra el label de instrucción
            return
        else:
            self.plot_label.pack_forget() # Oculta el label de instrucción

        # === ¡ARREGLO 4! ===
        # 1. Preparar datos para el gráfico
        df_country = df[df['NOC'] == selected_country]
        
        # Agrupar por año y medalla
        evolution = df_country.groupby(['Year', 'Medal']).size().unstack(fill_value=0)
        
        if evolution.empty:
            self.plot_label.config(text=f"No hay datos para {selected_country} con estos filtros.")
            self.plot_label.pack(expand=True)
            return

        # 2. Crear el gráfico con Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Colores
        colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
        
        # Asegurar que el gráfico tenga las columnas en el orden correcto si existen
        cols_to_plot = [col for col in ['Gold', 'Silver', 'Bronze'] if col in evolution.columns]
        
        # Graficar (gráfico de barras apiladas)
        evolution[cols_to_plot].plot(kind='bar', stacked=True, ax=ax, 
                                     color=[colors.get(c, '#888888') for c in cols_to_plot])
        
        ax.set_title(f"Evolución de Medallas para {selected_country}", color=COLOR_FG)
        ax.set_xlabel("Año", color=COLOR_FG)
        ax.set_ylabel("Número de Medallas", color=COLOR_FG)
        
        # Estilo del gráfico (para que combine con la UI oscura)
        fig.patch.set_facecolor(COLOR_CARD)
        ax.set_facecolor(COLOR_CARD)
        
        ax.tick_params(colors=COLOR_FG, axis='x')
        ax.tick_params(colors=COLOR_FG, axis='y')
        
        ax.xaxis.label.set_color(COLOR_FG)
        ax.yaxis.label.set_color(COLOR_FG)
        ax.title.set_color(COLOR_FG)
        
        ax.spines['top'].set_color(COLOR_CARD)
        ax.spines['right'].set_color(COLOR_CARD)
        ax.spines['bottom'].set_color(COLOR_FG)
        ax.spines['left'].set_color(COLOR_FG)
        
        ax.legend(facecolor=COLOR_CARD, labelcolor=COLOR_FG)
        
        # 3. Incrustar el gráfico en Tkinter
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
# --- 3. EJECUTAR LA APLICACIÓN ---

if __name__ == "__main__":
    app = OlimpiadasApp()
    app.mainloop()