import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import BytesIO
import os

# 1. CONFIGURACIÓN DE ESTILO 
COLOR_BG = "#F4F1EA"      
COLOR_CARD = "#E0D8C3"     
COLOR_FG = "#4A4036"      
COLOR_ACCENT = "#A65E44"   
COLOR_ACCENT_HOVER = "#8C4B34" 

FONT_TITLE = ("Courier New", 22, "bold")
FONT_BODY = ("Helvetica", 11)
FONT_BODY_BOLD = ("Helvetica", 11, "bold")

# 2. CLASE PRINCIPAL DE LA APLICACIÓN 

class OlimpiadasApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Módulo 1: Historia Olímpica")
        window_width = 1000
        window_height = 650  
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x_c = int((screen_width / 2) - (window_width / 2))
        y_c = int((screen_height / 2) - (window_height / 2))
        
        self.geometry(f"{window_width}x{window_height}+{x_c}+{y_c}")
        self.configure(bg=COLOR_BG)
        
        self.df_olympics = None 
        self.all_years = []
        self.all_countries = []

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_PATH = os.path.join(BASE_DIR, "data", "athlete_events.csv")
        
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.style.configure(".", 
                             background=COLOR_BG, 
                             foreground=COLOR_FG, 
                             fieldbackground=COLOR_BG)
        
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("Card.TFrame", background=COLOR_CARD, relief="ridge", borderwidth=2)
        
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG, foreground=COLOR_ACCENT)
        
        self.style.configure("TCombobox",
                             fieldbackground="#FFFFFF",
                             background=COLOR_CARD,
                             foreground=COLOR_FG,
                             arrowcolor=COLOR_ACCENT)
        
        self.style.configure("TButton",
                             font=FONT_BODY_BOLD,
                             background=COLOR_ACCENT,
                             foreground="#FFFFFF",
                             borderwidth=1,
                             focusthickness=3,
                             focuscolor="none")
        self.style.map("TButton",
                       background=[("active", COLOR_ACCENT_HOVER), ("pressed", COLOR_ACCENT_HOVER)])
        
        self.style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        self.style.configure("TNotebook.Tab",
                             background=COLOR_CARD,
                             foreground=COLOR_FG,
                             padding=[15, 5],
                             font=FONT_BODY_BOLD)
        self.style.map("TNotebook.Tab",
                       background=[("selected", COLOR_BG)],
                       foreground=[("selected", COLOR_ACCENT)])

        self.style.configure("Treeview",
                             background="#FFFFFF",
                             foreground=COLOR_FG,
                             fieldbackground="#FFFFFF",
                             rowheight=25,
                             font=FONT_BODY)
        self.style.configure("Treeview.Heading",
                             background=COLOR_CARD,
                             foreground=COLOR_FG,
                             font=FONT_BODY_BOLD,
                             relief="raised")
        self.style.map("Treeview", 
                       background=[('selected', COLOR_ACCENT)], 
                       foreground=[('selected', '#FFFFFF')])

        # Cargar datos 
        self.load_data()
        
        # Crear Widgets 
        if self.df_olympics is not None:
            self.create_widgets()
        else:
            self.show_error_loading()

    def load_data(self):
        try:
            self.df_olympics = pd.read_csv(self.DATA_PATH)
            self.df_olympics = self.df_olympics.dropna(subset=['Medal'])
            self.all_years = sorted(self.df_olympics['Year'].unique())
            self.all_countries = sorted(self.df_olympics['NOC'].dropna().unique())
            print("Datos cargados correctamente.")
        except Exception as e:
            print(f"Error: {e}")
            self.df_olympics = None

    def show_error_loading(self):
        ttk.Label(self, text="Error al cargar data/athlete_events.csv", style="Title.TLabel").pack(pady=50)

    def create_widgets(self):
        main_wrapper = ttk.Frame(self)
        main_wrapper.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)
        
        # Título
        ttk.Label(main_wrapper, text="HISTORIAL OLÍMPICO", style="Title.TLabel", anchor="center").pack(pady=(0, 10))
        
        # Panel de Filtros
        filter_card = ttk.Frame(main_wrapper, style="Card.TFrame", padding=10)
        filter_card.pack(fill=tk.X, pady=5)
        
        filter_inner = ttk.Frame(filter_card, style="Card.TFrame")
        filter_inner.pack(anchor="center")

        ttk.Label(filter_inner, text="Año:", style="Card.TLabel").grid(row=0, column=0, padx=5)
        self.year_combo = ttk.Combobox(filter_inner, values=['Todos'] + self.all_years, state="readonly", width=10)
        self.year_combo.set('Todos')
        self.year_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(filter_inner, text="País (NOC):", style="Card.TLabel").grid(row=0, column=2, padx=5)
        self.country_combo = ttk.Combobox(filter_inner, values=['Todos'] + self.all_countries, state="readonly", width=25)
        self.country_combo.set('Todos')
        self.country_combo.grid(row=0, column=3, padx=5)

        ttk.Label(filter_inner, text="Temporada:", style="Card.TLabel").grid(row=0, column=4, padx=5)
        self.season_combo = ttk.Combobox(filter_inner, values=['Todos', 'Summer', 'Winter'], state="readonly", width=10)
        self.season_combo.set('Todos')
        self.season_combo.grid(row=0, column=5, padx=5)
        
        self.analyze_button = ttk.Button(filter_inner, text="GENERAR REPORTE", command=self.run_analysis)
        self.analyze_button.grid(row=0, column=6, padx=20)
        
        # Panel de Resultados
        self.notebook = ttk.Notebook(main_wrapper)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text='TABLA DE DATOS')
        self.create_table_tab()
        
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text='GRÁFICA EVOLUTIVA')
        
        self.plot_canvas_frame = ttk.Frame(self.tab_plot, style="Card.TFrame", padding=5)
        self.plot_canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.plot_label = ttk.Label(self.plot_canvas_frame,
                                    text="Selecciona un país para visualizar su historia.",
                                    style="Card.TLabel", font=FONT_BODY_BOLD)
        self.plot_label.pack(expand=True)
        self.plot_canvas = None

        self.run_analysis()

    def create_table_tab(self):
        table_frame = ttk.Frame(self.tab_table)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        cols = ('País (NOC)', 'Oro', 'Plata', 'Bronce', 'Total')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        for col in cols:
            self.tree.heading(col, text=col, command=lambda _col=col: self.sort_treeview(_col, False))
            self.tree.column(col, width=120, anchor=tk.CENTER)
            
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def sort_treeview(self, col, reverse):
        try:
            data = [(float(self.tree.set(child, col)), child) for child in self.tree.get_children('')]
            data.sort(reverse=reverse, key=lambda x: x[0])
        except ValueError:
            data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
            data.sort(reverse=reverse)
        for index, (val, child) in enumerate(data):
            self.tree.move(child, '', index)
        self.tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))

    def run_analysis(self):
        year = self.year_combo.get()
        country = self.country_combo.get()
        season = self.season_combo.get()
        
        df_filtered = self.df_olympics.copy()
        
        if year != 'Todos':
            df_filtered = df_filtered[df_filtered['Year'] == int(year)]
        if country != 'Todos':
            df_filtered = df_filtered[df_filtered['NOC'] == country]
        if season != 'Todos':
            df_filtered = df_filtered[df_filtered['Season'] == season]
        
        self.update_table(df_filtered)
        self.update_plot(df_filtered, country)

    def update_table(self, df):
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        df_medals = df.drop_duplicates(subset=['Year', 'Event', 'Medal'])
        medallero = df_medals.groupby(['NOC', 'Medal']).size().unstack(fill_value=0)
        
        for medal in ['Gold', 'Silver', 'Bronze']:
            if medal not in medallero.columns:
                medallero[medal] = 0
        
        medallero['Total'] = medallero['Gold'] + medallero['Silver'] + medallero['Bronze']
        medallero = medallero.sort_values(by=['Total', 'Gold', 'Silver'], ascending=False)
        
        for index, row in medallero.iterrows():
            self.tree.insert("", tk.END, values=(index, row['Gold'], row['Silver'], row['Bronze'], row['Total']))

    def update_plot(self, df, selected_country):
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None
        
        if selected_country == 'Todos':
            self.plot_label.pack(expand=True)
            return
        else:
            self.plot_label.pack_forget()

        df_country = df[df['NOC'] == selected_country]
        evolution = df_country.groupby(['Year', 'Medal']).size().unstack(fill_value=0)
        
        if evolution.empty:
            self.plot_label.config(text=f"Sin datos registrados para {selected_country}")
            self.plot_label.pack(expand=True)
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = {'Gold': '#D4AF37', 'Silver': '#A8A9AD', 'Bronze': '#CD7F32'}
        cols_to_plot = [col for col in ['Gold', 'Silver', 'Bronze'] if col in evolution.columns]
        
        evolution[cols_to_plot].plot(kind='bar', stacked=True, ax=ax, 
                                     color=[colors.get(c, '#888888') for c in cols_to_plot],
                                     edgecolor="#4A4036", linewidth=0.5)
        
        ax.set_title(f"Historia de Medallas: {selected_country}", color=COLOR_FG, fontsize=14, fontname="Courier New", weight="bold")
        ax.set_xlabel("Año Olímpico", color=COLOR_FG, fontname="Courier New")
        ax.set_ylabel("Cantidad de Medallas", color=COLOR_FG, fontname="Courier New")
        
        fig.patch.set_facecolor(COLOR_CARD)
        ax.set_facecolor(COLOR_BG)
        
        ax.tick_params(colors=COLOR_FG, axis='x', labelsize=9)
        ax.tick_params(colors=COLOR_FG, axis='y', labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color(COLOR_FG)
            spine.set_linewidth(0.8)
            
        ax.legend(facecolor=COLOR_BG, labelcolor=COLOR_FG, edgecolor=COLOR_FG)
        
        plt.subplots_adjust(bottom=0.2)

        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

if __name__ == "__main__":
    app = OlimpiadasApp()
    app.mainloop()