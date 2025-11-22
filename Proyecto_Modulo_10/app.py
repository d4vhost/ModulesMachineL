# Proyecto_Modulo_10/app.py

import tkinter as tk
from tkinter import ttk, messagebox
from juego_snake import SnakeGame
from agente_rl_snake import AgenteSnakeQL
import os

# --- CONFIGURACIÓN DE ESTILO ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#00d4ff"
COLOR_SNAKE = "#00ff88"
COLOR_FOOD = "#ff3366"
COLOR_GRID = "#2a2a2a"

FONT_TITLE = ("SF Pro Display", 22, "bold")
FONT_STATS = ("SF Pro Text", 14, "bold")
FONT_BODY = ("SF Pro Text", 12)

MODEL_PATH = "models/agente_snake_ql.joblib"
CELL_SIZE = 28  # Celdas más pequeñas
GAME_SPEED = 100

class AppSnakeIA(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("🐍 Snake IA - Q-Learning (Módulo 10)")
        self.configure(bg=COLOR_BG)
        
        self.grid_size = 20  # Más ancho: 20x15 en lugar de 15x15
        self.grid_height = 15  # Alto reducido
        self.juego = SnakeGame(grid_width=self.grid_size, grid_height=self.grid_height, vidas_iniciales=3)
        self.agente = AgenteSnakeQL()
        
        # Verificar modelo
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Modelo no encontrado",
                f"No se encontró '{MODEL_PATH}'.\n\n"
                "Por favor, ejecuta 'entrenar.py' primero."
            )
            self.after(100, self.destroy)
            return
        
        self.agente.cargar_modelo(MODEL_PATH)
        self.agente.modo_juego()
        
        self.game_running = False
        
        self.setup_styles()
        self.crear_widgets()
        self.centrar_ventana()
        
    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE)
        self.style.configure("Stats.TLabel", font=FONT_STATS)
        
    def crear_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Título
        ttl_frame = ttk.Frame(main_frame)
        ttl_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(ttl_frame, text="🐍 Snake IA - Q-Learning", 
                 style="Title.TLabel").pack()
        
        # Panel de estadísticas
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill="x", pady=10)
        
        # Puntos
        self.label_puntos = ttk.Label(stats_frame, text="Puntos: 0", 
                                     style="Stats.TLabel", foreground=COLOR_ACCENT)
        self.label_puntos.pack(side="left", padx=15)
        
        # Vidas
        self.label_vidas = ttk.Label(stats_frame, text="❤️ Vidas: 3", 
                                    style="Stats.TLabel", foreground="#ff3366")
        self.label_vidas.pack(side="left", padx=15)
        
        # Record
        self.label_record = ttk.Label(stats_frame, text="🏆 Record: 0", 
                                     style="Stats.TLabel", foreground="#ffd700")
        self.label_record.pack(side="right", padx=15)
        
        # Canvas del juego - MÁS ANCHO, MENOS ALTO
        canvas_width = self.grid_size * CELL_SIZE
        canvas_height = self.grid_height * CELL_SIZE
        
        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_width,
            height=canvas_height,
            bg=COLOR_CARD,
            highlightthickness=2,
            highlightbackground=COLOR_ACCENT
        )
        self.canvas.pack(pady=15)
        
        # Dibujar grid
        self.dibujar_grid()
        
        # Botón de control
        self.btn_iniciar = tk.Button(
            main_frame, text="▶️ Empezar Juego", 
            font=("SF Pro Display", 14, "bold"), 
            bg=COLOR_ACCENT, fg="#000000",
            activebackground="#00a3cc", relief="flat", 
            padx=30, pady=12,
            command=self.toggle_juego,
            cursor="hand2"
        )
        self.btn_iniciar.pack(pady=15, fill="x")
        
        # Etiqueta de estado
        self.label_estado = ttk.Label(
            main_frame, 
            text="La IA está lista para jugar", 
            font=FONT_BODY,
            foreground="#888888"
        )
        self.label_estado.pack(pady=5)
        
    def dibujar_grid(self):
        """Dibuja la cuadrícula de fondo"""
        for i in range(self.grid_height):
            for j in range(self.grid_size):
                x1 = j * CELL_SIZE
                y1 = i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=COLOR_GRID, fill=COLOR_CARD, width=1
                )
    
    def dibujar_juego(self):
        """Dibuja la serpiente y la comida"""
        self.canvas.delete("snake", "food")
        
        # Dibujar serpiente
        for idx, (fila, col) in enumerate(self.juego.snake):
            x1 = col * CELL_SIZE + 2
            y1 = fila * CELL_SIZE + 2
            x2 = x1 + CELL_SIZE - 4
            y2 = y1 + CELL_SIZE - 4
            
            # Cabeza más brillante
            color = COLOR_SNAKE if idx > 0 else "#00ffcc"
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=color, outline="", tags="snake"
            )
        
        # Dibujar comida
        fila_food, col_food = self.juego.food
        x1 = col_food * CELL_SIZE + 5
        y1 = fila_food * CELL_SIZE + 5
        x2 = x1 + CELL_SIZE - 10
        y2 = y1 + CELL_SIZE - 10
        self.canvas.create_oval(
            x1, y1, x2, y2,
            fill=COLOR_FOOD, outline="", tags="food"
        )
    
    def actualizar_stats(self):
        """Actualiza las estadísticas en pantalla"""
        info = self.juego.get_info()
        self.label_puntos.config(text=f"Puntos: {info['puntos']}")
        self.label_vidas.config(text=f"❤️ Vidas: {info['vidas']}")
        self.label_record.config(text=f"🏆 Record: {info['record']}")
    
    def toggle_juego(self):
        """Inicia o pausa el juego"""
        if not self.game_running:
            self.game_running = True
            self.btn_iniciar.config(text="⏸️ Pausar Juego", bg="#ff9500")
            self.label_estado.config(text="🤖 La IA está jugando...")
            self.loop_juego()
        else:
            self.game_running = False
            self.btn_iniciar.config(text="▶️ Continuar", bg=COLOR_ACCENT)
            self.label_estado.config(text="⏸️ Juego pausado")
    
    def loop_juego(self):
        """Loop principal del juego"""
        if not self.game_running:
            return
        
        if self.juego.game_over:
            self.manejar_game_over()
            return
        
        # La IA elige la acción
        estado = self.juego.get_estado()
        acciones = self.juego.get_acciones_validas()
        accion = self.agente.elegir_accion(estado, acciones)
        
        # Ejecutar acción
        _, _, perdio_vida, game_over = self.juego.step(accion)
        
        # Si perdió una vida, mostrar notificación temporal
        if perdio_vida and not game_over:
            self.label_estado.config(
                text=f"💥 ¡Choque! Vidas restantes: {self.juego.vidas}",
                foreground="#ff3366"
            )
            self.after(1500, lambda: self.label_estado.config(
                text="🤖 La IA está jugando...",
                foreground="#888888"
            ))
        
        # Dibujar y actualizar
        self.dibujar_juego()
        self.actualizar_stats()
        
        # Próximo frame
        self.after(GAME_SPEED, self.loop_juego)
    
    def manejar_game_over(self):
        """Maneja el fin del juego"""
        info = self.juego.get_info()
        
        self.game_running = False
        self.btn_iniciar.config(text="▶️ Jugar de Nuevo", bg=COLOR_ACCENT)
        self.label_estado.config(
            text="💀 Game Over - Sin vidas restantes",
            foreground="#ff3366"
        )
        
        mensaje = (
            f"🐍 ¡Game Over!\n\n"
            f"Puntos: {info['puntos']}\n"
            f"Longitud: {info['longitud']}\n"
            f"Record: {info['record']}"
        )
        messagebox.showinfo("Game Over", mensaje)
        
        # Reiniciar juego completo
        self.juego.reset_completo()
        self.dibujar_juego()
        self.actualizar_stats()
        self.label_estado.config(
            text="La IA está lista para jugar",
            foreground="#888888"
        )
    
    def centrar_ventana(self):
        """Centra la ventana en la pantalla"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

if __name__ == "__main__":
    app = AppSnakeIA()
    app.mainloop()