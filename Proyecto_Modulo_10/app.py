# Proyecto_Modulo_10/app.py

import tkinter as tk
from tkinter import ttk, messagebox
from juego_snake import SnakeGame
from agente_rl_snake import AgenteSnakeQL
import os
import math

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
CELL_SIZE = 28
GAME_SPEED = 100

class AppSnakeIA(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Snake Q-Learning")
        self.configure(bg=COLOR_BG)
        
        self.grid_size = 20
        self.grid_height = 15
        self.juego = SnakeGame(grid_width=self.grid_size, grid_height=self.grid_height, vidas_iniciales=3)
        self.agente = AgenteSnakeQL()

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

        ttl_frame = ttk.Frame(main_frame)
        ttl_frame.pack(fill="x", pady=(0, 10))
        
        ttl_label = ttk.Label(ttl_frame, text="Snake Q-Learning", style="Title.TLabel")
        ttl_label.pack()

        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill="x", pady=10)

        self.label_puntos = ttk.Label(stats_frame, text="Puntos: 0", 
                                     style="Stats.TLabel", foreground=COLOR_ACCENT)
        self.label_puntos.pack(side="left", padx=15)

        self.label_vidas = ttk.Label(stats_frame, text="Vidas: 3", 
                                    style="Stats.TLabel", foreground="#ff3366")
        self.label_vidas.pack(side="left", padx=15)

        self.label_record = ttk.Label(stats_frame, text="Record: 0", 
                                     style="Stats.TLabel", foreground="#ffd700")
        self.label_record.pack(side="right", padx=15)

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

        self.dibujar_grid()

        self.btn_iniciar = tk.Button(
            main_frame, text="Empezar Juego", 
            font=("SF Pro Display", 14, "bold"), 
            bg=COLOR_ACCENT, fg="#000000",
            activebackground="#00a3cc", relief="flat", 
            padx=30, pady=12,
            command=self.toggle_juego,
            cursor="hand2"
        )
        self.btn_iniciar.pack(pady=15, fill="x")
        
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
        """Dibuja la serpiente como un cuerpo continuo tipo Slither.io"""
        self.canvas.delete("snake", "food", "eyes")
        
        if len(self.juego.snake) < 2:
            return
        
        # Radio base de la serpiente
        radio_base = CELL_SIZE // 2 - 2

        for idx in range(len(self.juego.snake) - 1):
            fila1, col1 = self.juego.snake[idx]
            fila2, col2 = self.juego.snake[idx + 1]
            
            x1 = col1 * CELL_SIZE + CELL_SIZE // 2
            y1 = fila1 * CELL_SIZE + CELL_SIZE // 2
            x2 = col2 * CELL_SIZE + CELL_SIZE // 2
            y2 = fila2 * CELL_SIZE + CELL_SIZE // 2
            
            # Calcular el radio para cada segmento 
            if idx == 0:
                radio = radio_base
            else:
                factor = 1 - (idx / len(self.juego.snake)) * 0.4
                radio = int(radio_base * factor)

            if idx == 0:
                color = "#00ffcc"
            else:
                color = COLOR_SNAKE

            self.canvas.create_oval(
                x1 - radio, y1 - radio,
                x1 + radio, y1 + radio,
                fill=color, outline="", tags="snake"
            )

            if idx < len(self.juego.snake) - 1:
                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    px = -dy / dist * radio
                    py = dx / dist * radio

                    self.canvas.create_polygon(
                        x1 + px, y1 + py,
                        x1 - px, y1 - py,
                        x2 - px, y2 - py,
                        x2 + px, y2 + py,
                        fill=color, outline="", tags="snake"
                    )

        ultima_fila, ultima_col = self.juego.snake[-1]
        x_ultima = ultima_col * CELL_SIZE + CELL_SIZE // 2
        y_ultima = ultima_fila * CELL_SIZE + CELL_SIZE // 2
        radio_cola = int(radio_base * 0.6)
        
        self.canvas.create_oval(
            x_ultima - radio_cola, y_ultima - radio_cola,
            x_ultima + radio_cola, y_ultima + radio_cola,
            fill=COLOR_SNAKE, outline="", tags="snake"
        )

        cabeza_fila, cabeza_col = self.juego.snake[0]
        x_cabeza = cabeza_col * CELL_SIZE + CELL_SIZE // 2
        y_cabeza = cabeza_fila * CELL_SIZE + CELL_SIZE // 2

        direccion = self.juego.direccion

        tamano_ojo = radio_base // 3
        separacion = radio_base // 2

        if direccion == 'RIGHT':
            ojo1_x, ojo1_y = x_cabeza + separacion//2, y_cabeza - separacion
            ojo2_x, ojo2_y = x_cabeza + separacion//2, y_cabeza + separacion
        elif direccion == 'LEFT':
            ojo1_x, ojo1_y = x_cabeza - separacion//2, y_cabeza - separacion
            ojo2_x, ojo2_y = x_cabeza - separacion//2, y_cabeza + separacion
        elif direccion == 'UP':
            ojo1_x, ojo1_y = x_cabeza - separacion, y_cabeza - separacion//2
            ojo2_x, ojo2_y = x_cabeza + separacion, y_cabeza - separacion//2
        elif direccion == 'DOWN':
            ojo1_x, ojo1_y = x_cabeza - separacion, y_cabeza + separacion//2
            ojo2_x, ojo2_y = x_cabeza + separacion, y_cabeza + separacion//2

        for ojo_x, ojo_y in [(ojo1_x, ojo1_y), (ojo2_x, ojo2_y)]:
            self.canvas.create_oval(
                ojo_x - tamano_ojo, ojo_y - tamano_ojo,
                ojo_x + tamano_ojo, ojo_y + tamano_ojo,
                fill="white", outline="", tags="eyes"
            )
            pupila = tamano_ojo // 2
            self.canvas.create_oval(
                ojo_x - pupila, ojo_y - pupila,
                ojo_x + pupila, ojo_y + pupila,
                fill="black", outline="", tags="eyes"
            )
        
        # Dibujar comida
        fila_food, col_food = self.juego.food
        x_center = col_food * CELL_SIZE + CELL_SIZE // 2
        y_center = fila_food * CELL_SIZE + CELL_SIZE // 2
        radio = CELL_SIZE // 3
        
        self.canvas.create_oval(
            x_center - radio, y_center - radio,
            x_center + radio, y_center + radio,
            fill=COLOR_FOOD, outline="", tags="food"
        )
    
    def actualizar_stats(self):
        """Actualiza las estadísticas en pantalla"""
        info = self.juego.get_info()
        self.label_puntos.config(text=f"Puntos: {info['puntos']}")
        self.label_vidas.config(text=f"Vidas: {info['vidas']}")
        self.label_record.config(text=f"Record: {info['record']}")
    
    def toggle_juego(self):
        """Inicia o pausa el juego"""
        if not self.game_running:
            self.game_running = True
            self.btn_iniciar.config(text="Pausar Juego", bg="#ff9500")
            self.loop_juego()
        else:
            self.game_running = False
            self.btn_iniciar.config(text="Continuar", bg=COLOR_ACCENT)
    
    def loop_juego(self):
        """Loop principal del juego"""
        if not self.game_running:
            return
        
        if self.juego.game_over:
            self.manejar_game_over()
            return

        estado = self.juego.get_estado()
        acciones = self.juego.get_acciones_validas()
        accion = self.agente.elegir_accion(estado, acciones)
        
        _, _, perdio_vida, game_over = self.juego.step(accion)
        
        # Dibujar y actualizar
        self.dibujar_juego()
        self.actualizar_stats()
        
        self.after(GAME_SPEED, self.loop_juego)
    
    def manejar_game_over(self):
        """Maneja el fin del juego"""
        info = self.juego.get_info()
        
        self.game_running = False
        self.btn_iniciar.config(text="Jugar de Nuevo", bg=COLOR_ACCENT)
        
        mensaje = (
            f"Game Over\n\n"
            f"Puntos: {info['puntos']}\n"
            f"Longitud: {info['longitud']}\n"
            f"Record: {info['record']}"
        )
        messagebox.showinfo("Game Over", mensaje)
        
        # Reiniciar juego completo y limpiar tablero
        self.juego.reset_completo()
        self.canvas.delete("snake", "food", "eyes")
        self.dibujar_grid()
        self.actualizar_stats()
    
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