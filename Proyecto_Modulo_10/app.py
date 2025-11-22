# Proyecto_Modulo_10/app.py

import tkinter as tk
from tkinter import ttk, messagebox
from juego import TresEnRaya
from agente_rl import AgenteQLearning
import os
import time

# --- 1. CONFIGURACIÓN DE ESTILO (Reutiliza tu estilo) ---
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#2c2c2c"
COLOR_FG = "#ffffff"
COLOR_ACCENT = "#007aff"
COLOR_X = "#ff3b30" # Color para X (Humano)
COLOR_O = "#34c759" # Color para O (IA)

FONT_TITLE = ("SF Pro Display", 20, "bold")
FONT_BODY = ("SF Pro Text", 12)
FONT_STATUS = ("SF Pro Text", 14, "bold")
FONT_BOTON_JUEGO = ("SF Pro Display", 48, "bold")

MODEL_PATH = "models/agente_tres_en_raya.joblib"

class AppTresEnRaya(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Módulo 10: IA de Tres en Raya (Q-Learning)")
        self.geometry("450x600")
        self.configure(bg=COLOR_BG)
        
        self.juego = TresEnRaya()
        self.agente = AgenteQLearning()
        self.humano_puede_jugar = True

        # Cargar modelo entrenado
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error de Modelo", 
                                 f"No se encontró '{MODEL_PATH}'.\n"
                                 "Por favor, ejecuta 'entrenar.py' primero.")
            self.after(100, self.destroy)
            return
        
        self.agente.cargar_modelo(MODEL_PATH)
        
        self.setup_styles()
        self.crear_widgets()

    def setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(".", background=COLOR_BG, foreground=COLOR_FG, fieldbackground=COLOR_CARD)
        self.style.configure("TFrame", background=COLOR_BG)
        self.style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG, font=FONT_BODY)
        self.style.configure("Title.TLabel", font=FONT_TITLE, background=COLOR_BG)
        self.style.configure("Status.TLabel", font=FONT_STATUS, background=COLOR_BG)
        self.style.configure("TButton", font=FONT_BODY, padding=10, background=COLOR_ACCENT)

    def crear_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(main_frame, text="Jugar contra la IA", style="Title.TLabel").pack(pady=(0, 20))
        
        # Tablero 3x3
        self.frame_tablero = ttk.Frame(main_frame)
        self.frame_tablero.pack(pady=10)
        
        self.botones_tablero = []
        for i in range(3):
            fila_botones = []
            for j in range(3):
                boton = tk.Button(self.frame_tablero, 
                                  text="", 
                                  font=FONT_BOTON_JUEGO, 
                                  width=4, height=2, 
                                  bg=COLOR_CARD, 
                                  fg=COLOR_ACCENT,
                                  activebackground="#3e3e3e",
                                  relief="flat",
                                  command=lambda f=i, c=j: self.click_humano(f, c))
                boton.grid(row=i, column=j, padx=5, pady=5)
                fila_botones.append(boton)
            self.botones_tablero.append(fila_botones)
            
        self.status_label = ttk.Label(main_frame, text="Tu turno (X)", 
                                     style="Status.TLabel", anchor="center")
        self.status_label.pack(fill="x", pady=20)
        
        self.reiniciar_button = ttk.Button(main_frame, text="Reiniciar Partida", 
                                          command=self.reiniciar_partida)
        self.reiniciar_button.pack(fill="x", ipady=10)

    def click_humano(self, fila, col):
        if not self.humano_puede_jugar or self.juego.juego_terminado:
            return

        if self.juego.realizar_movimiento(fila, col, 1): # Jugador 1 = Humano
            self.actualizar_tablero()
            
            if self.juego.juego_terminado:
                self.manejar_fin_juego()
                return
                
            self.humano_puede_jugar = False
            self.status_label.config(text="Turno de la IA (O)...")
            self.after(500, self.turno_ia) # Pequeña pausa para realismo
        else:
            self.status_label.config(text="Movimiento inválido. Intenta de nuevo.")

    def turno_ia(self):
        if self.juego.juego_terminado:
            return
            
        estado_actual = self.juego.get_estado()
        mov_validos = self.juego.get_mov_validos()
        
        # La IA (cargada) elige la mejor acción (explotación)
        accion_ia = self.agente.elegir_accion(estado_actual, mov_validos)
        
        self.juego.realizar_movimiento(accion_ia[0], accion_ia[1], 2) # Jugador 2 = IA
        
        self.actualizar_tablero()
        
        if self.juego.juego_terminado:
            self.manejar_fin_juego()
        else:
            self.status_label.config(text="Tu turno (X)")
            self.humano_puede_jugar = True

    def actualizar_tablero(self):
        for i in range(3):
            for j in range(3):
                if self.juego.tablero[i, j] == 1:
                    self.botones_tablero[i][j].config(text="X", fg=COLOR_X, bg=COLOR_BG)
                elif self.juego.tablero[i, j] == 2:
                    self.botones_tablero[i][j].config(text="O", fg=COLOR_O, bg=COLOR_BG)
                else:
                    self.botones_tablero[i][j].config(text="", bg=COLOR_CARD)

    def manejar_fin_juego(self):
        self.humano_puede_jugar = False
        if self.juego.ganador == 1:
            self.status_label.config(text="¡Ganaste! (Esto es... improbable)")
        elif self.juego.ganador == 2:
            self.status_label.config(text="¡La IA Gana!")
        elif self.juego.ganador == 0:
            self.status_label.config(text="Es un Empate")
            
        # Deshabilitar botones
        for i in range(3):
            for j in range(3):
                self.botones_tablero[i][j].config(state="disabled")

    def reiniciar_partida(self):
        self.juego.reiniciar()
        self.humano_puede_jugar = True
        self.status_label.config(text="Tu turno (X)")
        
        # Limpiar y reactivar botones
        for i in range(3):
            for j in range(3):
                self.botones_tablero[i][j].config(text="", bg=COLOR_CARD, state="normal")

if __name__ == "__main__":
    app = AppTresEnRaya()
    app.mainloop()