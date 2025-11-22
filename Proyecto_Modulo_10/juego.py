# Proyecto_Modulo_10/juego.py

import numpy as np

class TresEnRaya:
    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        # 0 = vacío, 1 = Jugador (Humano/X), 2 = Agente (IA/O)
        self.tablero = np.zeros((3, 3), dtype=int)
        self.juego_terminado = False
        self.ganador = None # None, 1, 2, o 0 (Empate)
        return self.get_estado()

    def get_estado(self):
        # Convertimos el tablero (array) en un string o tupla
        # para usarlo como clave en nuestra Tabla-Q.
        return tuple(self.tablero.flatten())

    def get_mov_validos(self):
        return [(i, j) for i in range(3) for j in range(3) if self.tablero[i, j] == 0]

    def realizar_movimiento(self, fila, col, jugador):
        if self.tablero[fila, col] == 0:
            self.tablero[fila, col] = jugador
            self.verificar_ganador()
            return True
        return False # Movimiento inválido

    def verificar_ganador(self):
        # Comprobar filas, columnas y diagonales
        for i in range(3):
            if self.tablero[i, 0] == self.tablero[i, 1] == self.tablero[i, 2] != 0:
                self.ganador = self.tablero[i, 0]
            if self.tablero[0, i] == self.tablero[1, i] == self.tablero[2, i] != 0:
                self.ganador = self.tablero[0, i]
        
        if self.tablero[0, 0] == self.tablero[1, 1] == self.tablero[2, 2] != 0:
            self.ganador = self.tablero[0, 0]
        if self.tablero[0, 2] == self.tablero[1, 1] == self.tablero[2, 0] != 0:
            self.ganador = self.tablero[0, 2]

        if self.ganador:
            self.juego_terminado = True
        elif not self.get_mov_validos(): # No hay más movimientos
            self.juego_terminado = True
            self.ganador = 0 # Empate