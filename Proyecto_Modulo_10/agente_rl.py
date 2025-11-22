# Proyecto_Modulo_10/agente_rl.py
import os
import numpy as np
import random
from collections import defaultdict
import joblib

class AgenteQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        # Q-Table: un diccionario que devuelve un diccionario que devuelve 0.0
        # Q[estado][accion] = valor
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        self.alpha = learning_rate     # Tasa de aprendizaje
        self.gamma = discount_factor   # Factor de descuento (importancia de premios futuros)
        self.epsilon = exploration_rate  # Tasa de exploración (aleatoriedad)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # Reducir la exploración con el tiempo

    def elegir_accion(self, estado, mov_validos):
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: Elige un movimiento válido al azar
            accion_idx = random.choice(range(len(mov_validos)))
            return mov_validos[accion_idx]
        else:
            # Explotación: Elige el mejor movimiento conocido
            
            q_valores = np.array([self.q_table[estado][(f,c)] for f, c in mov_validos])

            mejores_indices = np.where(q_valores == np.max(q_valores))[0]
            mejor_accion_idx = np.random.choice(mejores_indices)
            
            return mov_validos[mejor_accion_idx]

    def aprender(self, estado, accion, premio, proximo_estado, mov_validos_proximo):
        """ Esta es la fórmula mágica de Q-Learning (Bellman) """
        
        # Obtener el valor Q máximo para el próximo estado
        if not mov_validos_proximo: # Si el juego terminó
            max_q_proximo = 0.0
        else:
            q_valores_proximos = [self.q_table[proximo_estado][(f,c)] for f, c in mov_validos_proximo]
            max_q_proximo = np.max(q_valores_proximos)
            
        # El valor Q actual de la acción que tomamos
        q_actual = self.q_table[estado][accion]
        
        # Fórmula de actualización (Premio + Valor Futuro Descontado)
        nuevo_q = q_actual + self.alpha * (premio + self.gamma * max_q_proximo - q_actual)
        
        self.q_table[estado][accion] = nuevo_q

    def reducir_exploracion(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def guardar_modelo(self, ruta="models/q_table.joblib"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump(dict(self.q_table), ruta)
        print(f"Modelo (Tabla-Q) guardado en {ruta}")

    def cargar_modelo(self, ruta="models/q_table.joblib"):
        q_dict = joblib.load(ruta)
        # Recargar el defaultdict desde el dict guardado
        self.q_table = defaultdict(lambda: defaultdict(float), 
                                  {k: defaultdict(float, v) for k, v in q_dict.items()})
        self.epsilon = self.epsilon_min # Un agente cargado ya no explora
        print(f"Modelo (Tabla-Q) cargado de {ruta}")