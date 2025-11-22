# Proyecto_Modulo_10/agente_rl_snake.py

import numpy as np
import random
from collections import defaultdict
import joblib
import os

class AgenteSnakeQL:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_min=0.01, exploration_decay=0.9995):
        """
        Agente Q-Learning para Snake
        """
        # Q-Table: diccionario de diccionarios
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Hiperparámetros
        self.alpha = learning_rate          # Tasa de aprendizaje
        self.gamma = discount_factor        # Factor de descuento
        self.epsilon = exploration_rate     # Tasa de exploración inicial
        self.epsilon_min = exploration_min  # Exploración mínima
        self.epsilon_decay = exploration_decay
        
        # Estadísticas
        self.episodios_entrenados = 0
        self.recompensa_total_historico = []
    
    def elegir_accion(self, estado, acciones_validas):
        """
        Política epsilon-greedy:
        - Exploración: acción aleatoria
        - Explotación: mejor acción conocida
        """
        if random.uniform(0, 1) < self.epsilon:
            # EXPLORACIÓN: Acción aleatoria válida
            return random.choice(acciones_validas)
        else:
            # EXPLOTACIÓN: Mejor acción según Q-Table
            q_valores = [self.q_table[estado][accion] for accion in acciones_validas]
            max_q = max(q_valores)
            
            # Si hay empate, elegir aleatoriamente entre las mejores
            mejores_acciones = [
                acciones_validas[i] for i in range(len(acciones_validas))
                if q_valores[i] == max_q
            ]
            
            return random.choice(mejores_acciones)
    
    def aprender(self, estado, accion, recompensa, siguiente_estado, 
                 acciones_validas_siguiente, game_over):
        """
        Actualización Q-Learning (Ecuación de Bellman)
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        """
        # Valor Q actual
        q_actual = self.q_table[estado][accion]
        
        # Valor Q máximo del siguiente estado
        if game_over or not acciones_validas_siguiente:
            max_q_siguiente = 0.0
        else:
            q_valores_siguiente = [
                self.q_table[siguiente_estado][a] 
                for a in acciones_validas_siguiente
            ]
            max_q_siguiente = max(q_valores_siguiente)
        
        # Actualización Q-Learning
        nuevo_q = q_actual + self.alpha * (
            recompensa + self.gamma * max_q_siguiente - q_actual
        )
        
        self.q_table[estado][accion] = nuevo_q
    
    def reducir_exploracion(self):
        """Reduce epsilon (menos exploración con el tiempo)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def guardar_modelo(self, ruta="models/agente_snake_ql.joblib"):
        """Guarda la Q-Table y configuración del agente"""
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        
        modelo = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episodios': self.episodios_entrenados,
            'historial': self.recompensa_total_historico
        }
        
        joblib.dump(modelo, ruta)
        print(f"✅ Modelo guardado en: {ruta}")
        print(f"   - Estados explorados: {len(self.q_table)}")
        print(f"   - Epsilon actual: {self.epsilon:.4f}")
        print(f"   - Episodios entrenados: {self.episodios_entrenados}")
    
    def cargar_modelo(self, ruta="models/agente_snake_ql.joblib"):
        """Carga la Q-Table y configuración"""
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el modelo en: {ruta}")
        
        modelo = joblib.load(ruta)
        
        # Reconstruir Q-Table
        q_dict = modelo['q_table']
        self.q_table = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in q_dict.items()}
        )
        
        self.alpha = modelo.get('alpha', self.alpha)
        self.gamma = modelo.get('gamma', self.gamma)
        self.epsilon = modelo.get('epsilon', self.epsilon_min)
        self.episodios_entrenados = modelo.get('episodios', 0)
        self.recompensa_total_historico = modelo.get('historial', [])
        
        print(f"✅ Modelo cargado desde: {ruta}")
        print(f"   - Estados en memoria: {len(self.q_table)}")
        print(f"   - Episodios previos: {self.episodios_entrenados}")
    
    def modo_juego(self):
        """Configura el agente para jugar (sin exploración)"""
        self.epsilon = 0.0
        print("🎮 Modo juego activado (explotación pura)")