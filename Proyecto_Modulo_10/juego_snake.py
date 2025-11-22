# Proyecto_Modulo_10/juego_snake.py

import numpy as np
import random

class SnakeGame:
    def __init__(self, grid_width=20, grid_height=15, vidas_iniciales=3):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.vidas_totales = vidas_iniciales
        self.vidas = vidas_iniciales
        self.record = 0
        self.reset()
    
    def reset(self):
        """Reinicia posición pero MANTIENE el tamaño de la serpiente"""
        # Si ya existe una serpiente, guardar su tamaño
        longitud_actual = len(self.snake) if hasattr(self, 'snake') else 1
        
        # Serpiente empieza en el centro con el mismo tamaño
        centro_x = self.grid_width // 2
        centro_y = self.grid_height // 2
        
        # Crear serpiente con la longitud actual
        self.snake = [(centro_y, centro_x)]
        for i in range(1, longitud_actual):
            # Añadir segmentos hacia la izquierda
            self.snake.append((centro_y, centro_x - i))
        
        self.direccion = 'RIGHT'  # Dirección inicial fija
        
        # Colocar comida aleatoria
        self.spawn_food()
        
        # Estado del juego
        self.game_over = False
        self.pasos = 0
        self.pasos_sin_comida = 0
        self.max_pasos_sin_comida = self.grid_width * self.grid_height * 2
        
        return self.get_estado()
    
    def reset_completo(self):
        """Reset completo con vidas y puntos restaurados"""
        self.vidas = self.vidas_totales
        self.puntos = 0
        self.snake = None  # Forzar reset de tamaño
        return self.reset()
    
    def spawn_food(self):
        """Genera comida en posición aleatoria (que no sea la serpiente)"""
        max_intentos = 100
        for _ in range(max_intentos):
            self.food = (
                random.randint(0, self.grid_height - 1),
                random.randint(0, self.grid_width - 1)
            )
            if self.food not in self.snake:
                break
    
    def get_estado(self):
        """
        Estado del juego para la IA:
        - Peligro en cada dirección (arriba, abajo, izq, der)
        - Dirección de la comida (8 direcciones)
        - Dirección actual de movimiento
        """
        cabeza = self.snake[0]
        
        # Detectar peligros (pared o cuerpo) en 4 direcciones
        peligro_arriba = self._hay_peligro(cabeza[0] - 1, cabeza[1])
        peligro_abajo = self._hay_peligro(cabeza[0] + 1, cabeza[1])
        peligro_izq = self._hay_peligro(cabeza[0], cabeza[1] - 1)
        peligro_der = self._hay_peligro(cabeza[0], cabeza[1] + 1)
        
        # Dirección hacia la comida
        comida_arriba = 1 if self.food[0] < cabeza[0] else 0
        comida_abajo = 1 if self.food[0] > cabeza[0] else 0
        comida_izq = 1 if self.food[1] < cabeza[1] else 0
        comida_der = 1 if self.food[1] > cabeza[1] else 0
        
        # Dirección actual de movimiento
        dir_up = 1 if self.direccion == 'UP' else 0
        dir_down = 1 if self.direccion == 'DOWN' else 0
        dir_left = 1 if self.direccion == 'LEFT' else 0
        dir_right = 1 if self.direccion == 'RIGHT' else 0
        
        estado = (
            peligro_arriba, peligro_abajo, peligro_izq, peligro_der,
            comida_arriba, comida_abajo, comida_izq, comida_der,
            dir_up, dir_down, dir_left, dir_right
        )
        
        return estado
    
    def _hay_peligro(self, fila, col):
        """Verifica si hay peligro (pared o cuerpo) en una posición"""
        # Pared
        if fila < 0 or fila >= self.grid_height or col < 0 or col >= self.grid_width:
            return 1
        # Cuerpo de la serpiente
        if (fila, col) in self.snake[:-1]:
            return 1
        return 0
    
    def get_acciones_validas(self):
        """Retorna acciones válidas (no puede ir en dirección contraria)"""
        acciones = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # No puede ir en dirección opuesta
        if self.direccion == 'UP':
            acciones.remove('DOWN')
        elif self.direccion == 'DOWN':
            acciones.remove('UP')
        elif self.direccion == 'LEFT':
            acciones.remove('RIGHT')
        elif self.direccion == 'RIGHT':
            acciones.remove('LEFT')
        
        return acciones
    
    def step(self, accion):
        """
        Ejecuta una acción y retorna (nuevo_estado, recompensa, perdio_vida, game_over)
        """
        if self.game_over:
            return self.get_estado(), 0, False, True
        
        # Cambiar dirección (solo si no es opuesta)
        acciones_validas = self.get_acciones_validas()
        if accion in acciones_validas:
            self.direccion = accion
        
        # Mover la serpiente
        cabeza = self.snake[0]
        
        if self.direccion == 'UP':
            nueva_cabeza = (cabeza[0] - 1, cabeza[1])
        elif self.direccion == 'DOWN':
            nueva_cabeza = (cabeza[0] + 1, cabeza[1])
        elif self.direccion == 'LEFT':
            nueva_cabeza = (cabeza[0], cabeza[1] - 1)
        elif self.direccion == 'RIGHT':
            nueva_cabeza = (cabeza[0], cabeza[1] + 1)
        
        # Verificar colisión con pared
        if (nueva_cabeza[0] < 0 or nueva_cabeza[0] >= self.grid_height or
            nueva_cabeza[1] < 0 or nueva_cabeza[1] >= self.grid_width):
            self.vidas -= 1
            if self.vidas <= 0:
                self.game_over = True
                return self.get_estado(), -100, True, True
            else:
                # Pierde vida pero MANTIENE tamaño
                self.reset()
                return self.get_estado(), -100, True, False
        
        # Verificar colisión con el cuerpo
        if nueva_cabeza in self.snake[:-1]:
            self.vidas -= 1
            if self.vidas <= 0:
                self.game_over = True
                return self.get_estado(), -100, True, True
            else:
                # Pierde vida pero MANTIENE tamaño
                self.reset()
                return self.get_estado(), -100, True, False
        
        # Insertar nueva cabeza
        self.snake.insert(0, nueva_cabeza)
        
        recompensa = 0
        distancia_antes = self._distancia_manhattan(cabeza, self.food)
        distancia_despues = self._distancia_manhattan(nueva_cabeza, self.food)
        
        # Verificar si comió
        if nueva_cabeza == self.food:
            if not hasattr(self, 'puntos'):
                self.puntos = 0
            self.puntos += 10
            recompensa = 50  # PREMIO: Comer comida
            self.spawn_food()
            self.pasos_sin_comida = 0
            
            # Actualizar record
            if self.puntos > self.record:
                self.record = self.puntos
        else:
            # Quitar la cola (no crece)
            self.snake.pop()
            
            # PREMIO/CASTIGO: Acercarse/alejarse de la comida
            if distancia_despues < distancia_antes:
                recompensa = 2  # Se acerca
            else:
                recompensa = -1  # Se aleja
            
            self.pasos_sin_comida += 1
        
        # CASTIGO: Si tarda mucho sin comer (evita loops infinitos)
        if self.pasos_sin_comida > self.max_pasos_sin_comida:
            self.vidas -= 1
            if self.vidas <= 0:
                self.game_over = True
                return self.get_estado(), -50, True, True
            else:
                self.reset()
                return self.get_estado(), -50, True, False
        
        self.pasos += 1
        recompensa += 0.1  # PREMIO: Sobrevivir cada paso
        
        return self.get_estado(), recompensa, False, False
    
    def _distancia_manhattan(self, pos1, pos2):
        """Calcula distancia Manhattan entre dos posiciones"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_info(self):
        """Información del juego para mostrar"""
        if not hasattr(self, 'puntos'):
            self.puntos = 0
        return {
            'puntos': self.puntos,
            'longitud': len(self.snake),
            'pasos': self.pasos,
            'vidas': self.vidas,
            'record': self.record
        }