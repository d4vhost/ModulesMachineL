# Proyecto_Modulo_10/entrenar.py

from juego_snake import SnakeGame
from agente_rl_snake import AgenteSnakeQL
import numpy as np

def entrenar_agente(episodios=50000, grid_width=20, grid_height=15, guardar_cada=5000):
    """
    Entrena el agente Q-Learning para jugar Snake
    """
    print("="*60)
    print("🐍 ENTRENAMIENTO DE AGENTE SNAKE CON Q-LEARNING")
    print("="*60)
    print(f"📊 Episodios: {episodios}")
    print(f"📏 Tamaño del tablero: {grid_width}x{grid_height}")
    print(f"💾 Guardado cada: {guardar_cada} episodios")
    print("="*60)
    
    # Inicializar juego y agente
    juego = SnakeGame(grid_width=grid_width, grid_height=grid_height, vidas_iniciales=3)
    agente = AgenteSnakeQL(
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_min=0.01,
        exploration_decay=0.9995
    )
    
    # Estadísticas
    puntos_por_episodio = []
    longitud_maxima_por_episodio = []
    mejor_puntuacion = 0
    
    for episodio in range(1, episodios + 1):
        estado = juego.reset_completo()
        max_longitud = 1
        
        while not juego.game_over:
            # Elegir acción
            acciones_validas = juego.get_acciones_validas()
            accion = agente.elegir_accion(estado, acciones_validas)
            
            # Ejecutar acción
            siguiente_estado, recompensa, perdio_vida, game_over = juego.step(accion)
            acciones_validas_siguiente = juego.get_acciones_validas()
            
            # Aprender
            agente.aprender(
                estado, accion, recompensa, 
                siguiente_estado, acciones_validas_siguiente, game_over
            )
            
            # Actualizar estado
            estado = siguiente_estado
            max_longitud = max(max_longitud, len(juego.snake))
        
        # Reducir exploración
        agente.reducir_exploracion()
        agente.episodios_entrenados += 1
        
        # Guardar estadísticas
        info = juego.get_info()
        puntos_por_episodio.append(info['puntos'])
        longitud_maxima_por_episodio.append(max_longitud)
        
        if info['puntos'] > mejor_puntuacion:
            mejor_puntuacion = info['puntos']
        
        # Mostrar progreso
        if episodio % 1000 == 0:
            promedio_puntos = np.mean(puntos_por_episodio[-1000:])
            promedio_longitud = np.mean(longitud_maxima_por_episodio[-1000:])
            
            print(f"Episodio {episodio}/{episodios}")
            print(f"  🏆 Mejor puntuación: {mejor_puntuacion}")
            print(f"  🎯 Prom. Puntos (últimos 1000): {promedio_puntos:.2f}")
            print(f"  📏 Prom. Longitud (últimos 1000): {promedio_longitud:.2f}")
            print(f"  🔍 Epsilon (exploración): {agente.epsilon:.4f}")
            print(f"  🧠 Estados en Q-Table: {len(agente.q_table)}")
            print("-"*60)
        
        # Guardar modelo periódicamente
        if episodio % guardar_cada == 0:
            agente.guardar_modelo()
    
    # Guardar modelo final
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    agente.guardar_modelo()
    print(f"🏆 Mejor puntuación alcanzada: {mejor_puntuacion}")
    print(f"🧠 Total de estados explorados: {len(agente.q_table)}")

if __name__ == "__main__":
    entrenar_agente(
        episodios=50000,
        grid_width=20,   # Ancho
        grid_height=15,  # Alto
        guardar_cada=5000
    )