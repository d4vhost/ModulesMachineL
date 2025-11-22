# Proyecto_Modulo_10/entrenar.py

from juego import TresEnRaya
from agente_rl import AgenteQLearning
import os

# --- 1. Definir Premios y Castigos ---
PREMIO_GANAR = 1.0     # Premio máximo
PREMIO_EMPATAR = 0.5   # Un premio menor, es mejor que perder
CASTIGO_PERDER = -1.0  # Castigo
CASTIGO_MOV_INVALIDO = -2.0 # Castigo fuerte (aunque el agente no debería hacerlo)
PREMIO_POR_SEGUIR = 0.0 # Premio/Castigo por cada movimiento (0.0 es neutral)

# --- 2. Simulación de Entrenamiento ---

def entrenar_agente(episodios=50000):
    juego = TresEnRaya()
    # Agente 1 (el que queremos entrenar)
    agente_ia = AgenteQLearning()
    
    # Agente 2 (un oponente que también aprende, o puede ser aleatorio)
    # Entrenar contra otro agente que aprende es más robusto
    oponente_ia = AgenteQLearning()

    print(f"Iniciando entrenamiento de {episodios} partidas...")

    for episodio in range(episodios):
        estado = juego.reiniciar()
        turno_agente = True # Agente es Jugador 1 ('1')

        while not juego.juego_terminado:
            agente_actual = agente_ia if turno_agente else oponente_ia
            jugador_actual = 1 if turno_agente else 2
            
            # 1. Elige una acción
            mov_validos = juego.get_mov_validos()
            accion = agente_actual.elegir_accion(estado, mov_validos)
            
            # 2. Realiza la acción y observa el resultado
            juego.realizar_movimiento(accion[0], accion[1], jugador_actual)
            
            proximo_estado = juego.get_estado()
            mov_validos_proximos = juego.get_mov_validos()
            
            # 3. Asignar premio/castigo
            premio = PREMIO_POR_SEGUIR
            if juego.juego_terminado:
                if juego.ganador == jugador_actual:
                    premio = PREMIO_GANAR
                elif juego.ganador == 0:
                    premio = PREMIO_EMPATAR
                else: # Ganó el otro
                    premio = CASTIGO_PERDER

            # 4. Aprender (Actualizar Q-Table)
            agente_actual.aprender(estado, accion, premio, proximo_estado, mov_validos_proximos)
            
            # Si el juego terminó, el oponente también recibe su premio/castigo
            if juego.juego_terminado:
                premio_oponente = 0.0
                if juego.ganador == jugador_actual: premio_oponente = CASTIGO_PERDER
                elif juego.ganador == 0: premio_oponente = PREMIO_EMPATAR
                # (El estado/accion anterior del oponente no se guarda en este loop,
                # para un entrenamiento perfecto, necesitaríamos guardar el 'estado_anterior'
                # del oponente. Por simplicidad, solo el agente actual aprende)
            
            estado = proximo_estado
            turno_agente = not turno_agente

        # Reducir la exploración después de cada partida
        agente_ia.reducir_exploracion()
        oponente_ia.reducir_exploracion()

        if (episodio + 1) % 5000 == 0:
            print(f"Episodio {episodio+1}/{episodios} completado.")

    print("Entrenamiento finalizado.")
    # Guardamos solo el modelo que usaremos (agente_ia)
    agente_ia.guardar_modelo("models/agente_tres_en_raya.joblib")

if __name__ == "__main__":
    entrenar_agente()