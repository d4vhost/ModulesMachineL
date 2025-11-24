import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
from collections import defaultdict

DATA_PATH = 'data/international_matches.csv' 
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_futbol_v2.pkl') 
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_v2.pkl')

@st.cache_resource
def cargar_modelo_y_rankings_v2():
    """Carga el modelo v2, el scaler v2, y los rankings más recientes."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: No se encontraron '{MODEL_PATH}' o '{SCALER_PATH}'.")
        st.error("Por favor, ejecuta 'app.py' (el de rankings) primero.")
        return None, None, None
        
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['home_team_fifa_rank', 'away_team_fifa_rank'])
        
        # Obtenemos el ranking MÁS RECIENTE de cada equipo
        df_home = df[['date', 'home_team', 'home_team_fifa_rank']].rename(
            columns={'home_team': 'equipo', 'home_team_fifa_rank': 'ranking'}
        )
        df_away = df[['date', 'away_team', 'away_team_fifa_rank']].rename(
            columns={'away_team': 'equipo', 'away_team_fifa_rank': 'ranking'}
        )
        
        df_rankings = pd.concat([df_home, df_away]).sort_values(by='date')
        rankings_actuales = df_rankings.drop_duplicates(subset='equipo', keep='last')
        
        ranking_dict = rankings_actuales.set_index('equipo')['ranking'].to_dict()
        st.session_state.total_equipos_v2 = len(ranking_dict)
        return model, scaler, ranking_dict
        
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{DATA_PATH}'.")
        return None, None, None

def predecir_partido_v2(modelo, scaler, ranking_dict, equipo_local, equipo_visitante, es_eliminatoria=False):
    """Predice el resultado de un partido usando el modelo de RANKING."""
    try:
        rank_local = ranking_dict[equipo_local]
        rank_visitante = ranking_dict[equipo_visitante]
    except KeyError as e:
        equipo_faltante = str(e).strip("'")
        st.warning(f"Error: El equipo '{equipo_faltante}' no tiene ranking. Se asignará un ranking promedio (100).")
        if equipo_faltante == equipo_local: rank_local = 100
        if equipo_faltante == equipo_visitante: rank_visitante = 100
        if equipo_faltante not in ranking_dict: ranking_dict[equipo_faltante] = 100

    rank_diff = rank_local - rank_visitante
    features = np.array([[rank_local, rank_visitante, rank_diff]])
    features_scaled = scaler.transform(features)
    probabilidades = modelo.predict_proba(features_scaled)[0]
    
    if es_eliminatoria:
        prob_local = probabilidades[2] + (probabilidades[1] / 2)
        prob_visitante = probabilidades[0] + (probabilidades[1] / 2)
        return equipo_local if np.random.rand() < (prob_local / (prob_local + prob_visitante)) else equipo_visitante
    else:
        resultado = np.random.choice([0, 1, 2], p=probabilidades)
        if resultado == 2: return equipo_local
        elif resultado == 1: return "Empate"
        else: return equipo_visitante

def simular_grupo_v2(modelo, scaler, ranking_dict, equipos_grupo):
    """Simula una fase de grupos completa. Devuelve la tabla de posiciones."""
    tabla = defaultdict(lambda: {'Puntos': 0, 'PJ': 0, 'G': 0, 'E': 0, 'P': 0, 'GF': 0, 'GC': 0, 'DG': 0})
    
    for i in range(len(equipos_grupo)):
        for j in range(i + 1, len(equipos_grupo)):
            local, visitante = equipos_grupo[i], equipos_grupo[j]
            ganador = predecir_partido_v2(modelo, scaler, ranking_dict, local, visitante)
            
            goles_local = np.random.poisson(1.4); goles_visitante = np.random.poisson(1.1)
            tabla[local]['PJ'] += 1; tabla[visitante]['PJ'] += 1
            
            if ganador == local:
                tabla[local]['Puntos'] += 3; tabla[local]['G'] += 1; tabla[visitante]['P'] += 1
                if goles_local == goles_visitante: goles_local += 1
            elif ganador == visitante:
                tabla[visitante]['Puntos'] += 3; tabla[visitante]['G'] += 1; tabla[local]['P'] += 1
                if goles_local == goles_visitante: goles_visitante += 1
            else:
                tabla[local]['Puntos'] += 1; tabla[visitante]['Puntos'] += 1; tabla[local]['E'] += 1; tabla[visitante]['E'] += 1
                goles_local = goles_visitante

            tabla[local]['GF'] += goles_local; tabla[local]['GC'] += goles_visitante
            tabla[visitante]['GF'] += goles_visitante; tabla[visitante]['GC'] += goles_local
    
    for equipo in tabla: tabla[equipo]['DG'] = tabla[equipo]['GF'] - tabla[equipo]['GC']
    
    df_tabla = pd.DataFrame.from_dict(tabla, orient='index')
    df_tabla.index.name = 'Equipo'
    return df_tabla.sort_values(by=['Puntos', 'DG', 'GF'], ascending=False)

def simular_ronda_eliminatoria_interactiva_v2(modelo, scaler, ranking_dict, equipos, nombre_ronda):
    """Simula una ronda de eliminación Y muestra la tabla de resultados."""
    ganadores = []
    partidos = []
    st.subheader(f"Resultados de {nombre_ronda}")
    
    for i in range(0, len(equipos), 2):
        local, visitante = equipos[i], equipos[i+1]
        ganador = predecir_partido_v2(modelo, scaler, ranking_dict, local, visitante, es_eliminatoria=True)
        ganadores.append(ganador)
        partidos.append({"Partido": f"{local} vs {visitante}", "GANADOR": ganador})
        
    st.dataframe(pd.DataFrame(partidos), use_container_width=True)
    return ganadores

# Interfaz Gráfica de Streamlit 
st.set_page_config(page_title="Simulador Mundial 2026", layout="wide")
st.title("🏆 Simulador de la Copa Mundial de Fútbol 2026")

modelo_v2, scaler_v2, rankings_actuales = cargar_modelo_y_rankings_v2()

if 'total_equipos_v2' not in st.session_state: st.session_state.total_equipos_v2 = 0

st.markdown(f"Esta app usa el **modelo v2 (basado en Rankings)** entrenado con **{st.session_state.total_equipos_v2} equipos** de tu CSV.")
st.markdown("Los 48 equipos se eligen **basado en los 48 mejores rankings** y se sortean en **4 Bombos** para un torneo realista.")
st.markdown("---")

if modelo_v2 and scaler_v2 and rankings_actuales:

    # Ordenamos los equipos por su ranking (1 es el mejor)
    equipos_ordenados = sorted(rankings_actuales, key=rankings_actuales.get)
    
    # Seleccionamos los 48 MEJORES equipos para el mundial
    equipos_para_sorteo = equipos_ordenados[:48]
    
    st.subheader("Equipos Top 48 (Clasificados para esta simulación):")
    st.expander("Ver los 48 equipos clasificados").write(f"{', '.join(equipos_para_sorteo)}")
    st.markdown("---")

    if st.button("Simular MUNDIAL 2026 (con Sorteo por Bombos)", type="primary"):
        
        st.header("Fase de Grupos")
        with st.spinner("Creando 4 Bombos, sorteando 12 grupos y simulando partidos... ⚽"):
            
            # Tomamos los 48 mejores por ranking
            bombo1 = equipos_para_sorteo[0:12]  
            bombo2 = equipos_para_sorteo[12:24] 
            bombo3 = equipos_para_sorteo[24:36] 
            bombo4 = equipos_para_sorteo[36:48] 

            # Mezclamos CADA bombo por separado
            random.shuffle(bombo1)
            random.shuffle(bombo2)
            random.shuffle(bombo3)
            random.shuffle(bombo4)

            grupos_sorteados = {}
            for i in range(12):
                nombre_grupo = f"Grupo {chr(ord('A') + i)}"
                grupos_sorteados[nombre_grupo] = [
                    bombo1[i],
                    bombo2[i],
                    bombo3[i],
                    bombo4[i]
                ]

            clasificados_directos = [] 
            mejores_terceros = pd.DataFrame()
            cols = st.columns(3) 
            
            for i, (nombre_grupo, grupo) in enumerate(grupos_sorteados.items()):
                tabla_df = simular_grupo_v2(modelo_v2, scaler_v2, rankings_actuales, grupo)
                
                clasificados_directos.extend(tabla_df.index[:2].tolist())
                mejores_terceros = pd.concat([mejores_terceros, tabla_df.iloc[2:3]])
                
                with cols[i % 3]:
                    st.markdown(f"**{nombre_grupo}**")
                    st.dataframe(tabla_df)
        
        st.success("¡Fase de Grupos completada!")
        st.markdown("---")
        
        st.header("Clasificación de 3ros Lugares")
        mejores_terceros = mejores_terceros.sort_values(by=['Puntos', 'DG', 'GF'], ascending=False)
        clasificados_terceros = mejores_terceros.index[:8].tolist()
        st.write("Los 8 mejores 3ros lugares que clasifican:")
        st.dataframe(mejores_terceros.head(8), use_container_width=True)
        
        # Lista final de 32
        clasificados_r32 = clasificados_directos + clasificados_terceros
        random.shuffle(clasificados_r32) 
        
        st.markdown("---")
        st.header("Fase de Eliminatorias")
        
        with st.spinner("Simulando todas las fases eliminatorias... 🥅"):
            clasificados_r16 = simular_ronda_eliminatoria_interactiva_v2(modelo_v2, scaler_v2, rankings_actuales, clasificados_r32, "Dieciseisavos de Final (Ronda de 32)")
            clasificados_qf = simular_ronda_eliminatoria_interactiva_v2(modelo_v2, scaler_v2, rankings_actuales, clasificados_r16, "Octavos de Final")
            clasificados_sf = simular_ronda_eliminatoria_interactiva_v2(modelo_v2, scaler_v2, rankings_actuales, clasificados_qf, "Cuartos de Final")
            clasificados_final = simular_ronda_eliminatoria_interactiva_v2(modelo_v2, scaler_v2, rankings_actuales, clasificados_sf, "Semifinales")
            campeon = simular_ronda_eliminatoria_interactiva_v2(modelo_v2, scaler_v2, rankings_actuales, clasificados_final, "GRAN FINAL")

        st.markdown("---")
        
        st.header(f"¡El CAMPEÓN de la simulación MUNDIAL 2026 es: {campeon[0]}! 🏆")
        
else:
    st.error("No se pudo cargar el modelo v2 o los rankings.")
    st.warning("Asegúrate de que 'international_matches.csv' esté en 'data/' y haber ejecutado el 'app.py' (el de rankings) al menos una vez.")