import streamlit as st
import pandas as pd
from pysentimiento import create_analyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
import re

# Importa tu clave de API desde el archivo config.py
try:
    from config import API_KEY_YOUTUBE
except ImportError:
    st.error("Error: No se encontró el archivo 'config.py' o la variable 'API_KEY_YOUTUBE'.")
    st.stop()

# --- Funciones de Lógica Optimizadas (Con Caché) ---

@st.cache_resource
def cargar_modelo_sentimiento():
    """
    PASO 1: Carga el modelo de pysentimiento una sola vez y lo guarda en
    la caché de RECURSOS. Esto es lento la PRIMERA vez que se ejecuta.
    """
    print("CACHE MISS (RECURSO): Cargando modelo de sentimiento en memoria...")
    return create_analyzer(task="sentiment", lang="es")

@st.cache_data
def obtener_comentarios_youtube(video_id):
    """
    PASO 2: Extrae los comentarios de un video. Se guarda en la caché de DATOS
    basado en el 'video_id'. Solo llamará a la API de YouTube una vez por video.
    """
    if not API_KEY_YOUTUBE:
        st.error("No hay API key. Terminando extracción.")
        return None

    print(f"CACHE MISS (DATOS/API): Extrayendo comentarios para el Video ID: {video_id}...")
    comentarios = []
    
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY_YOUTUBE)
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100
        )
        
        while request:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comentarios.append(comment)
            
            # Romper el bucle si no hay más páginas
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list_next(request, response)
            else:
                request = None # Termina el bucle
            
            # Límite de seguridad para no gastar toda la cuota en pruebas
            # Puedes aumentar o quitar este límite
            if len(comentarios) >= 1000: 
                print("Límite de 1000 comentarios alcanzado.")
                break 

    except HttpError as e:
        error_details = e.content.decode()
        if "commentsDisabled" in error_details:
            st.error("Error: Los comentarios están desactivados para este video.")
        else:
            st.error(f"Error al llamar a la API de YouTube: {e}")
        return None
    
    print(f"Se extrajeron {len(comentarios)} comentarios.")
    return comentarios

@st.cache_data
def analizar_sentimientos_en_lote(comentarios_lista):
    """
    PASO 3: Analiza una lista de comentarios. Se guarda en la caché de DATOS
    basado en el contenido de la 'comentarios_lista'.
    Solo analizará la misma lista una vez.
    """
    if not comentarios_lista:
        return pd.DataFrame(columns=['comentario', 'sentimiento'])

    print(f"CACHE MISS (DATOS/Análisis): Analizando {len(comentarios_lista)} comentarios...")
    
    # Carga el modelo (esto es rápido, viene de la caché de RECURSOS)
    analyzer = cargar_modelo_sentimiento() 
    
    # Predecir
    resultados = analyzer.predict(comentarios_lista)
    
    # Crear DataFrame
    df = pd.DataFrame(comentarios_lista, columns=['comentario'])
    df['sentimiento'] = [r.output for r in resultados]
    return df

def extraer_video_id(url):
    """
    Extrae el Video ID de diferentes formatos de URL de YouTube.
    """
    match = re.search(r"(?:v=|\/|youtu\.be\/|embed\/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return None

# --- Interfaz de Usuario (Streamlit) ---

st.set_page_config(page_title="Análisis de Sentimiento", layout="wide")
st.title("Módulo 6: Analizador de Sentimiento de YouTube")

# Cargar el modelo al inicio. Gracias a @st.cache_resource,
# esto solo lo "siente" el usuario la primera vez que carga la página.
with st.spinner("Cargando modelo de IA... (solo la primera vez)"):
    cargar_modelo_sentimiento()

# 1. Entrada del usuario
url_video = st.text_input("Pega la URL del video de YouTube que quieres analizar:", 
                          placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analizar Video"):
    if not url_video:
        st.warning("Por favor, introduce una URL.")
    else:
        video_id = extraer_video_id(url_video)
        
        if not video_id:
            st.error("URL no válida. No se pudo extraer un ID de video.")
        else:
            
            # --- Proceso Optimizado ---
            
            # 1. Obtener comentarios (Rápido si ya está en caché)
            with st.spinner(f"Paso 1/2: Extrayendo comentarios (ID: {video_id})..."):
                lista_comentarios = obtener_comentarios_youtube(video_id)
            
            if lista_comentarios:
                st.success(f"¡Se obtuvieron {len(lista_comentarios)} comentarios!")
                
                # 2. Analizar sentimientos (Rápido si ya está en caché)
                with st.spinner("Paso 2/2: Analizando sentimientos..."):
                    df_analizado = analizar_sentimientos_en_lote(lista_comentarios)

                # 3. Mostrar Resultados (Instantáneo)
                st.subheader("Resumen General del Sentimiento")
                
                conteo_sentimientos = df_analizado['sentimiento'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Positivos", conteo_sentimientos.get('POS', 0))
                col2.metric("❌ Negativos", conteo_sentimientos.get('NEG', 0))
                col3.metric("➖ Neutrales", conteo_sentimientos.get('NEU', 0))
                
                # Gráfico de barras
                st.bar_chart(conteo_sentimientos)

                # Conclusión
                positivos = conteo_sentimientos.get('POS', 0)
                negativos = conteo_sentimientos.get('NEG', 0)
                if positivos > negativos:
                    st.success(f"Conclusión: El sentimiento general es POSITIVO (Hay más apoyo). ({positivos} vs {negativos})")
                elif negativos > positivos:
                    st.error(f"Conclusión: El sentimiento general es NEGATIVO (Hay menos apoyo). ({negativos} vs {positivos})")
                else:
                    st.info("Conclusión: El sentimiento general es NEUTRAL o MIXTO.")
                
                # Mostrar datos detallados
                st.subheader("Análisis Detallado de Comentarios")
                st.dataframe(df_analizado, use_container_width=True)
            
            elif lista_comentarios is None:
                # Esto maneja el caso donde la API falló (ej. comentarios desactivados)
                st.error("No se pudo completar el análisis (revisa el error de arriba).")
            else:
                # Esto maneja el caso de 0 comentarios
                st.info("El video no tiene comentarios para analizar.")