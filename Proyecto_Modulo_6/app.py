import pandas as pd
from pysentimiento import create_analyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys

# Importa tu clave de API desde el archivo config.py
try:
    from config import API_KEY_YOUTUBE
except ImportError:
    print("Error: No se encontró el archivo 'config.py' o la variable 'API_KEY_YOUTUBE'.")
    print("Por favor, crea config.py y añade tu API_KEY_YOUTUBE.")
    API_KEY_YOUTUBE = None

def obtener_comentarios_youtube(video_id, api_key):
    """
    Extrae los comentarios de un video de YouTube usando la API v3.
    """
    if not api_key:
        print("No hay API key. Terminando extracción.")
        return []

    print(f"Extrayendo comentarios para el Video ID: {video_id}...")
    comentarios = []
    
    try:
        # Construye el servicio de la API
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Llama a la API para obtener los hilos de comentarios
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

            request = youtube.commentThreads().list_next(request, response)

    except HttpError as e:
        print(f"Error al llamar a la API de YouTube: {e}")
        error_details = e.content.decode()
        
        if "commentsDisabled" in error_details:
            print("Error: Los comentarios están desactivados para este video.")
        elif e.resp.status == 403:
            print("Error 403: Revisa si tu API Key es correcta o si has superado la cuota.")
        else:
            print(f"Detalles del error: {error_details}")
        
        return [] 
    
    print(f"Se extrajeron {len(comentarios)} comentarios.")
    return comentarios

def analizar_sentimiento_proyecto():
    """
    Función principal para extraer comentarios, analizarlos y mostrar un resumen.
    """
    print("--- Iniciando Módulo 6: Análisis de Sentimiento con API de YouTube ---")
    
    if not API_KEY_YOUTUBE:
        print("Terminando script por falta de API Key en config.py.")
        sys.exit(1) 

    # EXTRACCIÓN DE DATOS POR CONSOLA
    VIDEO_ID_EJEMPLO = 'D5d5xinZI3E' 
    
    lista_comentarios = obtener_comentarios_youtube(VIDEO_ID_EJEMPLO, API_KEY_YOUTUBE)
    
    if not lista_comentarios:
        print("No se pudieron obtener comentarios. Terminando el script.")
        return

    # Convertir la lista a un DataFrame de Pandas
    df = pd.DataFrame(lista_comentarios, columns=['comentario'])
    print("Cargando modelo de sentimiento (esto puede tardar la primera vez)...")
    analyzer = create_analyzer(task="sentiment", lang="es")
    comentarios_lista = df['comentario'].tolist()

    print(f"Analizando {len(comentarios_lista)} comentarios (en lote)...")
    # Analizar TODOS los comentarios en un solo lote 
    resultados = analyzer.predict(comentarios_lista)

    # Extraer solo la etiqueta de sentimiento (POS, NEG, NEU) de los resultados
    df['sentimiento'] = [r.output for r in resultados]
    
    # MOSTRAR RESULTADOS
    print("\n" + "="*50)
    print("--- Análisis Detallado por Comentario (Primeros 20) ---")
    print(df.head(20))
    print("="*50 + "\n")
    
    conteo_sentimientos = df['sentimiento'].value_counts()
    
    print("Resumen General del Sentimiento")
    print(conteo_sentimientos)
    
    positivos = conteo_sentimientos.get('POS', 0)
    negativos = conteo_sentimientos.get('NEG', 0)
    neutrales = conteo_sentimientos.get('NEU', 0)
    
    print("\nConclusión")
    if positivos > negativos:
        print(f"El sentimiento general es POSITIVO (Hay más apoyo).")
        print(f"({positivos} positivos vs {negativos} negativos)")
    elif negativos > positivos:
        print(f"El sentimiento general es NEGATIVO (Hay menos apoyo).")
        print(f"({negativos} negativos vs {positivos} positivos)")
    else:
        print("El sentimiento general es NEUTRAL o MIXTO.")

# Ejecutar la función principal
if __name__ == "__main__":
    analizar_sentimiento_proyecto()