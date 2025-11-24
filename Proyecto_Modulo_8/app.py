import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_PATH = 'data/international_matches.csv' 
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_futbol_v2.pkl') 
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_v2.pkl') 

def determinar_resultado(row):
    """Crea la variable objetivo (y)"""
    if row['home_team_score'] > row['away_team_score']:
        return 2  # Gana Local
    elif row['home_team_score'] == row['away_team_score']:
        return 1  # Empate
    else:
        return 0  # Gana Visitante

def entrenar_nuevo_modelo_v2():
    print("Iniciando entrenamiento del Modelo v2 (basado en Rankings)...")
    
    # Cargar Datos
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{DATA_PATH}'.")
        print("Asegúrate de que esté en la carpeta 'data/'.")
        return
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        return

    # Limpieza y Preparación
    df['date'] = pd.to_datetime(df['date'])
    
    columnas_necesarias = [
        'date', 'home_team', 'away_team', 
        'home_team_score', 'away_team_score', 
        'home_team_fifa_rank', 'away_team_fifa_rank'
    ]
    
    for col in columnas_necesarias:
        if col not in df.columns:
            print(f"Error: La columna '{col}' no se encuentra en tu CSV.")
            return
            
    df = df[columnas_necesarias]
    
    # Eliminar filas donde no hay datos de ranking o puntajes
    df = df.dropna(subset=[
        'home_team_fifa_rank', 'away_team_fifa_rank', 
        'home_team_score', 'away_team_score'
    ])
    
    # Convertir puntajes a numéricos
    df['home_team_score'] = pd.to_numeric(df['home_team_score'])
    df['away_team_score'] = pd.to_numeric(df['away_team_score'])

    df['resultado'] = df.apply(determinar_resultado, axis=1)
    df['rank_difference'] = df['home_team_fifa_rank'] - df['away_team_fifa_rank']
    
    features = ['home_team_fifa_rank', 'away_team_fifa_rank', 'rank_difference']
    target = 'resultado'
    
    X = df[features]
    y = df[target]
    
    # Escalar los Datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entrenar el Modelo
    print("Entrenando RandomForestClassifier con features de ranking...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y) 
    
    print("¡Modelo v2 entrenado exitosamente!")

    # Guardar Modelo y Scaler
    if os.path.exists(os.path.join(MODEL_DIR, 'modelo_futbol_v1.pkl')):
        os.remove(os.path.join(MODEL_DIR, 'modelo_futbol_v1.pkl'))
    if os.path.exists(os.path.join(MODEL_DIR, 'label_encoder_teams.pkl')):
        os.remove(os.path.join(MODEL_DIR, 'label_encoder_teams.pkl'))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Nuevo modelo (v2) guardado en: {MODEL_PATH}")
    print(f"Nuevo scaler (v2) guardado en: {SCALER_PATH}")
    print("\n¡Entrenamiento v2 completado!")

if __name__ == "__main__":
    entrenar_nuevo_modelo_v2()