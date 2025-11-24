import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import joblib
import string
import re
import os # Importar 'os' para manejar rutas

print("Iniciando el proceso de entrenamiento...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Rutas a los archivos
data_path = os.path.join(DATA_DIR, "netflix_titles.csv")
model_path = os.path.join(MODEL_DIR, "modelo_generos.joblib")
mlb_path = os.path.join(MODEL_DIR, "binarizer_generos.joblib")

# Carga y Preparación de Datos
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: No se encontró '{data_path}'.")
    print("Por favor, asegúrate de colocar 'netflix_titles.csv' en la carpeta 'data/raw/'.")
    exit()

df = df[df['type'] == 'Movie'].copy()
df = df.dropna(subset=['description', 'listed_in'])

print(f"Datos cargados: {len(df)} películas.")

# Limpieza de Texto
def limpiar_texto(texto):
    texto = texto.lower() 
    texto = re.sub(f'[{re.escape(string.punctuation)}]', '', texto) 
    texto = re.sub(r'\d+', '', texto) 
    return texto.strip()

df['description_clean'] = df['description'].apply(limpiar_texto)

# Preparación de Etiquetas
df['genres_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])

# Usamos MultiLabelBinarizer para convertir las listas de géneros
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres_list'])

# Preparación de Texto
X = df['description_clean']

# Dividimos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Datos divididos y pre-procesados.")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', OneVsRestClassifier(LinearSVC(C=1.0, dual=True, max_iter=2000), n_jobs=-1)),
])

# Entrenamiento
print("Entrenando el modelo... (Esto puede tardar varios minutos)")
pipeline.fit(X_train, y_train)
print("¡Entrenamiento completo!")

score = pipeline.score(X_test, y_test)
print(f"Precisión (Accuracy) en datos de prueba: {score:.4f}")

# Guardar el Modelo y el Binarizer
joblib.dump(pipeline, model_path)
joblib.dump(mlb, mlb_path)

print(f"¡Modelo y Binarizer guardados en la carpeta '{MODEL_DIR}'!")