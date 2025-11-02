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

print("Iniciando el proceso de entrenamiento...")

# --- 1. Carga y Preparación de Datos ---
try:
    df = pd.read_csv("netflix_titles.csv")
except FileNotFoundError:
    print("Error: No se encontró 'netflix_titles.csv'.")
    print("Por favor, descárgalo de Kaggle y colócalo en esta carpeta.")
    exit()

# Nos quedamos solo con películas (por simplicidad) y limpiamos nulos
df = df[df['type'] == 'Movie'].copy()
df = df.dropna(subset=['description', 'listed_in'])

print(f"Datos cargados: {len(df)} películas.")

# --- 2. Limpieza de Texto (Simple) ---
def limpiar_texto(texto):
    texto = texto.lower() # minúsculas
    texto = re.sub(f'[{re.escape(string.punctuation)}]', '', texto) # quitar puntuación
    texto = re.sub(r'\d+', '', texto) # quitar números
    return texto.strip()

df['description_clean'] = df['description'].apply(limpiar_texto)

# --- 3. Preparación de Etiquetas (Y) ---
# 'listed_in' es un string "Action & Adventure, Comedies"
# Lo convertimos en una lista: ["Action & Adventure", "Comedies"]
df['genres_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])

# Usamos MultiLabelBinarizer para convertir las listas de géneros
# en un formato binario (ej. [1, 0, 1, 0, 0...])
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres_list'])

# --- 4. Preparación de Texto (X) ---
X = df['description_clean']

# Dividimos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Datos divididos y pre-procesados.")

# --- 5. Creación del Pipeline de Modelo ---
# Un Pipeline junta varios pasos en uno solo.
# 1. TfidfVectorizer: Convierte el texto en números (vectores TF-IDF)
#    - stop_words='english': ignora palabras comunes ("the", "a", "is")
#    - max_features=5000: se queda con las 5000 palabras más importantes
#
# 2. OneVsRestClassifier: Es la estrategia para multi-etiqueta.
#    Crea un clasificador (LinearSVC) para CADA género posible.
#    (Ej. un clasificador para "Acción", otro para "Drama", etc.)
#
# 3. LinearSVC: Un clasificador muy rápido y eficiente para texto.

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', OneVsRestClassifier(LinearSVC(C=1.0, dual=True, max_iter=2000), n_jobs=-1)),
])

# --- 6. Entrenamiento ---
print("Entrenando el modelo... (Esto puede tardar varios minutos)")
pipeline.fit(X_train, y_train)
print("¡Entrenamiento completo!")

# --- 7. Evaluación (Opcional) ---
score = pipeline.score(X_test, y_test)
print(f"Precisión (Accuracy) en datos de prueba: {score:.4f}")

# --- 8. Guardar el Modelo y el Binarizer ---
# Guardamos el pipeline entrenado
joblib.dump(pipeline, 'modelo_generos.joblib')
# Guardamos el binarizer (es crucial para decodificar las predicciones)
joblib.dump(mlb, 'binarizer_generos.joblib')

print("¡Modelo y Binarizer guardados como 'modelo_generos.joblib' y 'binarizer_generos.joblib'!")