import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

codigo_celdas = [
    (nbf.v4.new_markdown_cell, """# Análisis Exploratorio de Datos (EDA) - Proyecto RAG Aeronáutico
**Fase CRISP-DM:** Comprensión de los Datos (Data Understanding)

El presente notebook tiene como objetivo analizar el *corpus* documental generado sintéticamente a partir de la normativa de la Dirección General de Aeronáutica Civil (DGAC) y la OACI. 
Al tratarse de un proyecto de Procesamiento de Lenguaje Natural (NLP), el análisis exploratorio se centrará en la distribución de las fuentes, la justificación matemática de la fragmentación (chunking) para la ventana de contexto del modelo, y la coherencia semántica del espacio vectorial."""),
    
    (nbf.v4.new_code_cell, """import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el Mega Dataset
datos = []
ruta_dataset = "../dataset_sintetico/mega_dataset_aeronautico.jsonl"

with open(ruta_dataset, 'r', encoding='utf-8') as f:
    for linea in f:
        datos.append(json.loads(linea))

df = pd.DataFrame(datos)
print(f"Total de registros cargados: {len(df)}")

# 2. Gráfico de distribución por documento fuente
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='fuente', order=df['fuente'].value_counts().index, palette='viridis')
plt.title('Distribución del Dataset por Documento Normativo (DGAC/OACI)', fontsize=14)
plt.xlabel('Cantidad de Pares Pregunta/Respuesta', fontsize=12)
plt.ylabel('Documento Fuente', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""),

    (nbf.v4.new_code_cell, """# ==========================================
# VERIFICACIÓN DE DATOS FALTANTES (MISSING VALUES)
# Aplicando conceptos de la Unidad 1 del curso
# ==========================================
print("Análisis de Integridad del Dataset:")
print("-" * 40)
print(df.info())
print("\\nConteo de Valores Nulos (NaN) por columna:")
print(df.isnull().sum())

# En caso de existir algún registro donde la IA no generó respuesta, lo eliminamos (Dropna)
df_limpio = df.dropna().copy()
print(f"\\nRegistros tras limpieza de datos faltantes: {len(df_limpio)}")

# Usaremos df_limpio en adelante para garantizar que el modelo no falle
df = df_limpio"""),

    (nbf.v4.new_markdown_cell, """### 1. Análisis de Longitud y Justificación de Fragmentación (Chunking)
Para que el motor de inferencia local (Llama 3) no sature la memoria VRAM de la GPU y pierda atención (Lost in the middle), es crítico que los fragmentos de contexto sean uniformes. El siguiente análisis de distribución demuestra la efectividad de la técnica `RecursiveCharacterTextSplitter` aplicada en la fase de ingesta."""),

    (nbf.v4.new_code_cell, """# Calcular longitudes
df['longitud_contexto'] = df['contexto_original'].apply(len)
df['longitud_respuesta'] = df['respuesta'].apply(len)

# Crear subgráficos
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma del Contexto
sns.histplot(df['longitud_contexto'], bins=30, kde=True, color='royalblue', ax=axes[0])
axes[0].set_title('Distribución de Longitud del Contexto (Caracteres)')
axes[0].set_xlabel('Longitud')
axes[0].set_ylabel('Frecuencia')

# Histograma de las Respuestas
sns.histplot(df['longitud_respuesta'], bins=30, kde=True, color='darkorange', ax=axes[1])
axes[1].set_title('Distribución de Longitud de la Respuesta Generada')
axes[1].set_xlabel('Longitud')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

print(f"Promedio de caracteres por contexto: {df['longitud_contexto'].mean():.2f}")
print(f"Promedio de caracteres por respuesta: {df['longitud_respuesta'].mean():.2f}")"""),

    (nbf.v4.new_markdown_cell, """### 2. Análisis de Complejidad Léxica (Nube de Palabras)
Evaluamos el contenido léxico para comprobar que el proceso de generación sintética preservó la jerga técnica, médica y legal del entorno aeronáutico sin desviaciones ni alucinaciones severas."""),

    (nbf.v4.new_code_cell, """from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re

# Descargar stopwords en español si no están
nltk.download('stopwords', quiet=True)
stop_words_es = set(stopwords.words('spanish'))

# Añadir palabras vacías del entorno que no aportan peso semántico
stop_words_es.update(['si', 'debe', 'ser', 'así', 'cualquier', 'caso', 'puede', 'parte'])

# Unir todas las preguntas generadas
texto_completo = " ".join(df['pregunta'].tolist())

# Generar la nube de palabras
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      stopwords=stop_words_es,
                      colormap='inferno',
                      max_words=100).generate(texto_completo)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Conceptos Más Frecuentes en las Consultas Sintéticas', fontsize=16)
plt.show()"""),

    (nbf.v4.new_markdown_cell, """### 3. Topología del Espacio Vectorial (UMAP) - (Aprendizaje No Supervisado)
Esta es la prueba de viabilidad de la arquitectura RAG. Convertimos una muestra representativa de textos en vectores de 768 dimensiones utilizando el modelo `nomic-embed-text`. Luego, aplicamos el algoritmo UMAP (Uniform Manifold Approximation and Projection) para reducir la dimensionalidad a 2D. 

Si el modelo de *embeddings* comprende la diferencia entre normativas médicas y normativas operacionales, veremos agrupaciones (Clústeres) claramente definidos según el color del documento de origen."""),

    (nbf.v4.new_code_cell, """from ollama import Client
from tqdm.notebook import tqdm
import numpy as np
import umap

cliente = Client(host='http://localhost:11434')
modelo_embeddings = "nomic-embed-text"

# Para el EDA tomamos una muestra representativa estratificada (por rapidez de visualización)
# Tomamos 2000 registros manteniendo la proporción de las fuentes
df_muestra = df.groupby('fuente', group_keys=False).apply(lambda x: x.sample(min(len(x), 300), random_state=42)).reset_index(drop=True)

print("Matematizando textos a vectores de 768 dimensiones...")
embeddings = []
for contexto in tqdm(df_muestra['contexto_original']):
    respuesta = cliente.embeddings(model=modelo_embeddings, prompt=contexto)
    embeddings.append(respuesta['embedding'])

matriz_embeddings = np.array(embeddings)"""),

    (nbf.v4.new_code_cell, """# ==========================================
# REDUCCIÓN DE DIMENSIONALIDAD CON PCA
# Aplicación del algoritmo clásico visto en clases
# ==========================================
from sklearn.decomposition import PCA

print("Aplicando PCA para reducir de 768 a 2 componentes principales...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(matriz_embeddings)

df_muestra['PCA_1'] = embeddings_pca[:, 0]
df_muestra['PCA_2'] = embeddings_pca[:, 1]

# Varianza explicada por los componentes
varianza = pca.explained_variance_ratio_
print(f"Varianza explicada por PC1: {varianza[0]*100:.2f}%")
print(f"Varianza explicada por PC2: {varianza[1]*100:.2f}%")

# Gráfico PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_muestra, x='PCA_1', y='PCA_2', hue='fuente', palette='tab10', alpha=0.7, edgecolor='k')
plt.title('Reducción de Dimensionalidad con PCA (Análisis Clásico)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Nota analítica que puedes mencionar en tu presentación:
# PCA asume relaciones lineales. Como el lenguaje natural es altamente no lineal, 
# la varianza explicada suele ser baja, justificando el uso posterior de técnicas 
# topológicas no lineales como UMAP para una mejor separación de clústeres."""),

    (nbf.v4.new_code_cell, """print("Reduciendo dimensiones con UMAP...")
reductor = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embeddings_2d = reductor.fit_transform(matriz_embeddings)

df_muestra['UMAP_1'] = embeddings_2d[:, 0]
df_muestra['UMAP_2'] = embeddings_2d[:, 1]

# Graficar
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_muestra, x='UMAP_1', y='UMAP_2', hue='fuente', palette='tab10', s=60, alpha=0.8, edgecolor='k')

plt.title('Representación Semántica de las Normativas Aeronáuticas (UMAP)', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Documento Fuente')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()""")
]

for constructor, content in codigo_celdas:
    nb['cells'].append(constructor(content))

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/EDA_Aeronautico.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook EDA_Aeronautico.ipynb generado exitosamente en la carpeta 'notebooks/'.")
