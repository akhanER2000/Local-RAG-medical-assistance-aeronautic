import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

celdas = [
    (nbf.v4.new_markdown_cell, """# Análisis Exploratorio y Preprocesamiento de Datos (Dual-Engine V2 - Factores Humanos)
Este sistema pionero consta de dos ejes arquitectónicos:
1. **Módulo de Consulta Normativa (NLP):** Encargado del análisis profundo de los manuales, directivas y reglas aeronáuticas de la DGAC y la OACI.
2. **Módulo Predictivo de Riesgo Clínico (Factores Humanos):** Se enfoca en predecir patologías y factores humanos críticos para la aviación (como Estrés, Salud Mental, IMC y Fatiga) procesadas con algoritmos clásicos de Machine Learning apoyados en LLMs locales."""),
    
    (nbf.v4.new_markdown_cell, """---
## SECCIÓN A: Módulo Documental (NLP)
---"""),

    (nbf.v4.new_code_cell, """import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos JSONL generados sintéticamente
datos = []
# Ruta corregida asumiendo ejecución desde la raíz o notebooks
ruta_dataset = "dataset_sintetico/mega_dataset_aeronautico.jsonl" if os.path.exists("dataset_sintetico/mega_dataset_aeronautico.jsonl") else "../dataset_sintetico/mega_dataset_aeronautico.jsonl"

with open(ruta_dataset, 'r', encoding='utf-8') as f:
    for linea in f:
        datos.append(json.loads(linea))

df = pd.DataFrame(datos)

# Countplot horizontal de las fuentes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='fuente', order=df['fuente'].value_counts().index, palette='Blues_r')
plt.title('Distribución de Pares Q&A generados por Documento Normativo')
plt.xlabel('Cantidad de Preguntas')
plt.ylabel('Cuerpo Legal (Fuente)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""),

    (nbf.v4.new_code_cell, """# Análisis de Longitud (Caracteres) de los fragmentos
df['longitud_contexto'] = df['contexto_original'].apply(len)
df['longitud_respuesta'] = df['respuesta'].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(df['longitud_contexto'], bins=30, ax=axes[0], color='indigo', kde=True)
axes[0].set_title('Distribución de Tamaño del Chunking (VRAM Constraints)')
axes[0].set_xlabel('Nº de Caracteres por Fragmento')

sns.histplot(df['longitud_respuesta'], bins=30, ax=axes[1], color='coral', kde=True)
axes[1].set_title('Longitud Semántica de las Respuestas Sintéticas')
axes[1].set_xlabel('Nº de Caracteres de la Respuesta')

plt.tight_layout()
plt.show()"""),

    (nbf.v4.new_code_cell, """from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

stop_words_es = set(stopwords.words('spanish'))
stop_words_es.update(['si', 'debe', 'ser', 'así', 'caso', 'puede', 'cualquier', 'parte'])

texto_preguntas = " ".join(df['pregunta'].tolist())

wordcloud = WordCloud(width=800, height=400, background_color='white', 
                      colormap='viridis', stopwords=stop_words_es, max_words=120).generate(texto_preguntas)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Complejidad Léxica Aeronáutica (Frecuencia en Consultas)', fontsize=15)
plt.show()"""),

    (nbf.v4.new_code_cell, """import numpy as np
import umap
from sklearn.decomposition import PCA
from ollama import Client
from tqdm.notebook import tqdm

cliente = Client(host='http://localhost:11434')
modelo_embeddings = "nomic-embed-text"

# Muestreo estratificado robusto a 2000 entidades garantizando presencia de columnas (Pandas >= 2.x)
muestras = []
for nombre, grupo in df.groupby('fuente'):
    muestras.append(grupo.sample(min(len(grupo), 300), random_state=42))
df_muestra = pd.concat(muestras).reset_index(drop=True)

if len(df_muestra) > 2000:
    df_muestra = df_muestra.sample(2000, random_state=42).reset_index(drop=True)

print("Invocando a Motor Embeddings de Ollama para 768 dimensiones...")
embeddings = []
for contexto in tqdm(df_muestra['contexto_original']):
    resp = cliente.embeddings(model=modelo_embeddings, prompt=contexto)
    embeddings.append(resp['embedding'])
matriz_embeddings = np.array(embeddings)

print("Ejecutando Reducción Dimensional PCA...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(matriz_embeddings)

df_muestra['PCA_1'] = embeddings_pca[:, 0]
df_muestra['PCA_2'] = embeddings_pca[:, 1]
varianza = pca.explained_variance_ratio_

print(f"Varianza explicada matemáticamente por Componentes PC1/PC2: {varianza[0]*100:.2f}%, {varianza[1]*100:.2f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_muestra, x='PCA_1', y='PCA_2', hue='fuente', palette='Set1', alpha=0.8, edgecolor='none')
plt.title('Topología de los Vectores Documentales (PCA)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""),

    (nbf.v4.new_markdown_cell, """---
## SECCIÓN B: Módulo Predictivo (Estructurado - Dataset CCHS)
---"""),

    (nbf.v4.new_code_cell, """# ==========================================
# Carga y Limpieza basada en Data Dictionary
# ==========================================
import numpy as np

# Carga del nuevo repositorio clínico canadiense (CCHS)
# Soportará ejecución desde root o subcarpeta
ruta_csv = "datos_crudos/health_dataset.csv" if os.path.exists("datos_crudos/health_dataset.csv") else "../datos_crudos/health_dataset.csv"
df_tab = pd.read_csv(ruta_csv)

# Filtrar columnas de interés (Biometría y Psicología)
cols_interes = ['Gen_health_state', 'Mental_health_state', 'Stress_level', 'Work_stress', 'BMI_18_above', 'Smoked', 'Weekly_alcohol', 'Work_hours']
# Se asegura de que existan en el dataframe
cols_interes = [c for c in cols_interes if c in df_tab.columns]
df_tab = df_tab[cols_interes].copy()

# Purga de "Basura Estadística" basada en el Data Dictionary
# Para categoricas de 1 dígito (1-6 es válido, 7, 8, 9 son Not stated/Don't know/Refusal)
for col in df_tab.columns:
    if col not in ['Work_hours', 'BMI_18_above']: # Las de 1 dígito
        df_tab[col] = df_tab[col].replace([7, 8, 9, 7.0, 8.0, 9.0], np.nan)
    else: # Valores continuos de 2 dígitos (96-99)
        df_tab[col] = df_tab[col].replace([96, 97, 98, 99, 999, 99.9, 99.96, 99.99, 996, 997, 998, 999], np.nan)

print("--- Data Quality Report Post-Limpieza (Nulos Generados por Códigos Faltantes) ---")
print(df_tab.isnull().sum())"""),

    (nbf.v4.new_code_cell, """# ==========================================
# Preprocesamiento: K-NN Imputer y Scaling
# ==========================================
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Muestra 50.000 registros para evitar saturación de RAM
df_tab_sample = df_tab.sample(min(50000, len(df_tab)), random_state=42).copy()

print(f"Ejecutando algoritmo K-NN (n_neighbors=5) para imputación multidimensional sobre {len(df_tab_sample)} instancias...")
imputador_knn = KNNImputer(n_neighbors=5)

# Se imputan todos los np.nan generados en el paso anterior
columnas_a_imputar = df_tab_sample.columns
df_tab_sample[columnas_a_imputar] = imputador_knn.fit_transform(df_tab_sample[columnas_a_imputar])

print("Escalando las variables continuas (BMI_18_above, Work_hours)...")
escalador = StandardScaler()
cols_escalar = [c for c in ['BMI_18_above', 'Work_hours'] if c in df_tab_sample.columns]

if cols_escalar:
    df_tab_sample[cols_escalar] = escalador.fit_transform(df_tab_sample[cols_escalar])

print("✅ Matriz de datos Limpia, Imputada, Escalada y lista para el modelo.")"""),

    (nbf.v4.new_code_cell, """# ==========================================
# EDA Clínico - Matriz de Correlación Aeronáutica
# ==========================================
df_corr = df_tab_sample.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='Spectral', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación: Factores Humanos y Riesgo en la Aviación', fontsize=15)
plt.tight_layout()
plt.show()

# Se extrae el diccionario de correlaciones de la variable Gen_health_state
corr_salud = df_corr['Gen_health_state'].to_dict()"""),

    (nbf.v4.new_code_cell, """# ==========================================
# Análisis Automatizado con LLM Local (Mistral)
# ==========================================
prompt_medico = f\"\"\"
Eres un Médico Aeronáutico y Científico de Datos. Analiza las siguientes correlaciones de Pearson respecto al Estado de Salud General ('Gen_health_state') de un piloto. Identifica las 3 relaciones más críticas (ej. impacto del estrés laboral o salud mental en la salud general). Responde de forma técnica, en español, numerando las conclusiones. Solo entrega las conclusiones.
Datos: {corr_salud}
\"\"\"

print("Transfiriendo contexto tabular a LLM (Cargando modelo local: mistral)...")
try:
    resp = cliente.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_medico}])
    print("\\n🚀 CONCLUSIONES GENERADAS POR INTELIGENCIA ARTIFICIAL (MISTRAL):")
    print("-" * 75)
    print(resp['message']['content'].strip())
    print("-" * 75)
except Exception as e:
    print(f"Error invocando a Ollama/Mistral: {e}")""")
]

for constructor, content in celdas:
    nb['cells'].append(constructor(content))

# Path changes per user request to root directory
output_notebook_path = 'EDA_Avanzado_DualEngine_CCHS.ipynb'
with open(output_notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook avanzado {output_notebook_path} creado exitosamente.")
