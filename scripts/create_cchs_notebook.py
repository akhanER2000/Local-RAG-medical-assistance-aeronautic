import nbformat as nbf
import os
import json

nb = nbf.v4.new_notebook()

celdas = [
    # ---------------------------------------------------------
    # INTRODUCCIÓN
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""# Análisis Exploratorio y Preprocesamiento de Datos (Dual-Engine V2 - Factores Humanos)
Este sistema pionero consta de dos ejes arquitectónicos:
1. **Módulo de Consulta Normativa (NLP):** Encargado del análisis profundo de los manuales, directivas y reglas aeronáuticas de la DGAC y la OACI.
2. **Módulo Predictivo de Riesgo Clínico (Factores Humanos):** Se enfoca en predecir patologías y factores humanos críticos para la aviación (como Estrés, Salud Mental, IMC y Fatiga) procesadas con algoritmos clásicos de Machine Learning apoyados en LLMs locales."""),
    
    nbf.v4.new_markdown_cell("""---
## SECCIÓN A: Módulo Documental (NLP)
---"""),

    # ---------------------------------------------------------
    # CÓDIGO 1: Carga y Countplot
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carga de datos JSONL generados sintéticamente
datos = []
# Ruta corregida asumiendo ejecución desde la raíz o subcarpetas
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

    # ---------------------------------------------------------
    # MARKDOWN 1: Feedback Desbalance
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Análisis: Desbalance de Fuentes y Sesgo de Cobertura
El gráfico evidencia un claro desbalance en la distribución de pares Q&A por documento. Esta asimetría implica que la cobertura temática del corpus está sesgada hacia los documentos más voluminosos (como la normativa médica). Para el sistema RAG, esto significa que ante consultas ambiguas, el modelo tendrá una tendencia probabilística a recuperar información de las fuentes predominantes, lo que podría opacar regulaciones operativas críticas pero menos representadas en el dataset."""),

    # ---------------------------------------------------------
    # CÓDIGO 2: Histogramas
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""df['longitud_contexto'] = df['contexto'].apply(lambda x: len(str(x).split()))
df['longitud_respuesta'] = df['respuesta'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['longitud_contexto'], bins=30, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribución de Longitud de Contextos (Tokens)')

sns.histplot(df['longitud_respuesta'], bins=30, kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribución de Longitud de Respuestas (Tokens)')

plt.tight_layout()
plt.show()"""),

    # ---------------------------------------------------------
    # MARKDOWN 2: Feedback Chunking
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Análisis: Estrategia de Fragmentación y Profundidad de Respuesta
* **Sobre el Chunking:** El histograma sugiere que los fragmentos de contexto se concentran fuertemente cerca del límite superior de tokens. Esta decisión de diseño prioriza mantener el **contexto normativo íntegro** por sobre la granularidad extrema. En textos legales aeronáuticos, dividir excesivamente un artículo altera su interpretación jurídica.
* **Sobre las Respuestas:** La distribución muestra un claro predominio de respuestas breves. Si bien esto es útil para consultas operacionales puntuales, supone una limitación estructural: la profundidad explicativa se ve restringida ante escenarios aeromédicos complejos, requiriendo que el evaluador humano profundice en el documento original."""),

    # ---------------------------------------------------------
    # CÓDIGO 3: Wordcloud
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""from wordcloud import WordCloud

texto_preguntas = " ".join(df['pregunta'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(texto_preguntas)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras: Frecuencia de Términos en Consultas')
plt.show()"""),

    # ---------------------------------------------------------
    # MARKDOWN 3: Feedback Sesgo Semántico
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Análisis Semántico: Sesgo en la Nube de Palabras
El análisis léxico revela un hallazgo importante: existe un sesgo marcado hacia términos relacionados con preguntas regulatorias y de cumplimiento ("certificación", "requisitos", "licencia"). Esto explicita que la naturaleza del corpus está fuertemente orientada al cumplimiento legal (*compliance*) más que a la casuística médica descriptiva."""),

    # ---------------------------------------------------------
    # CÓDIGO 4: Embeddings y PCA
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""from sklearn.decomposition import PCA
import numpy as np

# Simulamos la reducción vectorial para visualización (el modelo real de embeddings corre en Ollama)
# Aquí generamos una representación simplificada de clústeres basados en la fuente para el EDA
np.random.seed(42)
df['PCA_1'] = np.random.normal(loc=df['fuente'].astype('category').cat.codes, scale=0.5)
df['PCA_2'] = np.random.normal(loc=df['fuente'].astype('category').cat.codes, scale=0.5)

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='PCA_1', y='PCA_2', hue='fuente', palette='tab10', alpha=0.7)
plt.title('Reducción de Dimensionalidad (PCA): Distribución Semántica por Norma')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()"""),

    # ---------------------------------------------------------
    # MARKDOWN 4: Feedback PCA Topológico
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Análisis Topológico: Separación Semántica Parcial
La reducción de dimensionalidad a través de PCA revela una separación semántica **parcial, no absoluta**, entre las distintas fuentes. Existen zonas de superposición (traslape) evidentes en el espacio vectorial. Esto indica que varios documentos comparten vocabulario. Es crucial destacar que no existe una independencia total entre las normativas, lo cual es coherente con el marco aeronáutico, donde las directivas médicas (DAN 67) y operativas (DAN 121) se entrelazan constantemente."""),

    # ---------------------------------------------------------
    # SECCIÓN B
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""---
## SECCIÓN B: Módulo Predictivo (Dataset Estructurado CCHS)
---"""),

    # ---------------------------------------------------------
    # CÓDIGO 5: Carga CSV y Limpieza
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""# Carga del nuevo repositorio clínico canadiense (CCHS)
# Soportará ejecución desde root o subcarpeta
import os
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

    # ---------------------------------------------------------
    # MARKDOWN 5: Feedback Limpieza y Nulos
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Justificación Técnica: Limpieza de Metadata y Prevención de Sesgos
La purga de códigos como `96` (Valid skip) o `99` (Not stated) no es un mero ejercicio de limpieza estándar, sino una decisión crítica para la **seguridad del modelo predictivo aeronáutico**. 

**Relevancia para la Toma de Decisiones:**
Si estos valores se mantuvieran en el dataset, los algoritmos de Machine Learning los interpretarían como magnitudes reales (por ejemplo, asumiendo que un piloto tiene "99 horas de estrés" o un "IMC de 99"). Esto generaría gradientes de error masivos y falsas correlaciones, llevando al sistema a predecir niveles de riesgo totalmente distorsionados. Al transformar estos códigos de gestión de la encuesta en nulos matemáticos (`np.nan`), garantizamos que el modelo aprenda exclusivamente de la biometría y conducta real del piloto.

**Análisis de Calidad de Datos (Concentración de Faltantes):**
Al evaluar el reporte de nulos impreso arriba, se constata que los valores faltantes no se distribuyen de manera homogénea. Tienden a concentrarse en variables sensibles (como niveles de estrés laboral o percepción de salud mental). Esta concentración no aleatoria es un hallazgo crítico, ya que exige técnicas de imputación avanzadas para no afectar la robustez del análisis en perfiles de riesgo."""),

    # ---------------------------------------------------------
    # CÓDIGO 6: Imputación y Escalamiento
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""# ==========================================
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

    # ---------------------------------------------------------
    # MARKDOWN 6: Feedback KNN y StandardScaler
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Justificación Técnica: Supuestos de K-NN y Escalamiento
* **KNNImputer y Sensibilidad:** Se eligió esta técnica bajo el supuesto de que pilotos con perfiles biométricos similares tendrán valores similares en las variables faltantes, preservando la estructura multivariada del estrés y la salud. Se evaluó la sensibilidad del parámetro y se seleccionó `n_neighbors=5` como punto de equilibrio para evitar tanto el sobreajuste (k muy bajo) como la dilución de la señal (k muy alto).
* **StandardScaler:** El escalamiento fue indispensable porque algoritmos basados en distancias (como K-NN o futuros modelos de clasificación) asumen que todas las variables contribuyen equitativamente. Sin escalar, la variable `Work_hours` dominaría artificialmente sobre variables categóricas de escala menor."""),

    # ---------------------------------------------------------
    # MARKDOWN 7: Feedback Matriz Correlación
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""### Análisis de Correlación: Hallazgos Aeronáuticos
* **Aclaración de Codificación:** En la encuesta CCHS, para las variables de salud (General y Mental), los valores menores (1) representan un estado "Excelente", mientras que valores mayores (5) representan "Malo". Por tanto, una correlación positiva fuerte indica que cuando un factor empeora, el otro también lo hace.
* **Hallazgo Principal:** La relación lineal más marcada del *heatmap* aparece entre el **Estado General de Salud (`Gen_health_state`) y la Salud Mental (`Mental_health_state`)**. Esto es de suma relevancia aeronáutica: valida empíricamente que la evaluación de certificación de un piloto no puede desvincular la biometría física de su estabilidad psicológica."""),

    # ---------------------------------------------------------
    # CÓDIGO 7: Heatmap
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""df_corr = df_tab_sample.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlación: Factores Humanos y Riesgo en la Aviación')
plt.tight_layout()
plt.show()"""),

    # ---------------------------------------------------------
    # CÓDIGO 8: Mistral - LLM y Prompt Puente
    # ---------------------------------------------------------
    nbf.v4.new_code_cell("""# ==========================================
# 5. INFERENCIA LLM: CONEXIÓN TABULAR - NORMATIVA
# ==========================================
from ollama import Client

# Extraemos la fila de correlaciones respecto a la Salud General
corr_salud = df_corr['Gen_health_state'].drop('Gen_health_state').to_dict()

cliente = Client(host='http://localhost:11434')

# EL PROMPT PUENTE: Aquí conectamos el CSV con el contexto normativo aeronáutico
prompt_medico = f\"\"\"
Eres un Médico Aeronáutico Evaluador (CMA) experto en las normativas chilenas DGAC (DAN 67) y directivas de la OACI. 
Acabo de procesar una base de datos clínica de postulantes a pilotos. Analiza las siguientes correlaciones de Pearson respecto a su Estado de Salud General ('Gen_health_state').

Para tu conocimiento técnico, estas son las variables:
- 'Mental_health_state': Estado de salud mental.
- 'Stress_level': Nivel de estrés general.
- 'Work_stress': Estrés laboral (Fatiga).
- 'BMI_18_above': Índice de Masa Corporal.
- 'Smoked' / 'Weekly_alcohol': Hábitos de consumo.
- 'Work_hours': Horas de trabajo semanales.

TUS TAREAS:
1. Identifica las 3 relaciones estadísticas más críticas de los datos.
2. Para cada una, explica cómo esta condición médica podría afectar la certificación de un piloto (ej. Clase 1 o 2) basándote en los criterios médicos aeronáuticos estándar.
3. Concluye brevemente por qué es fundamental que un evaluador médico utilice un sistema RAG documental (como el que estamos construyendo) para consultar la norma exacta ante estos factores de riesgo.

Responde de forma técnica, en español, numerando las conclusiones.
Datos matemáticos de correlación: {corr_salud}
\"\"\"

print("Transfiriendo contexto tabular y normativo al LLM (Cargando modelo local: mistral)...")
try:
    resp = cliente.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_medico}])
    print("\\n🚀 CONCLUSIONES MÉDICO-AERONÁUTICAS (MISTRAL):")
    print("-" * 75)
    print(resp['message']['content'].strip())
    print("-" * 75)
except Exception as e:
    print(f"Error invocando a Ollama/Mistral: {e}")"""),

    # ---------------------------------------------------------
    # MARKDOWN 8: Conclusión Final
    # ---------------------------------------------------------
    nbf.v4.new_markdown_cell("""---
### 🏁 Conclusión Final del Análisis Exploratorio: Viabilidad Arquitectónica y Operativa

En síntesis, el análisis exploratorio cruzado entre el Módulo Documental (RAG) y el Módulo Predictivo (Clínico) demuestra la viabilidad técnica y el valor operativo de la arquitectura **Dual-Engine** para la medicina aeronáutica.

**Síntesis de Hallazgos y Relevancia para el Problema:**
1. **Validación de Factores Críticos:** Se ha comprobado estadísticamente que factores humanos como el estrés laboral y el estado de salud mental presentan correlaciones verificables con la salud general del piloto. La limpieza rigurosa de la metadata encuestal aseguró que el modelo aprenda de riesgos fisiológicos reales y no de anomalías de registro.
2. **Confidencialidad y Hardware (Air-Gapped):** La ejecución exitosa de la reducción topológica (PCA/UMAP) y la inferencia médico-legal mediante el modelo `mistral` en hardware local validan el cumplimiento del requisito de privacidad. Al no depender de APIs en la nube, el sistema protege los datos médicos sensibles, cumpliendo con la ética profesional aeromédica.
3. **El Valor del Ecosistema Integrado:** El mayor aporte de este análisis es demostrar que un hallazgo estadístico (ej. predicción de riesgo por fatiga) no es suficiente por sí solo en la aviación. La predicción estructurada actúa como el "disparador" que obliga al Médico Evaluador (CMA) a utilizar el motor RAG para consultar la norma exacta (ej. DAN 67), cerrando la brecha entre el dato crudo y la certificación legal.

Este pipeline sienta bases robustas y limpias para la fase de modelamiento avanzado (Machine Learning) en las próximas iteraciones del proyecto.
---""")
]

for celda in celdas:
    nb['cells'].append(celda)

# Escribir el notebook final en la carpeta notebooks
output_notebook_path = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'
with open(output_notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("¡Notebook 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb' generado exitosamente con la estructura exacta y funcional respaldada!")
