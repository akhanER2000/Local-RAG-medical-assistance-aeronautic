# ✈️ Sistema Integral de Evaluación Médica Aeronáutica (Dual-Engine)

> **📌 Nota para la Evaluación (Entrega 1 - Versión Final):**
> Este repositorio ha sido reestructurado para garantizar la trazabilidad del proyecto. El archivo principal a evaluar es el notebook **`EDA_Avanzado_DualEngine_CCHS.ipynb`**, el cual contiene el análisis exploratorio definitivo utilizando el dataset de Factores Humanos (CCHS).
> *Nota: Los notebooks y datasets de las iteraciones tempranas (basadas en diabetes) no han sido eliminados, sino que han sido movidos a las carpetas `notebooks_archivados/` y `datasets_archivados/` para mantener el historial de desarrollo limpio y accesible.*

Este proyecto nace de la necesidad de aplicar la metodología científica de análisis de datos **CRISP-DM** para evaluar la aptitud médica de pilotos (con un foco analítico especial en patologías y factores humanos asociados al estrés crónico, fatiga aeromédica y salud mental). Para lograr una cobertura completa de los requerimientos clínicos y reglamentarios, el sistema opera bajo una innovadora arquitectura **Dual-Engine** que combina el rigor de las regulaciones de la DGAC/OACI (mediante técnicas de Procesamiento de Lenguaje Natural) con el modelamiento matemático predictivo sobre datos clínicos estructurados reales.

---

## 📄 Informe de Avance y Formulación del Proyecto

Toda la justificación teórica, los objetivos del negocio, el entendimiento clínico y las definiciones de alcance que enmarcan este repositorio se encuentran formalizados académicamente en la documentación oficial. Por favor consulte el documento directamente en el siguiente enlace:

🔗 **[Documentación de Investigación y Respaldo Teórico del Proyecto](https://docs.google.com/document/d/1MERbrd4tcaVxPxwzoyYjgJ-4ecfWIIenZio1oF4cc24/edit?usp=sharing)**

---

## 🚀 Ingeniería de Modelamiento Avanzado (Dual-Engine V2)

Tras consolidar el EDA, el sistema ha evolucionado hacia una arquitectura de Inteligencia Artificial Híbrida. A continuación se detallan los procesos de la fase de modelado bajo el estándar **CRISP-DM**:

### 1. Data Quality Report (DQR) y Limpieza Profunda (Punto 6)
No se realizó una limpieza genérica; se aplicó un diagnóstico de calidad industrial:
- **Gestión de Outliers:** Implementación del método de Rango Intercuartílico (IQR) para filtrar ruidos biométricos en variables continuas (IMC y Horas de trabajo), asegurando que el modelo aprenda de datos realistas.
- **Imputación Multivariada:** Uso de `KNNImputer` para nulos, asumiendo que el perfil de salud de un piloto es interdependiente (si falta un dato de estrés, se deduce de sus indicadores de fatiga y salud mental similares).
- **Balanceo de Clases con SMOTE:** El dataset original presentaba un desbalance crítico (90% de casos "Apto"). Se aplicó **SMOTE** (*Synthetic Minority Over-sampling Technique*) para generar minorías sintéticas de la clase "Riesgo", reduciendo drásticamente el sesgo hacia la clase mayoritaria y mejorando la sensibilidad del sistema.

### 2. Segmentación No Supervisada: Descubrimiento de Patrones (Punto 7)
Para la medicina aeronáutica preventiva, implementamos **K-Means Clustering (k=3)**:
- **Objetivo:** Identificar grupos de pilotos que, aunque no tienen un diagnóstico formal, presentan combinaciones peligrosas de fatiga silenciosa y estrés.
- **Visualización:** Se utilizó **PCA** (*Principal Component Analysis*) para reducir la dimensionalidad y visualizar los clústeres en un plano 2D, permitiendo al CMA (Examinador Médico) ver la topología del riesgo de su dotación.

### 3. Modelamiento Supervisado y Evaluación (Puntos 9 y 10)
Se evaluó un ecosistema de tres algoritmos para garantizar la mejor elección técnica:
- **Random Forest (Ganador):** Seleccionado por su capacidad para manejar relaciones no lineales y su robustez frente al sobreajuste.
- **SVM (Kernel RBF):** Utilizado para encontrar fronteras de decisión óptimas en espacios de alta dimensionalidad.
- **Regresión Logística:** Implementada como baseline comparativo de interpretabilidad.

> **Métricas Finales Alcanzadas:**
> - **Recall (Clase Riesgo):** Priorizado para minimizar el Riesgo Tipo II (falsos negativos), donde un piloto no apto es clasificado como apto.
> - **F1-Score:** Logró un equilibrio superior al **0.85** tras la aplicación de SMOTE.

### 4. Validación de Generalización (Punto 11)
Para garantizar que el sistema funcione con nuevos pilotos, se aplicó **K-Fold Cross-Validation (k=5)**. Este proceso validó que la precisión del modelo es estable (**±0.02**) y no depende de una división aleatoria afortunada de los datos, blindando el proyecto contra el *Overfitting*.

### 5. El "Prompt Puente": Integración Dual-Engine (Punto 12)
La arquitectura se cierra con una innovación técnica: la salida probabilística del Random Forest se inyecta dinámicamente como contexto en el motor RAG.

> **Ejemplo:** Si el modelo detecta "Riesgo de Fatiga", el sistema genera automáticamente un prompt para Mistral-7B solicitando: *"Basado en la detección de Riesgo Nivel 1, recupere de la DAN 67 los protocolos de suspensión de licencia médica"*.

## 📋 Guía Completa de Cumplimiento (Rúbrica Sumativa 1)

| Ítem Rúbrica | Estado | Ubicación de la Evidencia |
| :--- | :---: | :--- |
| **Punto 6:** DQR y Preprocesamiento | ✅ | Celdas 12-14 del Notebook (Análisis IQR y Balanceo). |
| **Punto 7:** Selección de Modelos | ✅ | Markdown en Sección B (Justificación SVM vs RF). |
| **Punto 8:** División Train/Test | ✅ | Celda de Split con `stratify=y` para consistencia médica. |
| **Punto 9:** Entrenamiento | ✅ | Flujo automatizado en Sección B del Notebook. |
| **Punto 10:** Evaluación y Métricas | ✅ | Matrices de Confusión y Classification Reports detallados. |
| **Punto 11:** Optimización y CV | ✅ | Celda de Cross-Validation al final del modelamiento. |
| **Punto 12:** Arquitectura Real | ✅ | Implementación funcional del "Prompt Puente" hacia Mistral. |

---

## 🧬 Arquitectura Evolucionada: Sistema Dual-Engine V2

El sistema ha trascendido de un análisis exploratorio a una plataforma de inteligencia artificial híbrida:
- **Motor Predictivo (Supervisado):** Basado en un ensamble de Random Forest, entrenado para clasificar la aptitud de vuelo en un entorno dicotómico (APTO vs. NO APTO). Se implementó SMOTE (Synthetic Minority Over-sampling Technique) para corregir el desbalance crítico del dataset original, garantizando un Recall superior en la detección de pilotos con riesgo de fatiga.
- **Motor de Segmentación (No Supervisado):** Implementación de K-Means Clustering para descubrir perfiles de riesgo latentes que no están etiquetados en la data oficial, permitiendo una medicina aeronáutica preventiva.
- **Motor RAG (NLP):** Inferencia local (Air-Gapped) mediante Mistral-7B que actúa como el respaldo legal, vinculando las alertas biométricas del modelo predictivo con los artículos específicos de la DAN 67 y OACI.

## 📊 Resultados de Modelamiento y Métricas

| Módulo | Técnica | Métrica Clave | Resultado |
| :--- | :--- | :--- | :--- |
| **Predictivo** | Random Forest + SMOTE | F1-Score (Clase Riesgo) | ~0.85+ |
| **Generalización** | K-Fold Cross Validation | Stability Accuracy | ±0.02 |
| **Segmentación** | K-Means (k=3) | PCA Variance Explained | 6.88% |
| **Recuperación** | Local RAG | Top-3 Accuracy | >85% |

## 📋 Guía Quick-Check para Evaluación (Rúbrica Sumativa 1)

- [x] **Punto 6: Preprocesamiento y Calidad:** Ver DQR de outliers y limpieza KNN en Sección B del Notebook.
- [x] **Punto 7, 9, 10: Modelamiento:** Comparativa SVM vs RF en Sección B.
- [x] **Punto 8: División Train/Test:** Implementado con estratificación para conservar proporciones médicas.
- [x] **Punto 11: Optimización:** Validación cruzada aplicada para mitigar el Overfitting.
- [x] **Punto 12: Arquitectura:** Implementación funcional del "Prompt Puente" entre ML y RAG.

---

## 🏗️ Arquitectura del Sistema (Dual-Engine)

El ecosistema computacional se nutre del esfuerzo cooperativo de dos motores integrados para garantizar diagnósticos y respuestas sin alucinaciones.

### 1. Módulo de Consulta Normativa (Motor RAG - NLP)
Diseñado para la asimilación legal y búsqueda semántica de reglamentos:
- **Procesamiento Masivo:** Fragmentación matemática de **14.501** pares de preguntas/respuestas generados sintéticamente a partir de las resoluciones DGAC (DAN 19, DAN 61, DAN 67, DAN 121, DAN 135) y el Manual Médico OACI 8984.
- **Topología e Ingestión Vectorial:** Emplea los embeddings cuánticos (`nomic-embed-text`) proyectados en mapas 2D mediante técnicas de reducción dimensional clásicas y avanzadas (**PCA** y **UMAP**) para validar que las "islas de conocimiento" médico estén aisladas matemáticamente de las normas operacionales de vuelo.

### 2. Módulo Predictivo de Riesgo Clínico (Motor Estructurado)
Cimentado sobre las bases del Machine Learning clásico para prevenir fallos humanos:
- **Corpus Analítico:** Exploración de la macro-base de datos médica del Canadian Community Health Survey (CCHS) enfocada en factores críticos para el vuelo (Dataset: [Healthcare Survey en Kaggle](https://www.kaggle.com/datasets/aradhanahirapara/healthcare-survey/data)).
- **Robustez Algorítmica:** Inyección simulada de defectos de sensores hospitalarios en biomarcadores continuos, neutralizada a través de imputaciones multivariadas con **K-Nearest Neighbors (K-NN)** y una estabilización total de pesos escalares aplicando un **StandardScaler**.

---

## 🔬 Fases CRISP-DM Implementadas (Entregables Actuales)

Actualmente, el cuaderno analítico maestro cubre con éxito las etapas fundamentales de pre-entrenamiento de los datos:

- **Data Understanding (Comprensión de los Datos):** Se extraen histogramas distribucionales precisos de textos y Nubes de Palabras filtradas. Adicionalmente, cuenta con el desarrollo pionero de una Matriz de Correlación de Pearson cuyos resultados estadísticos de los Factores de Riesgo son interpretados en tiempo real por Inteligencia Artificial generativa local.
- **Data Quality Report (Reporte de Integridad):** Reporte automatizado en código que incluye la detección forense de valores nulos (NaN), simulación de rotura de flujos de datos en mediciones como el *Body Mass Index* (BMI) para medir la resiliencia algorítmica.
- **Data Preparation (Preparación de Datos):** Transformación técnica orientada a modelos de Machine Learning (como Máquinas de Soporte Vectorial - SVM). Implica el llenado deductivo (imputación algorítmica multidimensional) y una escalabilidad (normalización) que ajusta la varianza a 1, eliminando ruidos métricos del *dataset tabular*.

---

## ⚙️ Requisitos Técnicos y Reproducibilidad

El sistema debe ejecutarse en el entorno para el cual fue nativamente conceptualizado.
- **Lenguaje Base:** Python 3.10+
- **Entorno de Operación:** Entornos Virtuales (`env`) sobre *Jupyter Notebook*.
- **Dependencias Clave:** `pandas`, `scikit-learn`, `seaborn`, `umap-learn`, `wordcloud`.
- **Hardware e IA Local Extricta:** Para dotar al sistema Dual-Engine y mantener privacidad PIV total sobre cuadros médicos, **el pipeline exige la instalación del servidor local `Ollama` ejecutándose en segundo plano**. Emplea concretamente los pesos de **`mistral`** (para inferencia lógica experta) y **`nomic-embed-text`** (para matematización vectorial), requiriendo de aceleración intensiva por GPU de la familia Turing/Ada/Lovelace/Blackwell.

---

## 🚥 Instrucciones de Ejecución Rápida

Sigue estos rigurosos pasos para auditar el funcionamiento matemático del proyecto:

1. **Clona el ecosistema a tu estación de trabajo:**
   ```bash
   git clone https://github.com/akhanER2000/Local-RAG-medical-assistance-aeronautic.git
   cd Local-RAG-medical-assistance-aeronautic
   ```

2. **Carga y arranca el Motor Ollama:**
   Inicia la aplicación de Ollama en tu ordenador. Asegúrate de tener los modelos base sincronizados ejecutando en consola:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Inicia tu IDE y el Cuaderno Predictivo:**
   Abre el archivo maestro `notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb` mediante VS Code u otro navegador para distribuciones Jupyter.

4. **Desencadena el CRISP-DM Pipeline:**
   Asegúrate de escoger el kernel de Python correspondiente a tu entorno pre-configurado y presiona **Run All**. Observarás en vivo la renderización dimensional de PCA, las matrices de calor clínicas y a *Mistral* emitiendo sus diagnósticos médicos concluyentes.
