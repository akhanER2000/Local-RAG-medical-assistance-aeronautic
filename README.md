# ✈️ Sistema Integral de Evaluación Médica Aeronáutica (Dual-Engine)

Este proyecto nace de la necesidad de aplicar la metodología científica de análisis de datos **CRISP-DM** para evaluar la aptitud médica de pilotos (con un foco analítico especial en patologías como la Diabetes Tipo 1). Para lograr una cobertura completa de los requerimientos clínicos y reglamentarios, el sistema opera bajo una innovadora arquitectura **Dual-Engine** que combina el rigor de las regulaciones de la DGAC/OACI (mediante técnicas de Procesamiento de Lenguaje Natural) con el modelamiento matemático predictivo sobre datos clínicos estructurados reales.

---

## 📄 Informe de Avance y Formulación del Proyecto

Toda la justificación teórica, los objetivos del negocio, el entendimiento clínico y las definiciones de alcance que enmarcan este repositorio se encuentran formalizados académicamente en la documentación oficial. Por favor consulte el documento directamente en el siguiente enlace:

🔗 **[Documentación de Investigación y Respaldo Teórico del Proyecto](https://docs.google.com/document/d/1MERbrd4tcaVxPxwzoyYjgJ-4ecfWIIenZio1oF4cc24/edit?usp=sharing)**

---

## 🏗️ Arquitectura del Sistema (Dual-Engine)

El ecosistema computacional se nutre del esfuerzo cooperativo de dos motores integrados para garantizar diagnósticos y respuestas sin alucinaciones.

### 1. Módulo de Consulta Normativa (Motor RAG - NLP)
Diseñado para la asimilación legal y búsqueda semántica de reglamentos:
- **Procesamiento Masivo:** Fragmentación matemática de **14.501** pares de preguntas/respuestas generados sintéticamente a partir de las resoluciones DGAC (DAN 19, DAN 61, DAN 67, DAN 121, DAN 135) y el Manual Médico OACI 8984.
- **Topología e Ingestión Vectorial:** Emplea los embeddings cuánticos (`nomic-embed-text`) proyectados en mapas 2D mediante técnicas de reducción dimensional clásicas y avanzadas (**PCA** y **UMAP**) para validar que las "islas de conocimiento" médico estén aisladas matemáticamente de las normas operacionales de vuelo.

### 2. Módulo Predictivo de Riesgo Clínico (Motor Estructurado)
Cimentado sobre las bases del Machine Learning clásico para prevenir fallos humanos:
- **Corpus Analítico:** Exploración de la macra-base de datos del Sistema de Vigilancia de Factores de Riesgo del CDC de EE. UU. (*BRFSS*) enfocado en cuadros diabéticos.
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
   Abre el archivo maestro `notebooks/EDA_Integral_DualEngine.ipynb` mediante VS Code u otro navegador para distribuciones Jupyter.

4. **Desencadena el CRISP-DM Pipeline:**
   Asegúrate de escoger el kernel de Python correspondiente a tu entorno pre-configurado y presiona **Run All**. Observarás en vivo la renderización dimensional de PCA, las matrices de calor clínicas y a *Mistral* emitiendo sus diagnósticos médicos concluyentes.
