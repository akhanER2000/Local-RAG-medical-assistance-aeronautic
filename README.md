# 🚁 Sistema RAG Local - Asistencia Médica Aeronáutica (DGAC Chile)

Bienvenido a la infraestructura base del sistema **RAG (Retrieval-Augmented Generation) Local**. Este es un proyecto de Ciencia de Datos y Procesamiento de Lenguaje Natural enfocado en crear un asistente virtual experto en la normativa médica y operacional de la Dirección General de Aeronáutica Civil (DGAC) y la OACI.

Todo el ecosistema de inteligencia artificial está configurado para operar de manera **100% local**, maximizando la privacidad de los datos mediante el uso avanzado de una tarjeta gráfica de alta gama (Serie RTX50) y prescindiendo de dependencias de nubes y cobros externos.

---

## 📑 Arquitectura Actual del Proyecto

El proyecto está diseñado metodológicamente inspirado en **CRISP-DM**, cubriendo hasta el momento las siguientes fases:

### 1. Generación de Dataset Sintético (`scripts/generar_dataset.py`)
Utilizando los manuales base (tales como las normativas `DAN 121`, `DAN 135`, `DAN 67`, `MAPE-MEDAV`, entre otros), este script:
- Se encarga de fragmentar limpiamente cada PDF protegiendo la cohesión de los párrafos usando `RecursiveCharacterTextSplitter` de LangChain.
- Invoca al modelo en local de Meta, **Llama 3** (a través de Ollama), para leer, analizar exhaustivamente cada fragmento y extraer sistemáticamente **hasta 3 pares de Pregunta/Respuesta** técnica.
- Como resultado: Se creó de manera autónoma un mega-dataset de más de **14.500 Q&A** empacado en `mega_dataset_aeronautico.jsonl`. Un activo invaluable para afinar cualquier sistema de IA en salud aeronáutica.

### 2. Análisis Exploratorio de Datos / EDA (`notebooks/EDA_Aeronautico.ipynb`)
Para garantizar que la metadata y contenido sintetizado se adhieren a la calidad exigida por el proyecto, el Notebook:
- Purga valores vacíos y justifica estadísticamente la distribución de caracteres (Context vs Respuesta).
- Muestra una Nube de Palabras generada tras aplicar filtros NLP (Stopwords).
- Ejecuta técnicas científicas para validar una representación de las normativas de "Vuelo" (DAN) vs "Médicas": usando la descomposición de varianza de **PCA** (Modelo de Aprendizaje Automático Clásico) y luego el mapeo topológico avanzado en gráficos mediante el algoritmo **UMAP**. Ambas técnicas alimentadas matemáticamente por los *embeddings* locales de `nomic-embed-text`.

---

## 🛠 Instalación y Configuración

Toda la infraestructura está desplegada dentro de un entorno virtual para asegurar versiones compatibles de librerías.

### Prerrequisitos
1. Instalador de Python 3.10+
2. **Ollama**: Pre-instalado e iniciado (Para levantar Llama 3 y nomic-embed-text).

### Comandos de Activación
```bash
# Entrar al entorno protegido
.\env\Scripts\activate

# Modelos locales de Inferencia requeridos
ollama pull llama3
ollama pull nomic-embed-text
```

---

## 🚀 Uso

### Re-Generar el Mega Dataset
Si añades más documentos PDF en la carpeta `datos_crudos/`, puedes volver a generar el dataset haciendo uso directo de Python:
```bash
python scripts/generar_dataset.py
```
*(Nota: Toma alrededor de 25-30 minutos procesar todo un compendio de reglamentos base con una RTX 5070 Ti)*

### Iniciar el Análisis Exploratorio
Carga las celdas directamente dentro de tu Notebook para ver las demostraciones de PCA y agrupamiento de Inteligencia Artificial No Supervisada:
1. Abre tu IDE / VS Code y selecciona el Entorno Virtual: `env (RAG Aeronautico)`.
2. O bien, inicia Jupyter desde terminal: `jupyter notebook notebooks/EDA_Aeronautico.ipynb`

---

## 🔮 Futuras Implementaciones
En las siguientes fases del proyecto, se construirá la Base de Datos Vectorial (ChromaDB o similar) donde ingeriremos todos los fragmentos y estableceremos el *pipeline* del "Buscador Inteligente", en el que el usuario podrá consultar regulaciones en vivo conversando con Llama 3 usando esta base como respaldo legal estricto.
