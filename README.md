# 👨‍✈️ Dual-Engine: Local RAG & Predictive Clinical Assessment for Aeronautic Safety

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/AI-Ollama-orange.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Este repositorio contiene el sistema integral de evaluación aeromédica basado en una arquitectura **Dual-Engine**, diseñado para garantizar la seguridad operacional mediante la detección proactiva de riesgos fisiológicos y el respaldo normativo instantáneo.

---

## 📋 Guía de Evaluación Interactiva (Rúbrica Sumativa 1)
Para facilitar la revisión docente, se detalla la ubicación de los requisitos evaluados:

- [x] **Comprensión de los Datos (Punto 5)** -> Sección A (NLP) y Sección B (EDA Tabular).
- [x] **Preprocesamiento Avanzado y DQR (Punto 6)** -> Ver Fase 1 de la Sección B (Tratamiento de Outliers e IQR).
- [x] **Selección de Algoritmos (Punto 7)** -> Comparativa entre SVM, Logistic Regression y Random Forest.
- [x] **Implementación de Pipelines (Punto 8)** -> Uso de `train_test_split` estratificado y balanceo SMOTE.
- [x] **Evaluación Cualitativa (Punto 9)** -> Análisis de Matrices de Confusión y Reportes de Clasificación.
- [x] **Métricas de Rendimiento (Punto 10)** -> F1-Score y Recall enfocados en Riesgo Aeronáutico.
- [x] **Optimización y Validación (Punto 11)** -> Implementación de 5-Fold Cross-Validation para mitigar Overfitting.

---

## 🚀 Resumen Ejecutivo: El Concepto "Dual-Engine"

El sistema opera bajo dos motores inteligentes que trabajan en simbiosis para asistir al Médico Examinador Aeronáutico (CMA):

1.  **Motor 1: Machine Learning Predictivo (Fase Bio-Conductual)**
    *   Analiza datos clínicos estructurados (estrés, fatiga, IMC, salud mental).
    *   Clasifica de forma proactiva si un postulante es **APTO** o **NO APTO**, emitiendo una alerta temprana antes de que ocurra un incidente.
2.  **Motor 2: RAG Local (Fase Normativa/Legal)**
    *   Utiliza Recuperación Aumentada por Generación (RAG) para consultar manuales de la **DGAC (DAN 67)** y **OACI**.
    *   Provee el fundamento legal exacto que justifica la decisión médica sugerida por el primer motor.

---

## 🧠 Justificación de Decisiones Técnicas (The "Why")

### ⚖️ Binarización de la Aptitud
Pasar de una escala de 5 niveles a una clasificación **APTO / NO APTO** simplifica la toma de decisiones crítica. En aviación, la ambigüedad es riesgo; el sistema traduce la probabilidad estadística en una acción operativa clara para el médico evaluador.

### 🧪 Balanceo con SMOTE
El conjunto de datos original presentaba un desbalance masivo (pocos pilotos enfermos vs. muchos sanos). Implementar **SMOTE** (*Synthetic Minority Over-sampling Technique*) nos permite "enseñar" al modelo a reconocer los perfiles de riesgo sin caer en la paradoja del Accuracy, priorizando el **Recall** (detección de casos críticos).

### 🔍 Segmentación con K-Means
Utilizamos aprendizaje no supervisado para descubrir **Perfiles Ocultos de Fatiga**. Esto permite identificar pilotos que, aunque pasen las pruebas estándar, muestran patrones biométricos compartidos con casos de fatiga severa, permitiendo una intervención preventiva.

### 🔐 Air-Gapped Privacy
Utilizar **Mistral-7B** ejecutándose localmente con Ollama asegura que los datos médicos sensibles de los pilotos nunca salgan del entorno local, cumpliendo estrictamente con la ética médica y los estándares de seguridad aeronáutica.

---

## 📊 Métricas de Desempeño Obtenidas

| Métrica | Resultado | Interpretación |
| :--- | :--- | :--- |
| **F1-Score (Promedio)** | ~85% | Equilibrio sólido entre precisión y sensibilidad clínica. |
| **Recall (No Apto)** | >90% | Alta capacidad de detectar pilotos en riesgo (Prioridad 1). |
| **Cross-Validation Accuracy** | 86.4% (±0.02) | Estabilidad probada; el modelo no presenta Overfitting. |
| **RAG Precision** | Hit@3 | El motor recupera la norma correcta en los primeros 3 resultados. |

---

## 🛠️ Requisitos de Instalación

1.  **Ollama**: Instalar desde [ollama.com](https://ollama.com/) y descargar los modelos necesarios:
    ```bash
    ollama pull mistral
    ollama pull nomic-embed-text
    ```
2.  **Dependencias de Python**:
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn sentence-transformers ollama nbformat
    ```

## 📂 Estructura del Proyecto

*   `notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb`: El núcleo del análisis y modelamiento.
*   `dataset_sintetico/`: Datos de normativa aeronáutica vectorizados.
*   `datos_crudos/`: Dataset CCHS de factores humanos.
*   `docs/`: Documentación técnica y rúbricas.
*   `scripts/`: Utilidades para parcheo y automatización del notebook.

---

**Desarrollado para la Evaluación de Seguridad Operacional en Aviación Civil.** 🚁🛡️
