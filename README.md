# Sistema de Evaluación Médica Aeronáutica — Arquitectura Dual-Engine

Motor de apoyo a la decisión que evalúa señales clínicas contra normativa
aeronáutica y médica, de forma **local, reproducible y trazable**, siguiendo la
metodología **CRISP-DM**. Es el componente técnico abierto sobre el que se
construye el producto privado *AeroFit*: el producto es privado; la ingeniería
que lo sostiene, no.

Combina dos motores: un modelo predictivo de Machine Learning clásico sobre
datos clínicos estructurados, y un motor RAG local (air-gapped) que responde
citando la normativa de la DGAC/OACI. Un **«prompt puente»** conecta ambos.

## Qué es

1. **Motor predictivo (supervisado).** Clasifica la aptitud (apto / no apto) a
   partir de variables clínicas, con foco en fatiga, estrés y salud mental y la
   clase «Riesgo» como objetivo prioritario.
2. **Motor de segmentación (no supervisado).** K-Means (k=3) para explorar
   perfiles de riesgo latentes no etiquetados en la data.
3. **Motor RAG (local, air-gapped).** Mistral-7B servido por Ollama, con
   embeddings vectoriales `nomic-embed-text`, sobre un corpus normativo
   público. Responde citando la fuente; no infiere fuera de ella.

## Arquitectura

```
   datos clínicos            ┌──────────────────────────┐
   estructurados  ─────────▶ │  Motor predictivo (RF)   │
   (CCHS, proxy)             │  scikit-learn + SMOTE    │
                             └───────────┬──────────────┘
                                         │ salida de riesgo
                              「prompt puente」 la inyecta como contexto
                                         ▼
   normativa pública   ┌──────────────────────────────────────┐
   DAN 19/61/67/121/135 │  Motor RAG local (air-gapped)        │ ─▶ respuesta
   OACI Doc 8984  ────▶ │  Ollama · Mistral-7B                  │    con la
                        │  embeddings vectoriales nomic-embed   │    norma
                        └──────────────────────────────────────┘    citada
```

- **Embeddings:** `nomic-embed-text`, un modelo de embeddings **vectoriales**
  ejecutado localmente vía Ollama.
- **Corpus normativo:** ~14.501 pares de pregunta/respuesta generados de forma
  **sintética** a partir de las resoluciones DGAC (DAN 19, 61, 67, 121, 135) y
  el Manual Médico Aeronáutico OACI (Doc 8984).
- **Air-gapped:** toda la inferencia corre en local; ningún dato sale del
  entorno.

## Metodología (CRISP-DM)

**Preparación de datos**
- **Outliers:** filtrado por Rango Intercuartílico (IQR) en variables continuas
  (IMC, horas de trabajo).
- **Imputación:** `KNNImputer` para valores faltantes, aprovechando la
  correlación entre indicadores de salud.
- **Escalado:** `StandardScaler` (media 0, varianza 1).
- **Desbalance:** el dataset presenta ~90 % de casos «Apto». Se aplicó **SMOTE**
  sobre el conjunto de entrenamiento para mejorar la sensibilidad a la clase
  «Riesgo».
- **División:** train/test estratificado (`stratify=y`) para conservar las
  proporciones de clase.

**Modelamiento supervisado.** Comparativa de tres algoritmos:
- **Random Forest** — modelo elegido (relaciones no lineales, robustez).
- **SVM (kernel RBF)** — frontera de decisión en alta dimensión.
- **Regresión Logística** — línea base interpretable.

**Segmentación no supervisada.** K-Means (k=3) para descubrir perfiles de
riesgo. Se usó PCA solo para *visualizar* los clústeres en 2D (ver
Limitaciones).

**Validación.** K-Fold estratificado (k=5): F1 promedio 0,75 con desviación
±1,3 % entre folds (folds 0,72–0,76).

**Integración (prompt puente).** La salida del Random Forest se inyecta como
contexto del RAG. Ejemplo: ante «Riesgo de fatiga», el sistema construye un
prompt para Mistral-7B pidiendo recuperar de la DAN 67 los protocolos de
suspensión de licencia médica.

## Resultados

El modelo prioriza el **Recall** sobre la clase «Riesgo»: en un dominio médico,
un falso negativo (declarar «apto» a quien no lo está) es el error costoso. Eso
se logra a costa de la precisión — el modelo marca de más. Las cifras son de la
evaluación sobre el conjunto de prueba (10.000 muestras) y de la validación
cruzada; corresponden al pipeline estable **previo a la optimización de
accuracy** (rama de trabajo).

| Módulo | Técnica | Métrica | Resultado |
| :--- | :--- | :--- | :--- |
| Predictivo | Random Forest + SMOTE | Recall (clase Riesgo) | 0,65 |
| Predictivo | Random Forest + SMOTE | F1 (clase Riesgo) | 0,48 |
| Predictivo | Random Forest + SMOTE | Precisión (clase Riesgo) | 0,38 |
| Generalización | K-Fold (k=5) | F1 promedio | 0,75 (±1,3 %) |
| Recuperación (RAG)¹ | evaluación simulada | Top-3 / MRR | 0,80 / 0,57 |

<sup>1</sup> La métrica del RAG proviene de una evaluación **ilustrativa sobre
5 consultas** con recuperación simulada (`RAG_Evaluation.ipynb`), no de una
evaluación del retriever en producción. Se reporta como referencia, no como
resultado del sistema real.

## Requisitos e instalación

- **Python** 3.10+
- **Dependencias:** `pandas`, `scikit-learn`, `seaborn`, `umap-learn`,
  `wordcloud`
- **[Ollama](https://ollama.com)** corriendo localmente, con los modelos:
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
- **GPU** recomendada para acelerar la inferencia local.

```bash
git clone https://github.com/akhanER2000/Local-RAG-medical-assistance-aeronautic.git
cd Local-RAG-medical-assistance-aeronautic
# con Ollama en ejecución, abrir el notebook maestro:
# notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb  →  Run All
```

## Limitaciones

- **Dataset proxy.** El modelo predictivo se entrenó sobre el **CCHS (Canadian
  Community Health Survey)**, una encuesta de salud de **población general
  canadiense, pública** ([Kaggle](https://www.kaggle.com/datasets/aradhanahirapara/healthcare-survey/data)).
  **No son datos de pilotos.** Se eligió como *proxy* ante la ausencia de datos
  aeromédicos abiertos; las conclusiones no son extrapolables a población
  aeronáutica sin re-entrenar con datos del dominio.
- **PCA con baja varianza explicada.** En el PCA sobre los embeddings del
  corpus, el primer componente explica ~**6,88 %** de la varianza y el segundo
  ~4,25 % (los dos juntos ~11 %): la estructura no es linealmente compacta en 2D.
  El PCA sirvió para *explorar y visualizar*, no como reducción de
  dimensionalidad productiva ni como resultado del modelo.
- **Modelo en optimización.** Las cifras corresponden al pipeline estable previo
  a la optimización de accuracy. El F1 de la clase «Riesgo» (0,48) es bajo: el
  modelo aún cambia sensibilidad por precisión. Es un trabajo en curso, no un
  sistema cerrado.
- **Corpus y etiquetas sintéticos.** Los pares pregunta/respuesta del RAG y las
  etiquetas de riesgo se generaron sintéticamente a partir de la normativa y la
  data; no sustituyen validación clínica.
- **No es un dispositivo médico.** Es un motor de investigación/apoyo. No emite
  diagnósticos ni reemplaza el juicio de un profesional.

## Documentación

La formulación del proyecto, los objetivos y el marco clínico están en el
documento de respaldo:
[Documentación de investigación](https://docs.google.com/document/d/1MERbrd4tcaVxPxwzoyYjgJ-4ecfWIIenZio1oF4cc24/edit?usp=sharing).

## Créditos

- Dataset: CCHS, Statistics Canada (vía Kaggle), uso académico.
- Normativa: documentos públicos DGAC / OACI.
- Autor: AKHAN (`akhanER2000`). Proyecto universitario, autor único.
