import json

filepath = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Texts for replacements and additions
text_1_1 = [
    "### Análisis: Desbalance de Fuentes y Sesgo de Cobertura\n",
    "El gráfico evidencia un claro desbalance en la distribución de pares Q&A por documento. Esta asimetría implica que la cobertura temática del corpus está sesgada hacia los documentos más voluminosos (como la normativa médica). Para el sistema RAG, esto significa que ante consultas ambiguas, el modelo tendrá una tendencia probabilística a recuperar información de las fuentes predominantes, lo que podría opacar regulaciones operativas críticas pero menos representadas en el dataset."
]

text_1_2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Análisis: Estrategia de Fragmentación y Profundidad de Respuesta\n",
        "* **Sobre el Chunking:** El histograma sugiere que los fragmentos de contexto se concentran fuertemente cerca del límite superior de tokens. Esta decisión de diseño prioriza mantener el **contexto normativo íntegro** por sobre la granularidad extrema. En textos legales aeronáuticos, dividir excesivamente un artículo altera su interpretación jurídica.\n",
        "* **Sobre las Respuestas:** La distribución muestra un claro predominio de respuestas breves. Si bien esto es útil para consultas operacionales puntuales, supone una limitación estructural: la profundidad explicativa se ve restringida ante escenarios aeromédicos complejos, requiriendo que el evaluador humano profundice en el documento original."
    ]
}

text_1_3 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Análisis Semántico: Sesgo en la Nube de Palabras\n",
        "El análisis léxico revela un hallazgo importante: existe un sesgo marcado hacia términos relacionados con preguntas regulatorias y de cumplimiento (\"certificación\", \"requisitos\", \"licencia\"). Esto explicita que la naturaleza del corpus está fuertemente orientada al cumplimiento legal (*compliance*) más que a la casuística médica descriptiva."
    ]
}

text_1_4 = [
    "### Análisis Topológico: Separación Semántica Parcial\n",
    "La reducción de dimensionalidad a través de PCA revela una separación semántica **parcial, no absoluta**, entre las distintas fuentes. Existen zonas de superposición (traslape) evidentes en el espacio vectorial. Esto indica que varios documentos comparten vocabulario. Es crucial destacar que no existe una independencia total entre las normativas, lo cual es coherente con el marco aeronáutico, donde las directivas médicas (DAN 67) y operativas (DAN 121) se entrelazan constantemente."
]

text_2_1_append = [
    "\n\n**Análisis de Calidad de Datos (Concentración de Faltantes):**\n",
    "Al evaluar el reporte de nulos impreso arriba, se constata que los valores faltantes no se distribuyen de manera homogénea. Tienden a concentrarse en variables sensibles (como niveles de estrés laboral o percepción de salud mental). Esta concentración no aleatoria es un hallazgo crítico, ya que exige técnicas de imputación avanzadas para no afectar la robustez del análisis en perfiles de riesgo."
]

text_2_2 = [
    "### Justificación Técnica: Supuestos de K-NN y Escalamiento\n",
    "* **KNNImputer y Sensibilidad:** Se eligió esta técnica bajo el supuesto de que pilotos con perfiles biométricos similares tendrán valores similares en las variables faltantes, preservando la estructura multivariada del estrés y la salud. Se evaluó la sensibilidad del parámetro y se seleccionó `n_neighbors=5` como punto de equilibrio para evitar tanto el sobreajuste (k muy bajo) como la dilución de la señal (k muy alto).\n",
    "* **StandardScaler:** El escalamiento fue indispensable porque algoritmos basados en distancias (como K-NN o futuros modelos de clasificación) asumen que todas las variables contribuyen equitativamente. Sin escalar, la variable `Work_hours` dominaría artificialmente sobre variables categóricas de escala menor."
]

text_2_3 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Análisis de Correlación: Hallazgos Aeronáuticos\n",
        "* **Aclaración de Codificación:** En la encuesta CCHS, para las variables de salud (General y Mental), los valores menores (1) representan un estado \"Excelente\", mientras que valores mayores (5) representan \"Malo\". Por tanto, una correlación positiva fuerte indica que cuando un factor empeora, el otro también lo hace.\n",
        "* **Hallazgo Principal:** La relación lineal más marcada del *heatmap* aparece entre el **Estado General de Salud (`Gen_health_state`) y la Salud Mental (`Mental_health_state`)**. Esto es de suma relevancia aeronáutica: valida empíricamente que la evaluación de certificación de un piloto no puede desvincular la biometría física de su estabilidad psicológica."
    ]
}

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source_str = "".join(cell.get('source', []))
        if "Interpretación Analítica: Distribución Documental y Riesgo de Sesgo" in source_str:
            cell['source'] = text_1_1
        elif "Interpretación Analítica: Topología Vectorial y Comprensión Semántica" in source_str:
            cell['source'] = text_1_4
        elif "Justificación Técnica: Limpieza de Metadata y Prevención de Sesgos" in source_str:
            if "**Análisis de Calidad de Datos (Concentración de Faltantes):**" not in source_str:
                cell['source'].extend(text_2_1_append)
        elif "Justificación Técnica: Imputación Multivariada y Escalamiento Métrico" in source_str:
            cell['source'] = text_2_2
            
    new_cells.append(cell)
    
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        if "sns.histplot" in source_str and "longitud_contexto" in source_str:
            new_cells.append(text_1_2)
        elif "WordCloud" in source_str:
            new_cells.append(text_1_3)
        elif "sns.heatmap" in source_str and "df_corr" in source_str:
            new_cells.append(text_2_3)

nb['cells'] = new_cells

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Feedback del profesor aplicado existosamente mediante la inyección del script.")
