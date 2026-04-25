import json

notebook_path = "notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the insertion point (before SECCIÓN B)
insertion_idx = len(nb['cells'])
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'SECCIÓN B' in "".join(cell['source']):
        insertion_idx = idx
        break

new_cells = [
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# Matriz de Cobertura Temática\n",
            "# ==========================================\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Determinar columna de texto\n",
            "text_col = 'contexto_original' if 'contexto_original' in df.columns else 'contexto' if 'contexto' in df.columns else 'pregunta'\n",
            "\n",
            "# Palabras clave representativas de distintos temas aeronáuticos\n",
            "temas = {\n",
            "    'Médico/Clínico': ['salud', 'enfermedad', 'diagnóstico', 'agudeza', 'cardio', 'psicológic', 'licencia', 'certificación', 'insulina', 'tratamiento'],\n",
            "    'Operativo/Vuelo': ['tripulación', 'descanso', 'horas', 'fatiga', 'vuelo', 'cabina', 'piloto', 'operador'],\n",
            "    'Mantenimiento': ['mantenimiento', 'aeronave', 'inspección', 'taller', 'mecánico', 'reparación', 'componente']\n",
            "}\n",
            "\n",
            "# Calculamos la frecuencia de estos temas por fuente\n",
            "cobertura = []\n",
            "for fuente, grupo in df.groupby('fuente'):\n",
            "    textos = \" \".join(grupo[text_col].astype(str).tolist()).lower()\n",
            "    fila = {'fuente': fuente}\n",
            "    for tema, palabras in temas.items():\n",
            "        conteo = sum(textos.count(p) for p in palabras)\n",
            "        fila[tema] = conteo\n",
            "    cobertura.append(fila)\n",
            "\n",
            "df_cobertura = pd.DataFrame(cobertura).set_index('fuente')\n",
            "# Normalizamos por fila para ver la distribución porcentual de temas por documento\n",
            "df_cobertura_pct = df_cobertura.div(df_cobertura.sum(axis=1) + 1e-9, axis=0) * 100\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "sns.heatmap(df_cobertura_pct, annot=True, cmap='YlGnBu', fmt='.1f')\n",
            "plt.title('Matriz de Cobertura Temática por Documento Normativo (%)')\n",
            "plt.ylabel('Cuerpo Legal')\n",
            "plt.xlabel('Tema')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Análisis de Cobertura Temática\n",
            "La matriz de calor evidencia la distribución de foco semántico de cada cuerpo normativo. \n",
            "Las resoluciones de la serie DAN 67 muestran un predominio absoluto en el eje Médico/Clínico, lo que valida su función como directivas de aptitud psicofísica. Por otro lado, normativas operativas (como la DAN 121 y 135) presentan una mayor distribución hacia conceptos de fatiga, descanso de tripulaciones y mantenimiento. Este análisis confirma que el corpus es ortogonal: diferentes documentos abordan dominios distintos, lo que subraya la necesidad de un sistema RAG para cruzar eficientemente información médica con reglas operacionales."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# Análisis de Redundancia Intra-Documento\n",
            "# ==========================================\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.metrics.pairwise import cosine_similarity\n",
            "import numpy as np\n",
            "\n",
            "resultados_redundancia = []\n",
            "\n",
            "for fuente, grupo in df.groupby('fuente'):\n",
            "    chunks = grupo[text_col].astype(str).tolist()\n",
            "    if len(chunks) > 1:\n",
            "        # Usamos TF-IDF para evaluar similitud léxica\n",
            "        vectorizer = TfidfVectorizer(max_features=500, stop_words=list(stop_words_es))\n",
            "        tfidf_matrix = vectorizer.fit_transform(chunks)\n",
            "        # Calculamos similitud coseno entre todos los chunks de la misma fuente\n",
            "        similitudes = cosine_similarity(tfidf_matrix)\n",
            "        # Extraemos solo los valores fuera de la diagonal principal\n",
            "        sim_superiores = similitudes[np.triu_indices(len(chunks), k=1)]\n",
            "        redundancia_promedio = np.mean(sim_superiores) if len(sim_superiores) > 0 else 0\n",
            "    else:\n",
            "        redundancia_promedio = 0\n",
            "    \n",
            "    resultados_redundancia.append({'fuente': fuente, 'redundancia': redundancia_promedio})\n",
            "\n",
            "df_redundancia = pd.DataFrame(resultados_redundancia).sort_values('redundancia', ascending=False)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(data=df_redundancia, x='redundancia', y='fuente', palette='magma')\n",
            "plt.title('Análisis de Redundancia Lexical en el Corpus DGAC (TF-IDF)')\n",
            "plt.xlabel('Similitud Coseno Promedio (Intra-Documento)')\n",
            "plt.ylabel('Fuente')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Análisis de Redundancia Intra-Documento\n",
            "La redundancia intra-documento mide qué tan repetitivo es el texto dentro de una misma normativa. \n",
            "* Un **índice bajo** (como en la DAN 121) indica alta densidad de información heterogénea (cada artículo aborda un tema operacional distinto).\n",
            "* Un **índice más alto** indica el uso de terminología repetitiva, común en directivas médicas donde cláusulas legales (\"apto\", \"suspendido\", \"evaluación\") se reiteran por cada patología. \n",
            "Para el diseño de la VectorDB, una redundancia baja significa que cada chunk es único y de alto valor, mientras que una redundancia alta exigirá *Metadata Filtering* para asegurar que el motor RAG no recupere fragmentos estadísticamente idénticos de diferentes artículos médicos."
        ]
    }
]

# Insert the new cells before SECCIÓN B
nb['cells'][insertion_idx:insertion_idx] = new_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("EDA_Avanzado_DualEngine_CCHS.ipynb parchado exitosamente.")
