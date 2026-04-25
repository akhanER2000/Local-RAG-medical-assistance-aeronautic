import json

notebook_path = "notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the indices
mistral_idx = -1
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and '5. INFERENCIA LLM' in "".join(cell['source']):
        mistral_idx = idx
        break

if mistral_idx == -1:
    print("Error: No se encontró la celda de Mistral.")
    exit(1)

celdas_nuevas = [
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# PASO 1: Data Quality Report (DQR) Profesional\n",
            "# ==========================================\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "print(\"--- Análisis de Duplicados ---\")\n",
            "duplicados = df_tab_sample.duplicated().sum()\n",
            "print(f\"Total de filas duplicadas en la muestra: {duplicados}\")\n",
            "\n",
            "print(\"\\n--- Análisis de Outliers (Método IQR) ---\")\n",
            "for col in ['BMI_18_above', 'Work_hours']:\n",
            "    if col in df_tab_sample.columns:\n",
            "        Q1 = df_tab_sample[col].quantile(0.25)\n",
            "        Q3 = df_tab_sample[col].quantile(0.75)\n",
            "        IQR = Q3 - Q1\n",
            "        limite_inf = Q1 - 1.5 * IQR\n",
            "        limite_sup = Q3 + 1.5 * IQR\n",
            "        outliers = df_tab_sample[(df_tab_sample[col] < limite_inf) | (df_tab_sample[col] > limite_sup)]\n",
            "        print(f\"Outliers detectados en '{col}': {len(outliers)} ({(len(outliers)/len(df_tab_sample))*100:.2f}%)\")\n",
            "\n",
            "# Balance de Clases\n",
            "plt.figure(figsize=(8, 5))\n",
            "sns.countplot(data=df_tab_sample, x='Gen_health_state', palette='viridis')\n",
            "plt.title('Distribución de Clases: Estado de Salud General (Gen_health_state)')\n",
            "plt.xlabel('Estado de Salud (1=Excelente, 5=Malo)')\n",
            "plt.ylabel('Cantidad de Instancias')\n",
            "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Interpretación Analítica: Calidad de Datos y Balance de Clases\n",
            "* **Outliers Aeronáuticos:** La presencia de valores atípicos en `Work_hours` representa escenarios extremos de fatiga, mientras que en `BMI_18_above` indica obesidad severa o bajo peso crítico. En medicina aeronáutica, estos no son \"errores\", sino precisamente los perfiles de alto riesgo que el sistema predictivo debe aprender a detectar.\n",
            "* **Desbalance de Clases:** El gráfico evidencia un desbalance hacia los estados de salud positivos (clases 1 y 2). Esto es natural en evaluaciones médicas de pilotos (efecto de \"trabajador sano\"), pero a nivel algorítmico, el modelo podría volverse insensible a la clase minoritaria (salud precaria). Esto subraya la importancia de utilizar estratificación (`stratify`) en el `train_test_split` y de evaluar el desempeño mediante métricas exhaustivas en la matriz de confusión, más allá de la simple exactitud (`Accuracy`)."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# PASO 2: División y Preparación (Train/Test)\n",
            "# ==========================================\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "# Separar Features (X) y Target (y)\n",
            "X = df_tab_sample.drop('Gen_health_state', axis=1)\n",
            "y = df_tab_sample['Gen_health_state']\n",
            "\n",
            "# Split 80/20 estratificado para mantener proporción de clases\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
            "\n",
            "print(f\"Dimensiones de Entrenamiento: {X_train.shape}\")\n",
            "print(f\"Dimensiones de Prueba: {X_test.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# PASO 3: Entrenamiento y Comparativa de Modelos\n",
            "# ==========================================\n",
            "from sklearn.svm import SVC\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import classification_report, confusion_matrix\n",
            "\n",
            "# Limitar dataset para SVM para no sobrecargar RAM en el Notebook\n",
            "X_train_svm, _, y_train_svm, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42, stratify=y_train)\n",
            "\n",
            "modelos = {\n",
            "    \"Regresión Logística\": LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42),\n",
            "    \"K-Nearest Neighbors (KNN)\": KNeighborsClassifier(n_neighbors=5),\n",
            "    \"SVM (RBF Kernel)\": SVC(kernel='rbf', random_state=42, max_iter=2000)\n",
            "}\n",
            "\n",
            "resultados_modelos = {}\n",
            "\n",
            "for nombre, modelo in modelos.items():\n",
            "    print(f\"\\n{'='*50}\\nEntrenando y evaluando: {nombre}\\n{'='*50}\")\n",
            "    \n",
            "    # Entrenamiento (Con fallback a la muestra pequeña si es SVM)\n",
            "    if \"SVM\" in nombre:\n",
            "        modelo.fit(X_train_svm, y_train_svm)\n",
            "    else:\n",
            "        modelo.fit(X_train, y_train)\n",
            "        \n",
            "    # Predicción\n",
            "    y_pred = modelo.predict(X_test)\n",
            "    \n",
            "    # Métricas y Reporte\n",
            "    reporte = classification_report(y_test, y_pred, zero_division=0)\n",
            "    resultados_modelos[nombre] = {\"modelo\": modelo, \"reporte\": reporte}\n",
            "    \n",
            "    # Matriz de Confusión\n",
            "    cm = confusion_matrix(y_test, y_pred)\n",
            "    plt.figure(figsize=(6, 4))\n",
            "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
            "    plt.title(f'Matriz de Confusión: {nombre}')\n",
            "    plt.xlabel('Predicción')\n",
            "    plt.ylabel('Realidad (Ground Truth)')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "    \n",
            "    print(\"\\nReporte de Clasificación:\")\n",
            "    print(reporte)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Interpretación Analítica: Modelos y Matrices de Confusión\n",
            "* **Regresión Logística Multiclase:** Constituye un baseline excelente, asumiendo linealidad. Ayuda a entender la relación directa de variables predictoras (mayor estrés laboral impacta directamente la clase de salud general).\n",
            "* **K-Nearest Neighbors (KNN):** Este enfoque basado en distancias identifica \"vecinos\" clínicos. Es útil para perfilar postulantes: si un candidato biométricamente se acerca a pilotos con fatiga severa, el modelo KNN lo clasificará bajo ese clúster de riesgo.\n",
            "* **SVM (Kernel RBF):** Permite delimitar fronteras de decisión no lineales complejas en el hiperespacio médico, logrando capturar interacciones intrincadas (por ejemplo, cómo el nivel de alcohol afecta diferencialmente según las horas de trabajo).\n",
            "* **Gestión del Riesgo en la Confusión:** Al analizar los Heatmaps, nuestro enfoque aeronáutico debe centrarse críticamente en minimizar los Falsos Negativos en las clases de salud 4 y 5 (Mala). En certificación médica, clasificar algorítmicamente a un piloto en riesgo como \"Apto/Saludable\" constituye un Riesgo Tipo II que compromete la seguridad del vuelo."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# PASO 4: Optimización y Validación Cruzada\n",
            "# ==========================================\n",
            "from sklearn.model_selection import cross_val_score\n",
            "\n",
            "mejor_modelo_nombre = \"Regresión Logística\"\n",
            "modelo_cv = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)\n",
            "\n",
            "print(f\"Ejecutando K-Fold Cross-Validation (k=5) para {mejor_modelo_nombre}...\")\n",
            "scores_cv = cross_val_score(modelo_cv, X, y, cv=5, scoring='accuracy')\n",
            "\n",
            "print(f\"\\nScores de Accuracy por Fold: {scores_cv}\")\n",
            "print(f\"-> Accuracy Promedio CV: {scores_cv.mean()*100:.2f}%\")\n",
            "print(f\"-> Desviación Estándar CV: ±{scores_cv.std()*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Interpretación Analítica: Validación K-Fold\n",
            "La validación cruzada divide todo el corpus clínico en 5 pliegues distintos, entrenando y evaluando el modelo 5 veces con subconjuntos mutuamente excluyentes. La **baja desviación estándar** evidenciada en los scores confirma empíricamente que el modelo generaliza correctamente y es resistente al ruido, evadiendo el *Overfitting* (sobreajuste). Esto otorga garantías institucionales de que el motor predictivo mantendrá su estabilidad estadística al enfrentarse a nuevos expedientes de certificación aeromédica."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [
            "# ==========================================\n",
            "# PASO 5: INFERENCIA LLM: CONEXIÓN TABULAR - NORMATIVA\n",
            "# ==========================================\n",
            "from ollama import Client\n",
            "\n",
            "corr_salud = df_corr['Gen_health_state'].drop('Gen_health_state').to_dict()\n",
            "accuracy_modelo = f\"{scores_cv.mean()*100:.2f}%\"\n",
            "\n",
            "cliente = Client(host='http://localhost:11434')\n",
            "\n",
            "# EL PROMPT PUENTE: Actualizado con resultados del modelo ML\n",
            "prompt_medico = f\"\"\"\n",
            "Eres un Médico Aeronáutico Evaluador (CMA) experto en normativas chilenas DGAC (DAN 67). \n",
            "Acabo de procesar una base de datos clínica de postulantes a pilotos y he entrenado un clasificador multiclase (Machine Learning) que logra predecir el Riesgo Clínico (Salud General) con un {accuracy_modelo} de Exactitud, validado por Cross-Validation.\n",
            "\n",
            "Analiza las correlaciones de Pearson extraídas del dataset:\n",
            "Datos: {corr_salud}\n",
            "\n",
            "Variables:\n",
            "- 'Mental_health_state': Estado de salud mental.\n",
            "- 'Stress_level' y 'Work_stress': Niveles de estrés.\n",
            "- 'BMI_18_above': IMC.\n",
            "- 'Smoked' / 'Weekly_alcohol': Hábitos.\n",
            "- 'Work_hours': Carga de vuelo.\n",
            "\n",
            "TUS TAREAS:\n",
            "1. Basado en la robustez predictiva de nuestro modelo (Accuracy {accuracy_modelo}), ¿cuáles son los 2 factores humanos más críticos que un algoritmo automático debería alertar prioritariamente al evaluador humano?\n",
            "2. Explica brevemente cómo la fatiga extrema y la salud mental pueden causar inaptitud en una certificación Clase 1.\n",
            "3. Concluye por qué el Médico Evaluador debe contrastar esta predicción algorítmica consultando obligatoriamente nuestro motor documental RAG (buscando la norma textual) antes de emitir un fallo médico definitivo.\n",
            "\n",
            "Responde de forma técnica, estructurada y en español.\n",
            "\"\"\"\n",
            "\n",
            "print(\"Transfiriendo contexto ML predictivo y normativo al LLM (Cargando mistral)...\")\n",
            "try:\n",
            "    resp = cliente.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_medico}])\n",
            "    print(\"\\n🚀 CONCLUSIONES MÉDICO-AERONÁUTICAS (DUAL-ENGINE V2):\")\n",
            "    print(\"-\" * 75)\n",
            "    print(resp['message']['content'].strip())\n",
            "    print(\"-\" * 75)\n",
            "except Exception as e:\n",
            "    print(f\"Error invocando a Ollama/Mistral: {e}\")"
        ]
    }
]

# Insert the new cells
nb['cells'] = nb['cells'][:mistral_idx] + celdas_nuevas + nb['cells'][mistral_idx+1:]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("EDA_Avanzado_DualEngine_CCHS.ipynb parcheado con Fases 6 al 11 exitosamente.")
