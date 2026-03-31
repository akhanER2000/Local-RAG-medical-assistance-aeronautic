import json
import os

filepath = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Justificación Técnica: Limpieza de Metadata y Prevención de Sesgos\n",
        "La purga de códigos como `96` (Valid skip) o `99` (Not stated) no es un mero ejercicio de limpieza estándar, sino una decisión crítica para la **seguridad del modelo predictivo aeronáutico**. \n",
        "\n",
        "**Relevancia para la Toma de Decisiones:**\n",
        "Si estos valores se mantuvieran en el dataset, los algoritmos de Machine Learning los interpretarían como magnitudes reales (por ejemplo, asumiendo que un piloto tiene \"99 horas de estrés\" o un \"IMC de 99\"). Esto generaría gradientes de error masivos y falsas correlaciones, llevando al sistema a predecir niveles de riesgo totalmente distorsionados. Al transformar estos códigos de gestión de la encuesta en nulos matemáticos (`np.nan`), garantizamos que el modelo aprenda exclusivamente de la biometría y conducta real del piloto."
    ]
}

markdown_2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Justificación Técnica: Imputación Multivariada y Escalamiento Métrico\n",
        "Para la preparación de la matriz de entrenamiento, se han tomado dos decisiones arquitectónicas fundamentadas en la naturaleza de los datos biométricos:\n",
        "\n",
        "1. **Uso de KNNImputer frente a Imputación Simple:** La salud humana es un sistema multivariado. Reemplazar un dato faltante de Índice de Masa Corporal (IMC) por el promedio global introduciría un sesgo severo, ignorando la realidad biológica. El algoritmo `KNNImputer` resuelve esto buscando a los \"5 vecinos más cercanos\" (pilotos con perfiles de estrés, hábitos y edades similares) para deducir el IMC faltante, preservando la varianza y correlación original de los datos.\n",
        "2. **Uso de StandardScaler:** Algoritmos de frontera de decisión que planeamos utilizar en etapas posteriores (como Support Vector Machines - SVM) son altamente sensibles a las distancias euclidianas. Variables como `Work_hours` (escala 0-60) y variables categóricas de salud (escala 1-5) no pueden competir matemáticamente tal como están. La estandarización centra los datos (media 0, varianza 1), permitiendo que la GPU procese las distancias correctamente sin darle un \"peso\" artificial a las variables con magnitudes más grandes."
    ]
}

new_cells = []
for cell in nb['cells']:
    new_cells.append(cell)
    
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        
        # Check if it's the Cell 6 (Cleaning Data Dictionary)
        if "df_tab.isnull().sum()" in source_str and "7, 8, 9, 7.0" in source_str:
            new_cells.append(markdown_1)
            
        # Check if it's the Cell 7 (KNNImputer and StandardScaler)
        elif "KNNImputer(n_neighbors=5)" in source_str and "StandardScaler" in source_str:
            new_cells.append(markdown_2)

nb['cells'] = new_cells

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Celdas académicas B inyectadas con éxito.")
