import json
import os

filepath = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new markdown text blocks structured as list of strings (lines)
markdown_1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Interpretación Analítica: Distribución Documental y Riesgo de Sesgo\n",
        "Al observar la distribución de pares Q&A por documento, se evidencia un claro **desbalance de clases** entre las fuentes normativas. Las directivas médicas (como la DAN 67) presentan una volumetría significativamente mayor en comparación con reglamentos puramente operativos (DAN 121 o DAN 135). \n",
        "\n",
        "**Relevancia y Limitaciones para el Sistema RAG:**\n",
        "Desde una perspectiva analítica, este desbalance introduce un riesgo de **sesgo semántico** en la fase de recuperación (Retrieval). Si un evaluador médico realiza una consulta ambigua, el motor de similitud del coseno tendrá una probabilidad estadística mayor de recuperar fragmentos de la DAN 67 por simple predominancia de tokens. \n",
        "Para mitigar este sesgo en futuras iteraciones, se requerirá implementar *Metadata Filtering* (filtrado por metadatos) en la base de datos vectorial, obligando al sistema a buscar exclusivamente en la normativa operativa cuando el contexto lo amerite, garantizando así la precisión legal de la respuesta aeronáutica."
    ]
}

markdown_2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Interpretación Analítica: Topología Vectorial y Comprensión Semántica\n",
        "La reducción de dimensionalidad mediante PCA (de 768 dimensiones a 2D) nos permite visualizar cómo el modelo local `nomic-embed-text` está interpretando el corpus. \n",
        "\n",
        "**Análisis de Patrones:**\n",
        "Se observan clústeres definidos según la fuente del documento. Esto demuestra que el modelo de *embeddings* no solo está leyendo palabras aisladas, sino que **ha comprendido matemáticamente la diferencia semántica** entre un concepto clínico-biológico (ej. diabetes, glicemia) y un concepto procedimental (ej. horas de vuelo, descanso). \n",
        "\n",
        "**Relevancia Aeronáutica:**\n",
        "Esta separación topológica es el fundamento que hace viable nuestro sistema RAG. Nos asegura que cuando el Médico Evaluador consulte por \"riesgo de fatiga\", el vector de búsqueda colisionará matemáticamente con el clúster de factores humanos operativos, y no con requerimientos técnicos de las aeronaves, evitando falsos positivos que podrían resultar en la certificación errónea de un piloto."
    ]
}

new_cells = []
for cell in nb['cells']:
    new_cells.append(cell)
    
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        
        # Check if it's the countplot cell
        if "sns.countplot" in source_str and "fuente" in source_str:
            new_cells.append(markdown_1)
            
        # Check if it's the PCA scatter plot cell
        elif "sns.scatterplot" in source_str and "PCA_1" in source_str:
            new_cells.append(markdown_2)

nb['cells'] = new_cells

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Celdas académicas inyectadas con éxito.")
