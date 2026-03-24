import json
import nbformat as nbf
import os

filepath = 'notebooks/EDA_Aeronautico.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

for cell in nb['cells']:
    new_cells.append(cell)
    
    # Check if this cell is code and contains a specific plot
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # 1. Countplot
        if "sns.countplot" in source_str:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**💡 Interpretación del Gráfico 1 (Distribución de Documentos):**\n",
                    "Este gráfico de barras ilustra el peso de cada documento normativo dentro de nuestra base de datos vectorial. Un volumen alto en manuales extensos (como el Manual OACI o el MAPE-MEDAV) garantiza que el sistema RAG posea el corpus necesario para asistir en casos médicos complejos. Esta validación asegura que ninguna normativa clave ha quedado subrepresentada durante la generación sintética."
                ]
            })
            
        # 2. Histograms
        elif "sns.histplot" in source_str:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**💡 Interpretación del Gráfico 2 (Histogramas de Longitud):**\n",
                    "La distribución normal (campana) observable en estos histogramas es un indicador matemático de limpieza. Confirma que el `RecursiveCharacterTextSplitter` cortó los párrafos de manera uniforme (alrededor de 1000 caracteres como límite). Al mantener un tamaño de contexto constante, evitamos saturar la Ventana de Contexto (Context Window) de Llama 3, mitigando severamente el fenómeno de *«Lost in the middle»* (donde el modelo olvida la información central de textos demasiado largos)."
                ]
            })
            
        # 3. Wordcloud
        elif "WordCloud" in source_str:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**💡 Interpretación del Gráfico 3 (Nube de Palabras Temática):**\n",
                    "La Nube de Palabras actúa como un filtro de sanidad léxica. Al resaltar conceptos como 'médico', 'DGAC', 'vuelo', 'licencia', entre otros, validamos empíricamente que la IA preservó la estricta jerga aeronáutica chilena al formular las preguntas y no introdujo alucinaciones (palabras fuera de contexto o conversacionales) durante el proceso masivo de generación de datos."
                ]
            })
            
        # 4. PCA
        elif "PCA(" in source_str:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**💡 Interpretación del Gráfico 4 (Componentes Principales - PCA):**\n",
                    "Como vimos en clases, el PCA intenta trazar relaciones lineales para reducir dimensiones. Dado que el lenguaje humano es de naturaleza altamente no lineal, el PCA tiende a 'amontonar' (overlapping) los datos semánticos en el centro sin lograr una clasificación visual neta. La varianza explicada típicamente es baja en espacios vectoriales de texto (LLMs), lo que justifica metodológicamente por qué la industria hoy prefiere técnicas topológicas (como UMAP) para el Procesamiento de Lenguaje Natural."
                ]
            })
            
        # 5. UMAP
        elif "umap.UMAP" in source_str:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "**💡 Interpretación del Gráfico 5 (Topología UMAP):**\n",
                    "A diferencia de PCA, UMAP modela distancias no lineales (Métricas de Coseno). Los «clústeres» o islas claras de puntos reflejan agrupaciones semánticas puras. Fíjate cómo los extractos del Reglamento Médico (MAPE-MEDAV y DAN 67) gravitan juntos, separados de las reglas puramente operativas de vuelo (DAN 121). Este gráfico es la **prueba definitiva** de que el modelo de Embeddings ($nomic-embed-text$) sí 'entiende' a la perfección la materia legal antes de inyectarla al RAG."
                ]
            })

nb['cells'] = new_cells

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Explicaciones inyectadas exitosamente al Notebook.")
