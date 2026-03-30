import json

filepath = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encuentra la última celda de código que invoca a Mistral
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source
            
        if "mistral" in source_str.lower() and "ollama" in source_str.lower() and "prompt" in source_str.lower():
            # Nuevo código inyectando las instrucciones del usuario
            # Usando df_corr porque en la celda 8 la matriz de correlación se llama df_corr
            nuevo_codigo = """# ==========================================
# 5. INFERENCIA LLM: CONEXIÓN TABULAR - NORMATIVA
# ==========================================
from ollama import Client

# Extraemos la fila de correlaciones respecto a la Salud General
# Adaptado inteligentemente a df_corr de la celda 8 para que compile sin fallos
corr_salud = df_corr['Gen_health_state'].drop('Gen_health_state').to_dict()

cliente = Client(host='http://localhost:11434')

# EL PROMPT PUENTE: Aquí conectamos el CSV con el contexto de los PDFs
prompt_medico = f\"\"\"
Eres un Médico Aeronáutico Evaluador (CMA) experto en las normativas chilenas DGAC (DAN 67) y directivas de la OACI. 
Acabo de procesar una base de datos clínica de postulantes a pilotos. Analiza las siguientes correlaciones de Pearson respecto a su Estado de Salud General ('Gen_health_state').

Para tu conocimiento técnico, estas son las variables:
- 'Mental_health_state': Estado de salud mental.
- 'Stress_level': Nivel de estrés general.
- 'Work_stress': Estrés laboral (Fatiga).
- 'BMI_18_above': Índice de Masa Corporal.
- 'Smoked' / 'Weekly_alcohol': Hábitos de consumo.

TUS TAREAS:
1. Identifica las 3 relaciones estadísticas más críticas de los datos.
2. Para cada una, explica cómo esta condición médica podría afectar la certificación de un piloto (ej. Clase 1 o 2) basándote en los criterios médicos aeronáuticos estándar.
3. Concluye brevemente por qué es fundamental que un evaluador médico utilice un sistema RAG documental (como el que estamos construyendo) para consultar la norma exacta ante estos factores de riesgo.

Responde de forma técnica, en español, numerando las conclusiones.
Datos matemáticos de correlación: {corr_salud}
\"\"\"

print("Transfiriendo contexto tabular y normativo al LLM (Cargando modelo local: mistral)...")
try:
    resp = cliente.chat(model='mistral', messages=[{'role': 'user', 'content': prompt_medico}])
    print("\\n🚀 CONCLUSIONES MÉDICO-AERONÁUTICAS (MISTRAL):")
    print("-" * 75)
    print(resp['message']['content'].strip())
    print("-" * 75)
except Exception as e:
    print(f"Error invocando a Ollama/Mistral: {e}")"""
            
            if isinstance(source, list):
                nb['cells'][idx]['source'] = [nuevo_codigo]
            else:
                nb['cells'][idx]['source'] = nuevo_codigo
            break

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("El Prompt Puente ha sido inyectado magistralmente en la Celda 9.")
