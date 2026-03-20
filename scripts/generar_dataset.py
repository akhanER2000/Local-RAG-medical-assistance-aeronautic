import fitz  # PyMuPDF
import json
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================
# Asegúrate de poner el nombre exacto de tu PDF aquí
RUTA_PDF = "datos_crudos/MAPE-MEDAV-ED-1-Enm-1-25.MAR.2025.pdf" 
RUTA_SALIDA = "dataset_sintetico/qa_medico_aeronautico.jsonl"
MODELO_GENERADOR = "llama3"

# Inicializar cliente local de Ollama
cliente_ollama = Client(host='http://localhost:11434')

def extraer_texto_pdf(ruta_archivo, pagina_inicio=8):
    """Extrae texto, saltándose los índices y tablas de contenido iniciales."""
    print(f"📄 Leyendo el PDF: {ruta_archivo} (Desde la página {pagina_inicio})")
    texto_completo = ""
    try:
        documento = fitz.open(ruta_archivo)
        # Empezamos a leer desde la página donde realmente empieza la materia (ej. pág 8)
        for num_pagina in range(pagina_inicio, len(documento)):
            texto_completo += documento[num_pagina].get_text("text") + "\n"
        return texto_completo
    except Exception as e:
        print(f"❌ Error leyendo el PDF: {e}")
        return ""

def fragmentar_texto(texto):
    """Divide el texto gigante en pedazos procesables (chunks)."""
    print("✂️ Fragmentando el documento...")
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Tamaño ideal para que el modelo no pierda contexto
        chunk_overlap=200, # Solapamiento para no cortar ideas por la mitad
        length_function=len
    )
    return separador.split_text(texto)

def generar_qa_sintetico(fragmento):
    """Envía el fragmento a Ollama con una vía de escape para texto inútil."""
    prompt_sistema = """Eres un examinador médico aeronáutico experto de la DGAC de Chile. 
    Tu tarea es leer el fragmento de texto y generar UNA pregunta técnica y su respuesta.
    
    REGLA 1: Si el texto es solo un índice, tabla de contenidos, o no contiene información útil o reglamentaria, responde ÚNICAMENTE con la palabra: IGNORAR.
    REGLA 2: Si el texto sí contiene información útil, responde ÚNICAMENTE en este formato JSON válido:
    {"pregunta": "tu pregunta", "respuesta": "tu respuesta basada ESTRICTAMENTE en el texto"}
    """
    
    prompt_usuario = f"Fragmento:\n\n{fragmento}\n\nGenera el JSON o escribe IGNORAR:"

    try:
        respuesta = cliente_ollama.chat(model=MODELO_GENERADOR, messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': prompt_usuario}
        ])
        
        contenido = respuesta['message']['content'].strip()
        
        # Si el modelo decide que el pedazo de texto es inútil (como un índice), lo saltamos
        if "IGNORAR" in contenido.upper():
            return None
            
        json_limpio = json.loads(contenido)
        json_limpio['contexto_original'] = fragmento 
        return json_limpio
    except Exception as e:
        return None

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

    # 1. Extraer texto
    texto_pdf = extraer_texto_pdf(RUTA_PDF)
    
    if texto_pdf:
        # 2. Fragmentar
        fragmentos = fragmentar_texto(texto_pdf)
        print(f"✅ Se generaron {len(fragmentos)} fragmentos de texto.")
        
        # Opcional: Para probar rápido, procesamos solo los primeros 10 fragmentos
        # Cambia fragmentos[:10] por fragmentos para procesar todo el PDF.
        fragmentos_prueba = fragmentos
        
        print("🧠 Iniciando generación sintética con Llama 3 local...")
        pares_qa_generados: int = 0
        
        # 3. Generar y guardar progresivamente
        with open(RUTA_SALIDA, 'w', encoding='utf-8') as archivo_salida:
            for chunk in tqdm(fragmentos_prueba, desc="Procesando Chunks"):
                if len(chunk.strip()) < 100: # Ignorar pedazos muy cortos o vacíos
                    continue
                    
                qa_json = generar_qa_sintetico(chunk)
                
                if qa_json:
                    archivo_salida.write(json.dumps(qa_json, ensure_ascii=False) + '\n')
                    pares_qa_generados = pares_qa_generados + 1  # type: ignore
                    
        print(f"\n🚀 ¡Proceso completado! Se generaron {pares_qa_generados} pares Q&A de alta calidad.")
        print(f"📂 Archivo guardado en: {RUTA_SALIDA}")
