import fitz  # PyMuPDF
import json
import os
import glob
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================
CARPETA_CRUDOS = "datos_crudos/"
RUTA_SALIDA = "dataset_sintetico/mega_dataset_aeronautico.jsonl"
MODELO_GENERADOR = "llama3"

cliente_ollama = Client(host='http://localhost:11434')

def extraer_texto_pdf(ruta_archivo, pagina_inicio=5):
    """Extrae texto saltándose las primeras páginas (índices/portadas)."""
    texto_completo = ""
    try:
        documento = fitz.open(ruta_archivo)
        # Empezamos un poco más adelante para evitar índices puros
        for num_pagina in range(min(pagina_inicio, len(documento)), len(documento)):
            texto_completo += documento[num_pagina].get_text("text") + "\n"
        return texto_completo
    except Exception as e:
        print(f"❌ Error leyendo {ruta_archivo}: {e}")
        return ""

def fragmentar_texto(texto):
    """Divide el texto en pedazos lógicos."""
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    return separador.split_text(texto)

def generar_multiples_qa(fragmento, nombre_documento):
    """Obliga al modelo a generar 3 preguntas por cada fragmento de texto."""
    prompt_sistema = """Eres un examinador médico y auditor aeronáutico experto de la DGAC de Chile. 
    Tu tarea es leer el fragmento de un reglamento y generar hasta 3 preguntas técnicas complejas y sus respuestas.
    
    REGLA 1: Si el texto es solo un índice, tabla de firmas, o no contiene información útil, responde ÚNICAMENTE con la palabra: IGNORAR.
    REGLA 2: Si tiene información útil, responde ÚNICAMENTE con un arreglo JSON válido que contenga entre 1 y 3 pares de pregunta/respuesta. 
    
    Formato estricto:
    [
      {"pregunta": "pregunta 1", "respuesta": "respuesta 1"},
      {"pregunta": "pregunta 2", "respuesta": "respuesta 2"}
    ]
    """
    
    prompt_usuario = f"Documento de origen: {nombre_documento}\nFragmento:\n\n{fragmento}\n\nGenera el JSON o escribe IGNORAR:"

    try:
        respuesta = cliente_ollama.chat(model=MODELO_GENERADOR, messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': prompt_usuario}
        ])
        
        contenido = respuesta['message']['content'].strip()
        
        if "IGNORAR" in contenido.upper():
            return []
            
        # Limpiar posibles bloques de código Markdown que el modelo a veces añade
        if contenido.startswith("```json"):
            contenido = contenido[7:-3].strip()
        elif contenido.startswith("```"):
            contenido = contenido[3:-3].strip()
            
        lista_qa = json.loads(contenido)
        
        # Le inyectamos el contexto y la fuente a cada pregunta generada
        for qa in lista_qa:
            qa['contexto_original'] = fragmento 
            qa['fuente'] = nombre_documento
            
        return lista_qa
    except Exception as e:
        # Si falla el formato, lo descartamos silenciosamente y seguimos
        return []

# ==========================================
# EJECUCIÓN PRINCIPAL (PROCESAMIENTO POR LOTES)
# ==========================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    
    # Buscar todos los PDFs en la carpeta de datos crudos
    lista_pdfs = glob.glob(os.path.join(CARPETA_CRUDOS, "*.pdf"))
    
    if not lista_pdfs:
        print("❌ No se encontraron archivos PDF en la carpeta datos_crudos.")
        exit()
        
    print(f"📚 Se encontraron {len(lista_pdfs)} documentos para procesar.")
    total_pares_generados = 0
    
    # Abrimos el archivo de salida en modo 'append' (agregar al final)
    with open(RUTA_SALIDA, 'w', encoding='utf-8') as archivo_salida:
        
        # Iterar sobre cada PDF encontrado
        for ruta_pdf in lista_pdfs:
            nombre_archivo = os.path.basename(ruta_pdf)
            print(f"\n⚙️ Procesando: {nombre_archivo}")
            
            texto_pdf = extraer_texto_pdf(ruta_pdf)
            if not texto_pdf:
                continue
                
            fragmentos = fragmentar_texto(texto_pdf)
            print(f"   ✂️ Se dividió en {len(fragmentos)} fragmentos.")
            
            # Procesar cada fragmento del PDF actual
            for chunk in tqdm(fragmentos, desc=f"Extrayendo de {nombre_archivo[:15]}..."):
                if len(chunk.strip()) < 100:
                    continue
                    
                # Ahora esta función nos devuelve una lista de preguntas (hasta 3)
                preguntas_generadas = generar_multiples_qa(chunk, nombre_archivo)
                
                for qa in preguntas_generadas:
                    archivo_salida.write(json.dumps(qa, ensure_ascii=False) + '\n')
                    total_pares_generados = total_pares_generados + 1  # type: ignore
                    
    print(f"\n🚀 ¡MEGA DATASET COMPLETADO!")
    print(f"📊 Se generaron un total de {total_pares_generados} pares Q&A de alta calidad.")
    print(f"📂 Archivo consolidado guardado en: {RUTA_SALIDA}")
