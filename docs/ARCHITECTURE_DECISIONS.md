# Decisiones de Arquitectura: Sistema Integral de Evaluación Médica Aeronáutica (Dual-Engine)

## 1. Justificación de Base de Datos Vectorial (FAISS / ChromaDB)
**Decisión:** Utilizar un motor de búsqueda de similitud vectorial (VectorDB) eficiente en memoria como FAISS.
**Justificación Técnica:**
* **Eficiencia en Recuperación (Retrieval):** El corpus aeronáutico de la DGAC/OACI contiene más de 14.000 fragmentos normativos. Realizar una búsqueda secuencial (Full-Text Search) sobre este volumen para encontrar directivas médicas específicas es computacionalmente ineficiente e incapaz de comprender el contexto semántico. 
* **Búsqueda Semántica:** FAISS indexa los *embeddings* matemáticos, permitiendo calcular la "distancia del coseno" o la "distancia euclidiana" (L2) en milisegundos. Esto asegura que si un evaluador médico busca "riesgo cardiovascular", recupere artículos sobre "hipertensión" o "infarto miocárdico" aunque las palabras exactas no coincidan.
* **Optimización de Memoria RAM:** FAISS permite técnicas de cuantización (ej. Product Quantization - PQ), comprimiendo los vectores para que la base de conocimientos entera pueda residir en la memoria RAM de hardware con recursos limitados.

## 2. Modelo de Embeddings y Estrategia de Fragmentación (Chunking)
**Decisión:** Utilizar el modelo `nomic-embed-text` de Ollama con un tamaño de *chunk* conservador orientado al contexto legal.
**Justificación Técnica:**
* **Optimización de Contexto Largo:** `nomic-embed-text` está matemáticamente diseñado para operar con una longitud de contexto masiva (hasta 8192 tokens), capturando el sentido global de resoluciones jurídicas complejas (como la DAN 67), a diferencia de modelos estándar truncados a 512 tokens.
* **Trade-off VRAM vs Retención Legal:** Dividir un artículo legal de aviación en fragmentos demasiado cortos (ej. 100 caracteres) destruye el condicional jurídico (ej. "Aplica la suspensión, *siempre que*..."). Hemos optado por *chunks* cercanos al límite del token, preservando la integridad del artículo a expensas de un mayor consumo temporal de VRAM durante la vectorización, asumiendo este costo durante el pre-procesamiento para asegurar 0% de pérdida de interpretabilidad en la respuesta.

## 3. Motor Generativo Local (Mistral-7B)
**Decisión:** Ejecución local (Air-Gapped) del Large Language Model (LLM) Mistral a través de Ollama.
**Justificación Técnica:**
* **Privacidad y Cumplimiento Regulatorio (Air-Gapped):** El sistema procesa PIV (Información Médica Privada) de postulantes a pilotos, como historial de salud mental, fatiga e IMC. Transmitir estas matrices clínicas a APIs de terceros en la nube (como OpenAI o Anthropic) violaría los principios de confidencialidad médica y regulaciones de ciberseguridad aeroespacial. La ejecución en hardware local garantiza que ningún dato sale del recinto de evaluación.
* **Razonamiento y Lógica Experta:** La arquitectura 7B de Mistral ha demostrado superar a modelos de su misma escala en benchmarks lógicos. Al acoplarse con la recuperación RAG, Mistral no necesita retener todo el conocimiento de la OACI en sus pesos internos (evitando alucinaciones), sino que se limita a *razonar* sobre el contexto explícito que la VectorDB le inyecta, actuando como un evaluador médico determinista.
