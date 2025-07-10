# translator.py

from openai import OpenAI
import os
import logging
from typing import Optional
import time
from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe
load_dotenv()

logger = logging.getLogger(__name__)

# Configuración de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Modelos disponibles
MODEL_DEFAULT = "gpt-4o"
MODEL_FALLBACK = "gpt-3.5-turbo"

def translate_text(text: str, target_language: str = "es", max_retries: int = 2) -> Optional[str]:
    """
    Traduce un texto al idioma deseado usando la API de OpenAI.
    Incluye manejo de errores y reintentos.

    :param text: Texto a traducir.
    :param target_language: Idioma destino (por defecto: español).
    :param max_retries: Número máximo de reintentos en caso de error.
    :return: Texto traducido o None si falla.
    """
    if not client:
        logger.warning("🔑 No se encontró la API key de OpenAI. La traducción no está disponible.")
        return None
        
    # Si el texto es muy largo, lo truncamos para evitar errores
    max_tokens = 15000
    if len(text) > max_tokens:
        logger.warning(f"⚠️ Texto demasiado largo ({len(text)} caracteres). Truncando a {max_tokens} caracteres.")
        text = text[:max_tokens] + "..."
    
    # Preparar el prompt de traducción
    prompt = f"Traduce el siguiente texto al idioma '{target_language}' manteniendo el contexto y estilo original:\n\n{text}"
    
    # Intentar con el modelo principal y luego con el fallback
    models = [MODEL_DEFAULT, MODEL_FALLBACK]
    
    for model in models:
        retries = 0
        while retries <= max_retries:
            try:
                logger.info(f"🌐 Traduciendo con modelo {model} (intento {retries+1})")
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=60  # 60 segundos de timeout
                )
                
                elapsed = time.time() - start_time
                translated_text = response.choices[0].message.content.strip()
                
                logger.info(f"✅ Traducción completada en {elapsed:.2f}s usando {model}")
                return translated_text
                
            except Exception as e:
                retries += 1
                logger.warning(f"⚠️ Error al traducir con {model} (intento {retries}): {str(e)}")
                
                # Si es el último reintento con este modelo, pasamos al siguiente
                if retries > max_retries:
                    logger.error(f"❌ Fallaron todos los intentos con {model}, probando con otro modelo")
                    break
                    
                # Esperar antes de reintentar (backoff exponencial)
                wait_time = 2 ** retries
                logger.info(f"⏳ Esperando {wait_time}s antes de reintentar...")
                time.sleep(wait_time)
    
    logger.error("❌ Fallaron todos los intentos de traducción")
    return None
