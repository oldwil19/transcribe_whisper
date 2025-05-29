# translator.py

import openai
import os
import logging
from typing import Optional
from datetime import datetime
import asyncio

# Configurar logging
logger = logging.getLogger(__name__)

# Configuración de la API
API_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 4096,
    "timeout": 30  # segundos
}

class TranslationConfig(BaseModel):
    model: str = API_CONFIG["model"]
    temperature: float = API_CONFIG["temperature"]
    max_tokens: int = API_CONFIG["max_tokens"]
    timeout: int = API_CONFIG["timeout"]

# Validar API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")

def split_text(text: str, max_tokens: int) -> list[str]:
    """Divide el texto en partes más pequeñas si es necesario."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Estimación aproximada de tokens (1 token ≈ 4 caracteres)
        word_length = len(word) // 4
        if current_length + word_length <= max_tokens:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

async def translate_text(
    text: str,
    target_language: str = "es",
    config: Optional[TranslationConfig] = None
) -> str:
    """
    Traduce un texto al idioma deseado usando la API de OpenAI.

    :param text: Texto a traducir.
    :param target_language: Idioma destino (por defecto: español).
    :param config: Configuración personalizada para la traducción.
    :return: Texto traducido.
    :raises ValueError: Si hay un error en la traducción.
    """
    try:
        # Validar configuración
        config = config or TranslationConfig()

        logger.info(
            f"Iniciando traducción - Target: {target_language}, Text length: {len(text)}"
        )

        # Manejar texto largo dividiéndolo en partes
        chunks = split_text(text, config.max_tokens)
        if len(chunks) > 1:
            logger.info(f"Texto dividido en {len(chunks)} partes")

        translated_chunks = []
        for chunk in chunks:
            # Construir prompt
            prompt = (
                f"Traduce el siguiente texto al idioma '{target_language}' manteniendo el contexto y estilo:\n\n{chunk}"
            )

            logger.info(f"Traduciendo chunk de {len(chunk)} caracteres...")

            # Llamar a la API con timeout
            try:
                response = await asyncio.wait_for(
                    openai.ChatCompletion.acreate(
                        model=config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens
                    ),
                    timeout=config.timeout
                )

                translated_text = response.choices[0].message.content.strip()
                translated_chunks.append(translated_text)
                logger.info(f"Chunk traducido exitosamente")

            except asyncio.TimeoutError:
                logger.error("Timeout al traducir chunk")
                raise TimeoutError(f"Timeout al traducir texto después de {config.timeout} segundos")

        # Unir todas las partes traducidas
        final_translation = " ".join(translated_chunks)
        logger.info(f"Traducción completada - Total caracteres: {len(final_translation)}")

        return final_translation

    except openai.error.AuthenticationError:
        logger.error("Error de autenticación con OpenAI")
        raise ValueError("Error de autenticación con OpenAI")
    except openai.error.RateLimitError:
        logger.error("Límite de velocidad alcanzado")
        raise ValueError("Límite de velocidad alcanzado")
    except openai.error.APIError as e:
        logger.error(f"Error de API: {str(e)}")
        raise ValueError(f"Error de API: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado durante la traducción: {str(e)}", exc_info=True)
        raise ValueError(f"Error durante la traducción: {str(e)}")
