import os
import whisper
import mimetypes
from pathlib import Path
from typing import Optional
import logging
from app.config import settings

# Configurar logging
logger = logging.getLogger(__name__)

# Lista de formatos de audio soportados
SUPPORTED_AUDIO_FORMATS = {
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.m4a': 'audio/mp4',
    '.ogg': 'audio/ogg',
    '.flac': 'audio/flac'
}

# Lista de modelos Whisper disponibles
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

def validate_audio_file(file_path: str) -> None:
    """Valida que el archivo sea un archivo de audio válido."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"{file_path} no es un archivo válido")
    
    if not any(path.suffix.lower() in SUPPORTED_AUDIO_FORMATS):
        raise ValueError(
            f"Formato de archivo no soportado. Formatos soportados: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}"
        )
    
    # Verificar el mimetype del archivo
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type not in SUPPORTED_AUDIO_FORMATS.values():
        raise ValueError(f"Tipo de archivo no soportado: {mime_type}")

def validate_model(model: str) -> str:
    """Valida que el modelo Whisper sea válido."""
    model = model.lower()
    if model not in WHISPER_MODELS:
        raise ValueError(
            f"Modelo no soportado: {model}. Modelos disponibles: {', '.join(WHISPER_MODELS)}"
        )
    return model

async def transcribe_audio(
    file_path: str,
    language: Optional[str] = None,
    model: str = "large",
    fp16: bool = False,
    sample_rate: int = 16000
) -> str:
    """
    Transcribe un archivo de audio usando Whisper con parámetros configurables.
    Especialmente optimizado para idiomas con caracteres especiales como Yorùbá.

    :param file_path: Ruta al archivo de audio (ej: .mp3)
    :param language: Código del idioma original del audio (ej: 'yo', 'es', 'en')
    :param model: Modelo Whisper a usar ('tiny', 'base', 'small', 'medium', 'large')
    :param fp16: True para usar precisión FP16 (requiere GPU). False para CPU (por defecto).
    :param sample_rate: Tasa de muestreo para el audio (por defecto 16000)
    :return: Texto transcrito manteniendo caracteres especiales
    """
    try:
        # Validar archivo y modelo
        validate_audio_file(file_path)
        model = validate_model(model)

        logger.info(
            f"Transcripción iniciada - File: {file_path}, Model: {model}, Language: {language}, FP16: {fp16}"
        )

        # Configurar opciones para mejor manejo de caracteres especiales
        options = {
            "fp16": fp16,
            "language": language,
            "sample_rate": sample_rate,
            "verbose": False,  # Evitar logs innecesarios
            "word_timestamps": True,  # Para mejor precisión
            "beam_size": 5,  # Para mejor precisión con caracteres especiales
            "best_of": 5,    # Para mejor precisión con caracteres especiales
            "temperature": 0.0  # Para resultados más consistentes
        }

        # Cargar modelo
        logger.info(f"Cargando modelo Whisper: {model}")
        whisper_model = whisper.load_model(model)

        # Transcribir con manejo especial de caracteres
        logger.info(f"Transcribiendo audio... (language={language}, fp16={fp16})")
        result = whisper_model.transcribe(
            file_path,
            **options
        )

        # Procesar el texto para mantener caracteres especiales
        text = result["text"]
        logger.info(f"Transcripción completada. Primera parte: {text[:200]}")

        # Asegurar que el texto esté en UTF-8
        text = text.encode('utf-8').decode('utf-8')

        return text

    except FileNotFoundError as e:
        logger.error(f"Error de archivo: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado durante la transcripción: {str(e)}", exc_info=True)
        raise ValueError(f"Error durante la transcripción: {str(e)}")
