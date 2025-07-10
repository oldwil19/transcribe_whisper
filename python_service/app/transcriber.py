import os
import logging
from faster_whisper import WhisperModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

def transcribe_audio(
    file_path: str,
    language: str = None,
    model: str = "large-v3",
    fp16: bool = False
) -> str:
    """
    Transcribe un archivo de audio usando Faster-Whisper con parámetros configurables.

    :param file_path: Ruta al archivo de audio (ej: .mp3)
    :param language: Código del idioma original del audio (ej: 'yo', 'es', 'en')
    :param model: Modelo Whisper a usar ('tiny', 'base', 'small', 'medium', 'large-v3')
    :param fp16: True para usar precisión FP16 (requiere GPU). False para CPU (por defecto).
    :return: Texto transcrito con fidelidad a los caracteres especiales
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    # Configuración de dispositivo y precisión
    device = "cuda" if fp16 else "cpu"
    compute_type = "float16" if fp16 else "float32"
    
    logger.info(f"🧠 Cargando modelo Faster-Whisper: {model} en {device} con {compute_type}")
    whisper_model = WhisperModel(model, device=device, compute_type=compute_type)

    # Configuración de opciones para transcripción
    beam_size = 5
    options: Dict[str, Any] = {
        "beam_size": beam_size,
        "best_of": beam_size,
        "patience": 1.0,
        "length_penalty": 1.0,
        "compression_ratio_threshold": 1.35,
        "condition_on_previous_text": True,
        "vad_filter": True,  # Filtro de detección de voz para mejorar resultados
        "vad_parameters": {"threshold": 0.5}
    }
    
    if language:
        options["language"] = language

    logger.info(f"🎙️ Transcribiendo... (language={language}, fp16={fp16})")
    
    # Transcripción con segmentos
    segments, info = whisper_model.transcribe(
        file_path, 
        task="transcribe", 
        **options
    )
    
    # Unir todos los segmentos en un solo texto
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    
    # Log de información y estadísticas
    logger.info(f"✅ Transcripción completada: {len(transcription)} caracteres")
    logger.info(f"📊 Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2f})")
    logger.debug(f"📝 Transcripción parcial: {transcription[:200]}")

    return transcription.strip()
