import os
import whisper

def transcribe_audio(
    file_path: str,
    language: str = None,
    model: str = "large",
    fp16: bool = False
) -> str:
    """
    Transcribe un archivo de audio usando Whisper con parámetros configurables.

    :param file_path: Ruta al archivo de audio (ej: .mp3)
    :param language: Código del idioma original del audio (ej: 'yo', 'es', 'en')
    :param model: Modelo Whisper a usar ('tiny', 'base', 'small', 'medium', 'large')
    :param fp16: True para usar precisión FP16 (requiere GPU). False para CPU (por defecto).
    :return: Texto transcrito con fidelidad a los caracteres especiales
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    print(f"🧠 Cargando modelo Whisper: {model}")
    whisper_model = whisper.load_model(model)

    # Configuración de opciones para transcripción
    options = {"fp16": fp16}
    if language:
        options["language"] = language

    print(f"🎙️ Transcribiendo... (language={language}, fp16={fp16})")
    result = whisper_model.transcribe(file_path, **options)

    # Log parcial para depuración
    print("📝 Transcripción parcial:", result["text"][:200])

    return result["text"]
