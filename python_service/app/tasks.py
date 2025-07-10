import os
import time
import logging
import gc
from .downloader import download_audio
from .transcriber import transcribe_audio
from .translator import translate_text

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de límites
MAX_AUDIO_DURATION_SEC = 2 * 60 * 60  # 2 horas
MAX_FILE_SIZE_MB = 500

def process_transcription(youtube_url: str, language: str, translate: bool, model: str, fp16: bool):
    """Procesa una tarea de transcripción completa"""
    job_id = str(int(time.time()))[-8:]
    start_time = time.time()
    audio_path = None
    
    try:
        logger.info(f"[{job_id}] 🔄 Iniciando solicitud - URL: {youtube_url}, Modelo: {model}, Idioma: {language}")
        
        # Paso 1: Descargar audio
        download_start = time.time()
        logger.info(f"[{job_id}] ⏬ Descargando audio de YouTube")
        
        audio_path = download_audio(
            youtube_url=youtube_url, 
            max_duration=MAX_AUDIO_DURATION_SEC
        )
        
        # Verificar tamaño del archivo
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"El archivo de audio es demasiado grande: {file_size_mb:.2f}MB (máximo: {MAX_FILE_SIZE_MB}MB)")
            
        download_time = time.time() - download_start
        logger.info(f"[{job_id}] ✅ Audio descargado en {download_time:.2f}s - Tamaño: {file_size_mb:.2f}MB")
        
        # Paso 2: Transcribir audio
        transcribe_start = time.time()
        logger.info(f"[{job_id}] 🎯 Iniciando transcripción con modelo {model}")
        
        transcription = transcribe_audio(
            file_path=audio_path,
            language=language,
            model=model,
            fp16=fp16
        )
        
        transcribe_time = time.time() - transcribe_start
        logger.info(f"[{job_id}] ✅ Transcripción completada en {transcribe_time:.2f}s - {len(transcription)} caracteres")
        
        # Paso 3: Traducir si es necesario
        translation = None
        if translate and transcription:
            translate_start = time.time()
            logger.info(f"[{job_id}] 🌐 Iniciando traducción")
            
            try:
                translation = translate_text(
                    text=transcription,
                    target_language="es"  # Por defecto al español
                )
                translate_time = time.time() - translate_start
                logger.info(f"[{job_id}] ✅ Traducción completada en {translate_time:.2f}s")
            except Exception as e:
                logger.error(f"[{job_id}] ⚠️ Error en traducción: {e}")
                translation = None
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        
        # Limpiar recursos
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"[{job_id}] 🧹 Archivo temporal eliminado: {audio_path}")
            except Exception as e:
                logger.error(f"[{job_id}] ❌ Error al limpiar archivo: {e}")
        
        # Forzar recolección de basura para liberar memoria
        gc.collect()
        
        logger.info(f"[{job_id}] 🏁 Proceso completado en {total_time:.2f}s")
        
        # Retornar resultados
        return {
            "transcription": transcription,
            "translation": translation,
            "language_detected": language,
            "processing_time": total_time
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"[{job_id}] ❌ Error después de {elapsed_time:.2f}s: {str(e)}", exc_info=True)
        
        # Limpiar recursos en caso de error
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"[{job_id}] 🧹 Archivo temporal eliminado tras error: {audio_path}")
            except:
                pass
        
        # Forzar recolección de basura
        gc.collect()
        raise
