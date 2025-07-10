from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from app.downloader import download_audio
from app.transcriber import transcribe_audio
from app.translator import translate_text
from typing import Optional, Dict, Any, Union
import asyncio
import tempfile
import os
import logging
import time
import gc
import json
import uuid
import shutil
from contextlib import asynccontextmanager
from worker import enqueue_task, get_job_status

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("whisper_service.log")
    ]
)
logger = logging.getLogger("whisper_service")

# Límites de memoria y configuración
MAX_CONCURRENT_JOBS = 2
MAX_FILE_SIZE_MB = 500
MAX_AUDIO_DURATION_SEC = 7200  # 2 horas

# Semáforo para limitar trabajos concurrentes
concurrent_jobs_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Gestor de contexto para limpiar archivos temporales
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código de inicialización
    temp_dir = tempfile.gettempdir()
    downloads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "downloads")
    
    # Asegurar que existe el directorio de descargas
    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)
    
    logger.info(f"🚀 Servicio iniciado - Directorio temporal: {temp_dir}")
    logger.info(f"📂 Directorio de descargas: {downloads_dir}")
    
    yield  # Aquí se ejecuta la aplicación
    
    # Código de limpieza al apagar
    try:
        # Limpiar archivos temporales
        for filename in os.listdir(downloads_dir):
            if filename.endswith(".mp3"):
                file_path = os.path.join(downloads_dir, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"🧹 Eliminado archivo temporal: {file_path}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error durante la limpieza: {e}")
    
    logger.info("👋 Servicio finalizado")

app = FastAPI(lifespan=lifespan)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscribeRequest(BaseModel):
    url: str = Field(..., description="URL del video de YouTube")
    language: str = Field("en", description="Código de idioma original del audio (ej: 'en', 'es', 'yo')")
    translate: bool = Field(True, description="Si se debe traducir el texto transcrito")
    model: str = Field("large-v3", description="Modelo Whisper a usar ('tiny', 'base', 'small', 'medium', 'large-v3')")
    fp16: bool = Field(False, description="Usar precisión FP16 (requiere GPU)")
    
    @validator('model')
    def validate_model(cls, v):
        valid_models = ['tiny', 'base', 'small', 'medium', 'large-v3']
        if v not in valid_models:
            raise ValueError(f"Modelo inválido. Debe ser uno de: {', '.join(valid_models)}")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        # Lista no exhaustiva de idiomas soportados
        valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'yo']
        if v not in valid_languages:
            logger.warning(f"Idioma potencialmente no soportado: {v}")
        return v

class TranscriptionResponse(BaseModel):
    transcription: str
    translation: Optional[str] = None
    language_detected: Optional[str] = None
    duration_seconds: Optional[float] = None
    processing_time: float

async def cleanup_resources(file_path: str, delay: int = 60):
    """Elimina recursos después de un retraso"""
    try:
        await asyncio.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🧹 Archivo temporal eliminado: {file_path}")
        gc.collect()  # Forzar recolección de basura
    except Exception as e:
        logger.error(f"Error al limpiar recursos: {e}")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def create_transcription_task(request: TranscribeRequest):
    """Crea una nueva tarea de transcripción"""
    try:
        task_id = enqueue_task(
            "app.tasks.process_transcription",
            request.url,
            request.language,
            request.translate,
            request.model,
            request.fp16
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Obtiene el estado de una tarea"""
    try:
        return get_job_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

@app.get("/health")
async def health_check():
    """Endpoint de salud del servicio"""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/memory-usage")
async def memory_usage():
    """Endpoint para monitorear el uso de memoria"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "concurrent_jobs": MAX_CONCURRENT_JOBS - concurrent_jobs_semaphore._value
        }
    except ImportError:
        return {"error": "psutil no instalado", "concurrent_jobs": MAX_CONCURRENT_JOBS - concurrent_jobs_semaphore._value}
