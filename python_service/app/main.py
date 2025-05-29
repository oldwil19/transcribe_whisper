from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
from app.downloader import download_audio
from app.transcriber import transcribe_audio
from app.translator import translate_text
from typing import Optional
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Transcriber API", version="1.0.0")

class TranscribeRequest(BaseModel):
    url: HttpUrl                          # URL del video de YouTube
    language: str = "en"                  # Idioma original del audio
    translate: bool = True                # Si se debe traducir o no
    model: Optional[str] = "large"        # Modelo Whisper a usar
    fp16: Optional[bool] = False          # Modo FP16 (GPU). False si CPU

    @validator('language')
    def validate_language(cls, v):
        # Lista de idiomas soportados
        supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "yo"
        ]
        if v.lower() not in supported_languages:
            raise ValueError(f"Language must be one of: {', '.join(supported_languages)}")
        return v

    @validator('model')
    def validate_model(cls, v):
        if v and v.lower() not in ["tiny", "base", "small", "medium", "large"]:
            raise ValueError("Model must be one of: tiny, base, small, medium, large")
        return v

@app.post("/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_and_translate(req: TranscribeRequest):
    try:
        log_data = {
            "url": req.url,
            "language": req.language,
            "translate": req.translate,
            "model": req.model,
            "fp16": req.fp16
        }
        logger.info("Transcription request received", extra={"data": log_data})

        # Paso 1: Descargar el audio del video de YouTube
        logger.info("Downloading audio from YouTube...")
        audio_path = await download_audio(req.url)

        # Paso 2: Transcribir el audio usando Whisper
        logger.info(f"Transcribing audio with model: {req.model}")
        transcription = await transcribe_audio(
            file_path=audio_path,
            language=req.language,
            model=req.model,
            fp16=req.fp16
        )

        logger.info("Transcription completed")
        result = {
            "transcription": transcription,
            "timestamp": datetime.utcnow().isoformat(),
            "model_used": req.model,
            "language": req.language
        }

        # Paso 3: Traducir si se solicita
        if req.translate:
            logger.info(f"Translating text to: {req.language}")
            translation = await translate_text(transcription, target_language=req.language)
            result["translation"] = translation

        logger.info("Request processed successfully")
        return JSONResponse(
            content=result,
            media_type="application/json; charset=utf-8"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
