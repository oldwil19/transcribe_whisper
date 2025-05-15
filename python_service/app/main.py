from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.downloader import download_audio
from app.transcriber import transcribe_audio
from app.translator import translate_text
from typing import Optional

app = FastAPI()

class TranscribeRequest(BaseModel):
    url: str                          # URL del video de YouTube
    language: str = "en"              # Idioma original del audio
    translate: bool = True            # Si se debe traducir o no
    model: Optional[str] = "large"    # Modelo Whisper a usar
    fp16: Optional[bool] = False      # Modo FP16 (GPU). False si CPU

@app.post("/transcribe")
def transcribe_and_translate(req: TranscribeRequest):
    try:
        print(f"📥 Recibida URL: {req.url}")
        print(f"🌍 Idioma: {req.language}")
        print(f"🧠 Modelo: {req.model}")
        print(f"⚙️  Translate: {req.translate}, fp16: {req.fp16}")

        # Paso 1: Descargar el audio del video de YouTube
        audio_path = download_audio(req.url)

        # Paso 2: Transcribir el audio usando Whisper
        transcription = transcribe_audio(
            file_path=audio_path,
            language=req.language,
            model=req.model,
            fp16=req.fp16
        )

        print("📝 Transcripción parcial:", transcription[:200])

        result = {
            "transcription": transcription
        }

        # Paso 3: Traducir si se solicita
        if req.translate:
            translation = translate_text(transcription, target_language=req.language)
            result["translation"] = translation

        # Retornar respuesta codificada en UTF-8 para preservar caracteres
        return JSONResponse(content=result, media_type="application/json; charset=utf-8")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
