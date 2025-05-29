import os
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Configuración general
    DEBUG: bool = False
    
    # Configuración de descarga
    MAX_FILE_SIZE_MB: int = 100  # 100MB por defecto
    AUDIO_FORMAT: str = "mp3"
    AUDIO_QUALITY: str = "0"  # Mejor calidad
    TIMEOUT_SECONDS: int = 300  # 5 minutos
    OUTPUT_PATH: str = "downloads"
    
    # Configuración de Whisper
    WHISPER_MODEL: str = "large"
    WHISPER_FP16: bool = False
    
    # Configuración de OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.3
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
