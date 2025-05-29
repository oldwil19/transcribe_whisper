from pathlib import Path
import uuid
import asyncio
from datetime import datetime
from typing import Optional
import logging
from abc import ABC, abstractmethod
from app.config import settings

# Configurar logging
logger = logging.getLogger(__name__)

class BaseDownloader(ABC):
    """Clase base abstracta para descargadores."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.output_path = Path(self.config.get("output_path", settings.OUTPUT_PATH))
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def validate_url(self, url: str) -> None:
        """Valida la URL."""
        pass

    @abstractmethod
    async def download(self, url: str) -> str:
        """Descarga el recurso."""
        pass

    async def generate_filename(self) -> str:
        """Genera un nombre de archivo único."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"audio_{timestamp}_{uuid.uuid4()}"

    async def cleanup(self, filepath: Path) -> None:
        """Limpia archivos incompletos."""
        if filepath.exists():
            filepath.unlink()

class YouTubeDownloader(BaseDownloader):
    """Implementación específica para YouTube."""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.max_file_size_mb = config.get("max_file_size_mb", settings.MAX_FILE_SIZE_MB)
        self.audio_format = config.get("audio_format", settings.AUDIO_FORMAT)
        self.audio_quality = config.get("audio_quality", settings.AUDIO_QUALITY)
        self.timeout_seconds = config.get("timeout_seconds", settings.TIMEOUT_SECONDS)

    async def validate_url(self, url: str) -> None:
        """Valida que sea una URL de YouTube válida."""
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("URL debe comenzar con http:// o https://")
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValueError("URL no es una URL de YouTube válida")

    async def check_file_size(self, filepath: Path) -> None:
        """Verifica el tamaño del archivo."""
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"Archivo demasiado grande ({size_mb:.2f}MB). Tamaño máximo permitido: {self.max_file_size_mb}MB"
            )

    async def download(self, url: str) -> str:
        """Descarga audio de YouTube usando yt-dlp."""
        try:
            # Validar URL
            await self.validate_url(url)

            # Generar nombre de archivo
            filename = await self.generate_filename()
            filepath = self.output_path / f"{filename}.{self.audio_format}"

            logger.info(f"Iniciando descarga - URL: {url}")
            logger.info(f"Configuración: {self.config}")

            # Construir comando yt-dlp
            command = [
                "yt-dlp",
                "-x",  # Extraer audio
                f"--audio-format={self.audio_format}",
                f"--audio-quality={self.audio_quality}",
                f"--max-filesize={self.max_file_size_mb}M",
                "-o", str(filepath),
                url
            ]

            logger.info(f"Ejecutando comando: {' '.join(command)}")

            # Ejecutar comando con timeout
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Esperar con timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    process.terminate()
                    raise TimeoutError(
                        f"Timeout al descargar audio después de {self.timeout_seconds} segundos"
                    )

                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Error desconocido"
                    raise RuntimeError(f"Error al descargar audio: {error_msg}")

                # Verificar tamaño del archivo
                await self.check_file_size(filepath)

                logger.info(f"Descarga completada - Tamaño: {filepath.stat().st_size/1024/1024:.2f}MB")
                return str(filepath)

            except Exception as e:
                await self.cleanup(filepath)
                raise RuntimeError(f"Error al ejecutar yt-dlp: {str(e)}")

        except ValueError as e:
            logger.error(f"Error de validación: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado durante la descarga: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error al descargar audio: {str(e)}")
