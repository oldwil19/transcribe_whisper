from abc import ABC, abstractmethod
from typing import Optional

class IAudioDownloader(ABC):
    """Interfaz para descargadores de audio."""
    @abstractmethod
    async def validate_url(self, url: str) -> None:
        """Valida que la URL sea válida."""
        pass

    @abstractmethod
    async def download(self, url: str, config: Optional[dict] = None) -> str:
        """Descarga el audio."""
        pass

class ITranslator(ABC):
    """Interfaz para traductores."""
    @abstractmethod
    async def translate(self, text: str, target_language: str) -> str:
        """Traduce el texto."""
        pass

class ITranscriber(ABC):
    """Interfaz para transcritores."""
    @abstractmethod
    async def transcribe(self, file_path: str, config: Optional[dict] = None) -> str:
        """Transcribe el audio."""
        pass
