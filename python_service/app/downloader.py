import os
import subprocess
import uuid
import logging
import asyncio
from typing import Optional, Tuple
import shutil
import time

logger = logging.getLogger(__name__)

async def download_audio(youtube_url: str, output_path: str = "downloads", max_duration: int = 7200) -> str:
    """
    Descarga audio de YouTube usando yt-dlp y lo guarda como MP3.
    Optimizado para archivos grandes con límite de duración y manejo eficiente de memoria.
    
    :param youtube_url: URL del video de YouTube
    :param output_path: Directorio donde se guardará el archivo
    :param max_duration: Duración máxima en segundos (por defecto 2 horas)
    :return: Ruta al archivo MP3 descargado
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Verificar espacio en disco antes de descargar
        free_space = shutil.disk_usage(output_path).free
        required_space = 100 * 1024 * 1024  # Estimamos 100MB mínimo
        
        if free_space < required_space:
            raise RuntimeError(f"Espacio insuficiente en disco: {free_space / (1024*1024):.2f}MB disponible, se requiere al menos 100MB")

        # Primero obtenemos información del video para verificar duración
        info_command = [
            "yt-dlp",
            "--print", "duration",
            "--no-download",
            youtube_url
        ]
        
        logger.info(f"Obteniendo información del video: {youtube_url}")
        duration_process = await asyncio.create_subprocess_exec(
            *info_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await duration_process.communicate()
        
        if duration_process.returncode != 0:
            logger.error(f"Error al obtener información: {stderr.decode()}")
            raise RuntimeError(f"No se pudo obtener información del video: {stderr.decode()}")
            
        try:
            duration = int(stdout.decode().strip())
            logger.info(f"Duración del video: {duration} segundos")
            
            if duration > max_duration:
                raise RuntimeError(f"El video es demasiado largo: {duration} segundos (máximo permitido: {max_duration} segundos)")
                
        except ValueError:
            logger.warning("No se pudo determinar la duración, continuando con precaución")

        # Usamos un nombre temporal único
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(output_path, filename)

        # Construimos el comando yt-dlp con optimizaciones
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "128K",  # Calidad media para reducir tamaño
            "--no-playlist",            # Evitar descargar playlists completas
            "--no-cache-dir",          # No usar caché
            "--no-part",               # No crear archivos temporales .part
            "-o", filepath,
            youtube_url
        ]

        logger.info(f"⏬ Ejecutando descarga: {' '.join(command)}")
        start_time = time.time()
        
        # Ejecutar como proceso asíncrono
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error en descarga: {stderr.decode()}")
            raise RuntimeError(f"❌ yt-dlp falló: {stderr.decode()}")
            
        elapsed = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Tamaño en MB
        
        logger.info(f"✅ Descarga completada en {elapsed:.2f}s - Tamaño: {file_size:.2f}MB - Ruta: {filepath}")
        return filepath

    except subprocess.CalledProcessError as e:
        logger.error(f"Error en proceso: {str(e)}")
        raise RuntimeError(f"❌ yt-dlp falló: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise RuntimeError(f"❌ Error inesperado al descargar audio: {e}")
