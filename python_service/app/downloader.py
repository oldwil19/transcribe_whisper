import os
import subprocess
import uuid

def download_audio(youtube_url: str, output_path: str = "downloads") -> str:
    """
    Descarga audio de YouTube usando yt-dlp y lo guarda como MP3.
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Usamos un nombre temporal único
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(output_path, filename)

        # Construimos el comando yt-dlp
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "-o", filepath,
            youtube_url
        ]

        print(f"⏬ Ejecutando: {' '.join(command)}")
        subprocess.run(command, check=True)

        return filepath

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ yt-dlp falló: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ Error inesperado al descargar audio: {e}")
