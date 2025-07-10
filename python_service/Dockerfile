# Usa una imagen base ligera
FROM python:3.9-slim

# 1. Primero instalar las dependencias necesarias incluyendo curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Luego descargar y configurar yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp

# Crea el directorio de trabajo
WORKDIR /app

# Copia requirements.txt e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación
COPY . .

# Comandos para iniciar los servicios
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python -m app.worker; wait"]
