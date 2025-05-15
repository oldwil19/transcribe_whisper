# Usa una imagen base ligera
FROM python:3.10-slim

# Instala ffmpeg, curl y yt-dlp
RUN apt-get update && \
    apt-get install -y ffmpeg git curl && \
    curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && \
    chmod a+rx /usr/local/bin/yt-dlp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Crea el directorio de trabajo
WORKDIR /app

# Copia requirements.txt e instala dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia el código de la aplicación
COPY app/ app/

# Expone el puerto para FastAPI
EXPOSE 8000

# Comando por defecto para correr el servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
