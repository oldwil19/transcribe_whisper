import unittest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from app.main import app, TranscribeRequest

class TestMainAPI(unittest.TestCase):
    """Pruebas de integración para la API principal"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        self.client = TestClient(app)
    
    @patch('app.main.download_audio')
    @patch('app.main.transcribe_audio')
    @patch('app.main.asyncio.to_thread')
    async def test_transcribe_endpoint_success(self, mock_to_thread, mock_transcribe, mock_download):
        """Prueba que el endpoint /transcribe funcione correctamente"""
        # Configurar mocks
        audio_path = "/tmp/test_audio.mp3"
        mock_download.return_value = audio_path
        mock_transcribe.return_value = "Texto transcrito de prueba"
        mock_to_thread.return_value = "Texto traducido de prueba"
        
        # Crear un archivo temporal para simular el audio
        with open(audio_path, "wb") as f:
            f.write(b"fake audio content")
        
        # Ejecutar la solicitud
        with TestClient(app) as client:
            response = client.post(
                "/transcribe",
                json={
                    "url": "https://www.youtube.com/watch?v=test",
                    "language": "es",
                    "translate": True,
                    "model": "small",
                    "fp16": False
                }
            )
        
        # Verificar la respuesta
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["transcription"], "Texto transcrito de prueba")
        self.assertEqual(data["translation"], "Texto traducido de prueba")
        self.assertEqual(data["language_detected"], "es")
        self.assertIsNotNone(data["processing_time"])
        
        # Verificar que se llamaron las funciones correctas
        mock_download.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_to_thread.assert_called_once()
        
        # Limpiar
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    @patch('app.main.download_audio')
    @patch('app.main.transcribe_audio')
    async def test_transcribe_endpoint_without_translation(self, mock_transcribe, mock_download):
        """Prueba que el endpoint /transcribe funcione sin traducción"""
        # Configurar mocks
        audio_path = "/tmp/test_audio_no_translate.mp3"
        mock_download.return_value = audio_path
        mock_transcribe.return_value = "Texto transcrito sin traducción"
        
        # Crear un archivo temporal para simular el audio
        with open(audio_path, "wb") as f:
            f.write(b"fake audio content")
        
        # Ejecutar la solicitud
        with TestClient(app) as client:
            response = client.post(
                "/transcribe",
                json={
                    "url": "https://www.youtube.com/watch?v=test",
                    "language": "en",
                    "translate": False,
                    "model": "tiny",
                    "fp16": False
                }
            )
        
        # Verificar la respuesta
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["transcription"], "Texto transcrito sin traducción")
        self.assertIsNone(data["translation"])
        
        # Limpiar
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    @patch('app.main.download_audio')
    async def test_transcribe_endpoint_download_error(self, mock_download):
        """Prueba que se manejen correctamente los errores de descarga"""
        # Configurar mock para que lance una excepción
        mock_download.side_effect = RuntimeError("Error al descargar el video")
        
        # Ejecutar la solicitud
        with TestClient(app) as client:
            response = client.post(
                "/transcribe",
                json={
                    "url": "https://www.youtube.com/watch?v=invalid",
                    "language": "en",
                    "translate": True
                }
            )
        
        # Verificar la respuesta de error
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Error al descargar el video", data["detail"])
    
    @patch('app.main.download_audio')
    async def test_transcribe_endpoint_file_too_large(self, mock_download):
        """Prueba que se rechacen archivos demasiado grandes"""
        # Configurar mock
        audio_path = "/tmp/test_audio_large.mp3"
        mock_download.return_value = audio_path
        
        # Crear un archivo grande (simulado)
        with open(audio_path, "wb") as f:
            f.write(b"x" * (501 * 1024 * 1024))  # 501 MB
        
        # Ejecutar la solicitud
        with TestClient(app) as client:
            response = client.post(
                "/transcribe",
                json={
                    "url": "https://www.youtube.com/watch?v=large_video",
                    "language": "en",
                    "translate": True
                }
            )
        
        # Verificar la respuesta de error
        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertIn("demasiado grande", data["detail"])
        
        # Limpiar
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    def test_health_check_endpoint(self):
        """Prueba que el endpoint /health funcione correctamente"""
        with TestClient(app) as client:
            response = client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsNotNone(data["version"])
    
    @patch('app.main.psutil.Process')
    def test_memory_usage_endpoint(self, mock_process):
        """Prueba que el endpoint /memory-usage funcione correctamente"""
        # Configurar mock
        mock_instance = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200 MB
        mock_instance.memory_info.return_value = mock_memory_info
        mock_instance.memory_percent.return_value = 5.5
        mock_process.return_value = mock_instance
        
        # Ejecutar la solicitud
        with TestClient(app) as client:
            response = client.get("/memory-usage")
        
        # Verificar la respuesta
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["rss_mb"], 100.0)
        self.assertEqual(data["vms_mb"], 200.0)
        self.assertEqual(data["percent"], 5.5)
        self.assertIsNotNone(data["concurrent_jobs"])

def run_async_test(coro):
    return asyncio.run(coro)

if __name__ == "__main__":
    unittest.main()
