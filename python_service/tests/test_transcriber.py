import unittest
import os
import tempfile
import asyncio
import logging
from unittest.mock import patch, MagicMock
from app.transcriber import transcribe_audio

# Configurar logging para pruebas
logging.basicConfig(level=logging.ERROR)

class TestTranscriber(unittest.TestCase):
    """Pruebas unitarias para el módulo transcriber"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Crear un archivo de audio temporal para pruebas
        self.temp_dir = tempfile.mkdtemp()
        self.temp_audio = os.path.join(self.temp_dir, "test_audio.mp3")
        
        # Crear un archivo vacío
        with open(self.temp_audio, "wb") as f:
            f.write(b"dummy audio content")
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Eliminar archivos temporales
        if os.path.exists(self.temp_audio):
            os.remove(self.temp_audio)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_file_not_found(self):
        """Prueba que se lance FileNotFoundError cuando el archivo no existe"""
        # Ejecutar la prueba de forma asíncrona
        async def run_test():
            with self.assertRaises(FileNotFoundError):
                await transcribe_audio("archivo_inexistente.mp3")
        
        # Ejecutar la corrutina
        asyncio.run(run_test())
    
    @patch('app.transcriber.WhisperModel')
    def test_transcribe_audio_success(self, mock_whisper_model):
        """Prueba que la transcripción funcione correctamente"""
        # Configurar el mock
        mock_instance = MagicMock()
        mock_whisper_model.return_value = mock_instance
        
        # Configurar el resultado de la transcripción
        mock_segment = MagicMock()
        mock_segment.text = "Este es un texto de prueba"
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.98
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)
        
        # Ejecutar la prueba de forma asíncrona
        async def run_test():
            result = await transcribe_audio(self.temp_audio, language="es")
            self.assertEqual(result, "Este es un texto de prueba")
            mock_whisper_model.assert_called_once()
            mock_instance.transcribe.assert_called_once()
        
        # Ejecutar la corrutina
        asyncio.run(run_test())
    
    @patch('app.transcriber.WhisperModel')
    def test_transcribe_audio_multiple_segments(self, mock_whisper_model):
        """Prueba que la transcripción funcione con múltiples segmentos"""
        # Configurar el mock
        mock_instance = MagicMock()
        mock_whisper_model.return_value = mock_instance
        
        # Configurar múltiples segmentos
        mock_segment1 = MagicMock()
        mock_segment1.text = "Primer segmento."
        mock_segment2 = MagicMock()
        mock_segment2.text = "Segundo segmento."
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.98
        mock_instance.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        
        # Ejecutar la prueba de forma asíncrona
        async def run_test():
            result = await transcribe_audio(self.temp_audio)
            self.assertEqual(result, "Primer segmento. Segundo segmento.")
        
        # Ejecutar la corrutina
        asyncio.run(run_test())
    
    @patch('app.transcriber.WhisperModel')
    def test_transcribe_audio_with_fp16(self, mock_whisper_model):
        """Prueba que la transcripción funcione con fp16 activado"""
        # Configurar el mock
        mock_instance = MagicMock()
        mock_whisper_model.return_value = mock_instance
        
        # Configurar el resultado
        mock_segment = MagicMock()
        mock_segment.text = "Texto de prueba con GPU"
        mock_info = MagicMock()
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)
        
        # Ejecutar la prueba de forma asíncrona
        async def run_test():
            result = await transcribe_audio(self.temp_audio, fp16=True)
            self.assertEqual(result, "Texto de prueba con GPU")
            # Verificar que se llamó con los parámetros correctos
            mock_whisper_model.assert_called_once_with(
                "large-v3", device="cuda", compute_type="float16"
            )
        
        # Ejecutar la corrutina
        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()