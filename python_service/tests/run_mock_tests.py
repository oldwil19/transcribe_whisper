#!/usr/bin/env python3
"""
Script para ejecutar pruebas unitarias con mocks para evitar dependencias externas.
"""

import unittest
import sys
import os
import logging
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Configurar logging para pruebas
logging.basicConfig(level=logging.ERROR)

# Agregar el directorio raíz al path para importar los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crear mocks para las dependencias externas
class MockWhisperModel:
    def __init__(self, *args, **kwargs):
        pass
    
    def transcribe(self, *args, **kwargs):
        mock_segment = MagicMock()
        mock_segment.text = "Texto transcrito de prueba"
        mock_info = MagicMock()
        mock_info.language = "es"
        mock_info.language_probability = 0.98
        return [mock_segment], mock_info

# Crear módulos mock
mock_faster_whisper = MagicMock()
mock_faster_whisper.WhisperModel = MockWhisperModel

mock_openai = MagicMock()
mock_openai.OpenAI = MagicMock()

mock_yt_dlp = MagicMock()

# Aplicar mocks globales
sys.modules['faster_whisper'] = mock_faster_whisper
sys.modules['openai'] = mock_openai
sys.modules['yt_dlp'] = mock_yt_dlp
sys.modules['psutil'] = MagicMock()

# Parchar las funciones asíncronas para pruebas
def async_mock_decorator(func):
    async def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

# Parchar os.path.exists para que devuelva True para archivos de prueba
original_exists = os.path.exists
def mock_exists(path):
    if 'test_audio' in path or 'archivo_inexistente' in path:
        return 'inexistente' not in path
    return original_exists(path)

os.path.exists = mock_exists

# Parchar os.stat para que devuelva un tamaño de archivo simulado
class MockStat:
    def __init__(self, size):
        self.st_size = size

original_stat = os.stat
def mock_stat(path):
    if 'test_audio' in path:
        if 'large' in path:
            return MockStat(501 * 1024 * 1024)  # 501 MB
        return MockStat(10 * 1024 * 1024)  # 10 MB
    return original_stat(path)

os.stat = mock_stat

# Importar las pruebas después de configurar los mocks
with patch('app.transcriber.WhisperModel', MockWhisperModel):
    from tests.test_transcriber import TestTranscriber
    from tests.test_translator import TestTranslator
    from tests.test_downloader import TestDownloader

if __name__ == "__main__":
    # Configurar y ejecutar las pruebas
    test_suite = unittest.TestSuite()
    
    # Agregar pruebas del transcriptor
    loader = unittest.TestLoader()
    test_suite.addTests(loader.loadTestsFromTestCase(TestTranscriber))
    
    # Agregar pruebas del traductor
    test_suite.addTests(loader.loadTestsFromTestCase(TestTranslator))
    
    # Agregar pruebas del descargador
    test_suite.addTests(loader.loadTestsFromTestCase(TestDownloader))
    
    # Ejecutar las pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Salir con código de error si alguna prueba falló
    if not result.wasSuccessful():
        sys.exit(1)
