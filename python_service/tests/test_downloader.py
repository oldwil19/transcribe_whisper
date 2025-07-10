import unittest
import os
import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock
from app.downloader import download_audio

class TestDownloader(unittest.TestCase):
    """Pruebas unitarias para el módulo downloader"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Crear un directorio temporal para las descargas
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Eliminar directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('app.downloader.asyncio.create_subprocess_exec')
    @patch('app.downloader.shutil.disk_usage')
    async def test_download_audio_success(self, mock_disk_usage, mock_subprocess):
        """Prueba que la descarga funcione correctamente"""
        # Configurar mock para espacio en disco
        disk_usage_result = MagicMock()
        disk_usage_result.free = 1000 * 1024 * 1024  # 1GB libre
        mock_disk_usage.return_value = disk_usage_result
        
        # Configurar mock para el proceso de duración
        duration_process = AsyncMock()
        duration_process.returncode = 0
        duration_process.communicate.return_value = (b"120\n", b"")
        
        # Configurar mock para el proceso de descarga
        download_process = AsyncMock()
        download_process.returncode = 0
        download_process.communicate.return_value = (b"", b"")
        
        # Configurar el comportamiento del mock para create_subprocess_exec
        mock_subprocess.side_effect = [duration_process, download_process]
        
        # Crear un archivo falso para simular la descarga
        expected_path = None
        
        def side_effect(*args, **kwargs):
            nonlocal expected_path
            # Extraer la ruta del archivo de los argumentos
            for i, arg in enumerate(args[0]):
                if arg == "-o" and i + 1 < len(args[0]):
                    expected_path = args[0][i + 1]
                    # Crear un archivo falso
                    with open(expected_path, "wb") as f:
                        f.write(b"fake audio content")
                    break
            return download_process
        
        mock_subprocess.side_effect = [duration_process, side_effect]
        
        # Ejecutar la función
        result = await download_audio("https://www.youtube.com/watch?v=test", self.temp_dir)
        
        # Verificar que se llamó a los procesos correctamente
        self.assertEqual(mock_subprocess.call_count, 2)
        
        # Verificar que se creó un archivo
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".mp3"))
    
    @patch('app.downloader.asyncio.create_subprocess_exec')
    @patch('app.downloader.shutil.disk_usage')
    async def test_download_audio_duration_too_long(self, mock_disk_usage, mock_subprocess):
        """Prueba que se lance una excepción cuando el video es demasiado largo"""
        # Configurar mock para espacio en disco
        disk_usage_result = MagicMock()
        disk_usage_result.free = 1000 * 1024 * 1024  # 1GB libre
        mock_disk_usage.return_value = disk_usage_result
        
        # Configurar mock para el proceso de duración
        duration_process = AsyncMock()
        duration_process.returncode = 0
        duration_process.communicate.return_value = (b"10000\n", b"")  # 10000 segundos (casi 3 horas)
        
        # Configurar el comportamiento del mock
        mock_subprocess.return_value = duration_process
        
        # Verificar que se lance la excepción
        with self.assertRaises(RuntimeError) as context:
            await download_audio("https://www.youtube.com/watch?v=test", self.temp_dir, max_duration=7200)
        
        # Verificar el mensaje de error
        self.assertIn("demasiado largo", str(context.exception))
    
    @patch('app.downloader.asyncio.create_subprocess_exec')
    @patch('app.downloader.shutil.disk_usage')
    async def test_download_audio_insufficient_disk_space(self, mock_disk_usage, mock_subprocess):
        """Prueba que se lance una excepción cuando no hay suficiente espacio en disco"""
        # Configurar mock para espacio en disco (50MB, menos de los 100MB requeridos)
        disk_usage_result = MagicMock()
        disk_usage_result.free = 50 * 1024 * 1024
        mock_disk_usage.return_value = disk_usage_result
        
        # Verificar que se lance la excepción
        with self.assertRaises(RuntimeError) as context:
            await download_audio("https://www.youtube.com/watch?v=test", self.temp_dir)
        
        # Verificar el mensaje de error
        self.assertIn("Espacio insuficiente", str(context.exception))
        
        # Verificar que no se llamó al proceso
        mock_subprocess.assert_not_called()
    
    @patch('app.downloader.asyncio.create_subprocess_exec')
    @patch('app.downloader.shutil.disk_usage')
    async def test_download_audio_yt_dlp_error(self, mock_disk_usage, mock_subprocess):
        """Prueba que se maneje correctamente un error de yt-dlp"""
        # Configurar mock para espacio en disco
        disk_usage_result = MagicMock()
        disk_usage_result.free = 1000 * 1024 * 1024  # 1GB libre
        mock_disk_usage.return_value = disk_usage_result
        
        # Configurar mock para el proceso de duración
        duration_process = AsyncMock()
        duration_process.returncode = 0
        duration_process.communicate.return_value = (b"120\n", b"")
        
        # Configurar mock para el proceso de descarga (con error)
        download_process = AsyncMock()
        download_process.returncode = 1
        download_process.communicate.return_value = (b"", b"Error: video no disponible")
        
        # Configurar el comportamiento del mock
        mock_subprocess.side_effect = [duration_process, download_process]
        
        # Verificar que se lance la excepción
        with self.assertRaises(RuntimeError) as context:
            await download_audio("https://www.youtube.com/watch?v=test", self.temp_dir)
        
        # Verificar el mensaje de error
        self.assertIn("yt-dlp falló", str(context.exception))

def run_async_test(coro):
    return asyncio.run(coro)

if __name__ == "__main__":
    unittest.main()
