import unittest
import os
import time
from unittest.mock import patch, MagicMock
from app.translator import translate_text

class TestTranslator(unittest.TestCase):
    """Pruebas unitarias para el módulo translator"""
    
    def setUp(self):
        """Configuración inicial para cada prueba"""
        # Guardar valor original de la variable de entorno
        self.original_api_key = os.environ.get("OPENAI_API_KEY")
    
    def tearDown(self):
        """Limpieza después de cada prueba"""
        # Restaurar la variable de entorno a su valor original
        if self.original_api_key:
            os.environ["OPENAI_API_KEY"] = self.original_api_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    @patch('app.translator.client')
    def test_translate_text_success(self, mock_client):
        """Prueba que la traducción funcione correctamente"""
        # Configurar el mock
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Texto traducido"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_response
        mock_client.chat = mock_chat
        
        # Ejecutar la función
        result = translate_text("Text to translate", "es")
        
        # Verificar el resultado
        self.assertEqual(result, "Texto traducido")
        mock_chat.completions.create.assert_called_once()
    
    @patch('app.translator.client')
    def test_translate_text_with_long_text(self, mock_client):
        """Prueba que se trunque el texto si es demasiado largo"""
        # Configurar el mock
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Texto muy largo traducido"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_chat = MagicMock()
        mock_chat.completions.create.return_value = mock_response
        mock_client.chat = mock_chat
        
        # Crear un texto muy largo
        long_text = "a" * 20000
        
        # Ejecutar la función
        result = translate_text(long_text, "es")
        
        # Verificar el resultado
        self.assertEqual(result, "Texto muy largo traducido")
        
        # Verificar que se llamó con un texto truncado
        args, kwargs = mock_chat.completions.create.call_args
        self.assertIn("messages", kwargs)
        self.assertTrue(len(kwargs["messages"][0]["content"]) < 20000)
        self.assertIn("...", kwargs["messages"][0]["content"])
    
    @patch('app.translator.client')
    @patch('app.translator.time.sleep')
    def test_translate_text_retry_on_error(self, mock_sleep, mock_client):
        """Prueba que se reintente la traducción en caso de error"""
        # Configurar el mock para que falle la primera vez y luego tenga éxito
        mock_chat = MagicMock()
        mock_client.chat = mock_chat
        
        # Primera llamada: error
        mock_chat.completions.create.side_effect = [
            Exception("Error de API"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Texto traducido tras reintento"))])
        ]
        
        # Ejecutar la función
        result = translate_text("Text to translate", "es")
        
        # Verificar el resultado
        self.assertEqual(result, "Texto traducido tras reintento")
        self.assertEqual(mock_chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once()
    
    @patch('app.translator.client', None)
    def test_translate_text_no_api_key(self):
        """Prueba que se maneje correctamente la falta de API key"""
        result = translate_text("Text to translate", "es")
        self.assertIsNone(result)
    
    @patch('app.translator.client')
    @patch('app.translator.time.sleep')
    def test_translate_text_all_retries_fail(self, mock_sleep, mock_client):
        """Prueba que se maneje correctamente cuando fallan todos los reintentos"""
        # Configurar el mock para que siempre falle
        mock_chat = MagicMock()
        mock_client.chat = mock_chat
        
        # Todas las llamadas fallan
        mock_chat.completions.create.side_effect = Exception("Error persistente")
        
        # Ejecutar la función
        result = translate_text("Text to translate", "es", max_retries=1)
        
        # Verificar el resultado
        self.assertIsNone(result)
        self.assertEqual(mock_chat.completions.create.call_count, 4)  # 2 modelos x 2 intentos cada uno
    
    @patch('app.translator.client')
    def test_translate_text_fallback_model(self, mock_client):
        """Prueba que se use el modelo de respaldo si el principal falla"""
        # Configurar el mock
        mock_chat = MagicMock()
        mock_client.chat = mock_chat
        
        # El modelo principal falla, el de respaldo funciona
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Texto traducido con modelo de respaldo"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_chat.completions.create.side_effect = [
            Exception("Error con modelo principal"),
            Exception("Error con modelo principal reintento"),
            Exception("Error con modelo principal último intento"),
            mock_response
        ]
        
        # Ejecutar la función
        result = translate_text("Text to translate", "es")
        
        # Verificar el resultado
        self.assertEqual(result, "Texto traducido con modelo de respaldo")
        self.assertEqual(mock_chat.completions.create.call_count, 4)

if __name__ == "__main__":
    unittest.main()
