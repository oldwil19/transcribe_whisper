# translator.py

import openai
import os

# Asegúrate de tener la API Key en una variable de entorno llamada OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_text(text: str, target_language: str = "es") -> str:
    """
    Traduce un texto al idioma deseado usando la API de OpenAI.

    :param text: Texto a traducir.
    :param target_language: Idioma destino (por defecto: español).
    :return: Texto traducido.
    """
    prompt = (
        f"Traduce el siguiente texto al idioma '{target_language}' manteniendo el contexto y estilo:\n\n{text}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
