#!/usr/bin/env python3
"""
Script para ejecutar todas las pruebas y generar un informe de cobertura.
"""

import unittest
import coverage
import sys
import os
import subprocess

def run_tests_with_coverage():
    """Ejecuta todas las pruebas con cobertura y genera un informe."""
    # Configurar la cobertura
    cov = coverage.Coverage(
        source=["app"],
        omit=["*/__pycache__/*", "*/tests/*"],
        branch=True
    )
    
    # Iniciar la medición de cobertura
    cov.start()
    
    # Descubrir y ejecutar todas las pruebas
    loader = unittest.TestLoader()
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(tests_dir)
    
    # Ejecutar las pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Detener la medición de cobertura
    cov.stop()
    
    # Guardar los resultados
    cov.save()
    
    # Generar informe en consola
    print("\n\n===== Informe de Cobertura =====")
    cov.report()
    
    # Generar informe HTML
    html_dir = os.path.join(os.path.dirname(__file__), "coverage_html")
    cov.html_report(directory=html_dir)
    
    print(f"\nInforme HTML generado en: {html_dir}")
    
    # Devolver el resultado para determinar el código de salida
    return result

def install_dependencies():
    """Instala las dependencias necesarias para las pruebas."""
    dependencies = ["coverage", "pytest", "pytest-asyncio", "httpx", "pytest-cov"]
    
    print("Instalando dependencias para pruebas...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("Dependencias instaladas correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar dependencias: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Verificar si se debe instalar dependencias
    if "--install-deps" in sys.argv:
        install_dependencies()
    
    # Ejecutar pruebas con cobertura
    result = run_tests_with_coverage()
    
    # Salir con código de error si alguna prueba falló
    if not result.wasSuccessful():
        sys.exit(1)
