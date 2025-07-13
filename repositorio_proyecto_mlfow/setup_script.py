"""
Setup MLflow Environment - Configuración Automática
===================================================

Script para configurar automáticamente el entorno MLflow
en instancias EC2 y sistemas locales.

Autor: Equipo Analítica
Fecha: Julio 2025
"""

import os
import sys
import subprocess
import json
import socket
from datetime import datetime

def check_python_version():
    """Verifica la versión de Python."""
    if sys.version_info < (3, 7):
        print("ERROR: Se requiere Python 3.7 o superior")
        return False
    print(f"✓ Python {sys.version}")
    return True

def install_requirements():
    """Instala paquetes requeridos."""
    requirements = [
        "mlflow>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "boto3>=1.20.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0"
    ]
    
    print("Instalando dependencias...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Error instalando {package}")
            return False
    
    return True

def setup_mlflow_directories():
    """Crea directorios necesarios para MLflow."""
    directories = [
        "mlruns",
        "mlflow_artifacts",
        "logs",
        "reports",
        "models"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Directorio creado: {dir_name}")

def create_mlflow_config():
    """Crea archivo de configuración MLflow."""
    config = {
        "tracking_uri": "http://localhost:5000",
        "artifact_location": "./mlflow_artifacts",
        "default_experiment": "segmentacion_clientes_v2",
        "backend_store_uri": "./mlruns",
        "serve_artifacts": True
    }
    
    with open("mlflow_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuración MLflow creada")

def setup_aws_ec2():
    """Configuración específica para AWS EC2."""
    try:
        import requests
        # Verificar si estamos en EC2
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-id',
            timeout=2
        )
        
        if response.status_code == 200:
            print("✓ Instancia EC2 detectada")
            
            # Configurar puertos
            commands = [
                "sudo ufw allow 5000",
                "sudo ufw allow 22",
                "sudo ufw allow 80"
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd.split(), check=True)
                    print(f"✓ {cmd}")
                except:
                    print(f"⚠ No se pudo ejecutar: {cmd}")
            
            return True
    except:
        print("✓ Entorno local detectado")
        return False

def create_startup_script():
    """Crea script de inicio para MLflow."""
    script_content = """#!/bin/bash
# MLflow Startup Script

echo "Iniciando MLflow Server..."

# Verificar puerto disponible
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Puerto 5000 ya está en uso"
    pkill -f "mlflow ui"
    sleep 2
fi

# Iniciar MLflow UI
nohup mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlflow_artifacts > mlflow_ui.log 2>&1 &

echo "MLflow UI iniciado en puerto 5000"
echo "PID: $(pgrep -f 'mlflow ui')"
echo "Log: mlflow_ui.log"
"""
    
    with open("start_mlflow.sh", "w") as f:
        f.write(script_content)
    
    # Hacer ejecutable
    os.chmod("start_mlflow.sh", 0o755)
    print("✓ Script de inicio creado: start_mlflow.sh")

def create_stop_script():
    """Crea script para detener MLflow."""
    script_content = """#!/bin/bash
# MLflow Stop Script

echo "Deteniendo MLflow Server..."

# Buscar y matar procesos MLflow
pkill -f "mlflow ui"

if [ $? -eq 0 ]; then
    echo "✓ MLflow detenido exitosamente"
else
    echo "⚠ No se encontraron procesos MLflow activos"
fi
"""
    
    with open("stop_mlflow.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("stop_mlflow.sh", 0o755)
    print("✓ Script de parada creado: stop_mlflow.sh")

def create_experiment_runner():
    """Crea script para ejecutar experimentos."""
    script_content = """#!/bin/bash
# Experiment Runner Script

echo "==================================="
echo "EJECUTOR DE EXPERIMENTOS MLFLOW"
echo "==================================="

# Verificar archivo de datos
if [ ! -f "resumen por item final.xlsx" ]; then
    echo "ERROR: Archivo 'resumen por item final.xlsx' no encontrado"
    echo "Archivos Excel disponibles:"
    ls -la *.xlsx 2>/dev/null || echo "No hay archivos Excel"
    exit 1
fi

# Verificar MLflow UI
if ! lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Iniciando MLflow UI..."
    ./start_mlflow.sh
    sleep 5
fi

# Ejecutar experimentos
echo "Ejecutando experimentos..."
python mlflow_experiments_final.py

echo "Experimentos completados"
echo "MLflow UI: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'localhost'):5000"
"""
    
    with open("run_experiments.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("run_experiments.sh", 0o755)
    print("✓ Script ejecutor creado: run_experiments.sh")

def test_mlflow_installation():
    """Prueba la instalación de MLflow."""
    try:
        import mlflow
        print(f"✓ MLflow {mlflow.__version__} instalado correctamente")
        
        # Probar tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        print("✓ Tracking URI configurado")
        
        return True
    except ImportError:
        print("✗ Error: MLflow no está instalado")
        return False

def generate_setup_report():
    """Genera reporte de configuración."""
    import platform
    
    try:
        import requests
        public_ip = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=5).text
        environment = "AWS_EC2"
    except:
        public_ip = "localhost"
        environment = "LOCAL"
    
    report = {
        "setup_timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "public_ip": public_ip,
        "environment": environment,
        "platform": platform.platform(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "mlflow_config": {
            "tracking_uri": "http://localhost:5000",
            "backend_store": "./mlruns",
            "artifacts_location": "./mlflow_artifacts"
        },
        "files_created": [
            "mlflow_config.json",
            "start_mlflow.sh",
            "stop_mlflow.sh",
            "run_experiments.sh"
        ],
        "directories_created": [
            "mlruns",
            "mlflow_artifacts",
            "logs",
            "reports",
            "models"
        ]
    }
    
    with open("setup_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Reporte en texto
    text_report = f"""
REPORTE DE CONFIGURACIÓN MLFLOW
==============================

INFORMACIÓN DEL SISTEMA
-----------------------
Timestamp: {report['setup_timestamp']}
Hostname: {report['hostname']}
IP: {report['public_ip']}
Entorno: {report['environment']}
Plataforma: {report['platform']}

CONFIGURACIÓN MLFLOW
-------------------
Tracking URI: {report['mlflow_config']['tracking_uri']}
Backend Store: {report['mlflow_config']['backend_store']}
Artifacts: {report['mlflow_config']['artifacts_location']}

COMANDOS DISPONIBLES
-------------------
Iniciar MLflow: ./start_mlflow.sh
Detener MLflow: ./stop_mlflow.sh
Ejecutar experimentos: ./run_experiments.sh

ACCESO MLFLOW UI
---------------
Local: http://localhost:5000
EC2: http://{public_ip}:5000

VERIFICACIÓN
-----------
1. Verificar archivo de datos: ls -la "resumen por item final.xlsx"
2. Iniciar MLflow: ./start_mlflow.sh
3. Verificar UI: curl http://localhost:5000
4. Ejecutar experimentos: ./run_experiments.sh

SIGUIENTE PASO
--------------
Ejecutar: python mlflow_experiments_final.py
"""
    
    with open("CONFIGURACION_MLFLOW.txt", "w", encoding='utf-8') as f:
        f.write(text_report)
    
    print("✓ Reporte de configuración generado")

def main():
    """Función principal de configuración."""
    print("="*50)
    print("CONFIGURADOR AUTOMÁTICO MLFLOW")
    print("="*50)
    
    # Verificaciones básicas
    if not check_python_version():
        return False
    
    # Instalación de dependencias
    print("\n1. Instalando dependencias...")
    if not install_requirements():
        print("✗ Error instalando dependencias")
        return False
    
    # Configuración de directorios
    print("\n2. Configurando directorios...")
    setup_mlflow_directories()
    
    # Configuración MLflow
    print("\n3. Configurando MLflow...")
    create_mlflow_config()
    
    # Configuración específica de entorno
    print("\n4. Configurando entorno...")
    is_ec2 = setup_aws_ec2()
    
    # Crear scripts
    print("\n5. Creando scripts...")
    create_startup_script()
    create_stop_script()
    create_experiment_runner()
    
    # Pruebas
    print("\n6. Verificando instalación...")
    if not test_mlflow_installation():
        return False
    
    # Generar reporte
    print("\n7. Generando reporte...")
    generate_setup_report()
    
    print("\n" + "="*50)
    print("CONFIGURACIÓN COMPLETADA EXITOSAMENTE")
    print("="*50)
    print("Para continuar:")
    print("1. ./start_mlflow.sh")
    print("2. python mlflow_experiments_final.py")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)