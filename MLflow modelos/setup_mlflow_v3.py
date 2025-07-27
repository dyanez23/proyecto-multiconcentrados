"""
Setup MLflow Environment V3 - Configuración Automática para Entrega 3
=====================================================================

Script para configurar automáticamente el entorno MLflow V3
en instancias EC2 y sistemas locales para la entrega final.

Autor: Equipo Analítica
Fecha: Julio 2025
Versión: 3.0 - Entrega Final
"""

import os
import sys
import subprocess
import json
import socket
import platform
from datetime import datetime
from pathlib import Path

class MLflowSetupV3:
    """Configurador automático para MLflow V3."""
    
    def __init__(self):
        self.setup_results = {
            'python_check': False,
            'dependencies_installed': False,
            'directories_created': False,
            'scripts_created': False,
            'mlflow_tested': False,
            'ec2_configured': False
        }
        self.is_ec2 = self._detect_ec2()
        self.public_ip = self._get_public_ip()
        
    def run_complete_setup(self):
        """Ejecuta configuración completa para V3."""
        print("="*70)
        print("CONFIGURADOR AUTOMÁTICO MLFLOW V3 - ENTREGA FINAL")
        print("="*70)
        
        steps = [
            ("Verificando Python", self.check_python_version),
            ("Instalando dependencias", self.install_requirements),
            ("Creando directorios", self.setup_directories),
            ("Configurando MLflow", self.create_mlflow_config_v3),
            ("Creando scripts", self.create_execution_scripts),
            ("Configurando EC2", self.setup_ec2_environment),
            ("Probando MLflow", self.test_mlflow_installation),
            ("Generando reportes", self.generate_setup_report_v3)
        ]
        
        for step_name, step_function in steps:
            print(f"\n{step_name}...")
            try:
                result = step_function()
                if result:
                    print(f"✓ {step_name} completado")
                else:
                    print(f"⚠ {step_name} con advertencias")
            except Exception as e:
                print(f"✗ Error en {step_name}: {str(e)}")
                return False
        
        self._print_final_summary()
        return True
    
    def check_python_version(self):
        """Verifica versión de Python."""
        if sys.version_info < (3, 7):
            print(f"ERROR: Se requiere Python 3.7+, encontrado {sys.version}")
            return False
        
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        self.setup_results['python_check'] = True
        return True
    
    def install_requirements(self):
        """Instala dependencias para V3."""
        requirements_v3 = [
            "mlflow>=2.0.0",
            "scikit-learn>=1.0.0", 
            "pandas>=1.3.0",
            "numpy>=1.20.0",
            "boto3>=1.20.0",
            "requests>=2.25.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "psutil>=5.8.0"  # Para verificar memoria
        ]
        
        failed_packages = []
        
        for package in requirements_v3:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, "--quiet"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"✓ {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Error instalando {package}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"Paquetes fallidos: {failed_packages}")
            return False
        
        self.setup_results['dependencies_installed'] = True
        return True
    
    def setup_directories(self):
        """Crea estructura de directorios para V3."""
        directories_v3 = [
            "mlruns",
            "mlflow_artifacts",
            "logs",
            "reports_v3",
            "models_v3",
            "data",
            "scripts_v3",
            "docs_v3"
        ]
        
        for dir_name in directories_v3:
            try:
                Path(dir_name).mkdir(exist_ok=True)
                print(f"✓ {dir_name}/")
            except Exception as e:
                print(f"✗ Error creando {dir_name}: {e}")
                return False
        
        self.setup_results['directories_created'] = True
        return True
    
    def create_mlflow_config_v3(self):
        """Crea configuración MLflow V3."""
        config_v3 = {
            "version": "3.0",
            "tracking_uri": "http://localhost:5000",
            "artifact_location": "./mlflow_artifacts",
            "default_experiment": "segmentacion_clientes_v3",
            "backend_store_uri": "./mlruns",
            "serve_artifacts": True,
            "experiment_settings": {
                "clustering_algorithms": ["kmeans", "agglomerative", "dbscan"],
                "classification_algorithms": ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"],
                "optimization": "production_ready",
                "data_validation": True,
                "quality_threshold": 50
            },
            "ec2_settings": {
                "is_ec2": self.is_ec2,
                "public_ip": self.public_ip,
                "ports": [5000, 22, 80],
                "host": "0.0.0.0"
            }
        }
        
        with open("mlflow_config_v3.json", "w") as f:
            json.dump(config_v3, f, indent=2)
        
        print("✓ Configuración MLflow V3 creada")
        return True
    
    def create_execution_scripts(self):
        """Crea scripts de ejecución para V3."""
        
        # Script de inicio MLflow V3
        start_script = """#!/bin/bash
# MLflow V3 Startup Script - Entrega Final

echo "Iniciando MLflow Server V3..."

# Verificar puerto
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Puerto 5000 ocupado, deteniendo procesos previos..."
    pkill -f "mlflow ui" 2>/dev/null
    sleep 3
fi

# Crear directorio de logs si no existe
mkdir -p logs

# Iniciar MLflow UI V3
nohup mlflow ui \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri ./mlruns \\
    --default-artifact-root ./mlflow_artifacts \\
    > logs/mlflow_ui_v3.log 2>&1 &

sleep 3

if pgrep -f "mlflow ui" >/dev/null; then
    echo "✓ MLflow UI V3 iniciado exitosamente"
    echo "PID: $(pgrep -f 'mlflow ui')"
    echo "Log: logs/mlflow_ui_v3.log"
    echo "URL Local: http://localhost:5000"
    if [ -n "$PUBLIC_IP" ]; then
        echo "URL EC2: http://$PUBLIC_IP:5000"
    fi
else
    echo "✗ Error iniciando MLflow UI"
    cat logs/mlflow_ui_v3.log
    exit 1
fi
"""
        
        # Script de parada
        stop_script = """#!/bin/bash
# MLflow V3 Stop Script

echo "Deteniendo MLflow Server V3..."

pkill -f "mlflow ui"
sleep 2

if pgrep -f "mlflow ui" >/dev/null; then
    echo "⚠ Forzando cierre de MLflow..."
    pkill -9 -f "mlflow ui"
fi

echo "✓ MLflow V3 detenido"
"""
        
        # Script ejecutor de experimentos V3
        run_experiments_script = """#!/bin/bash
# Experiment Runner V3 - Entrega Final

echo "=========================================="
echo "EJECUTOR DE EXPERIMENTOS MLFLOW V3"
echo "=========================================="

# Verificar archivo de datos
DATA_FILE="resumen por item final.xlsx"
if [ ! -f "$DATA_FILE" ]; then
    echo "✗ ERROR: Archivo '$DATA_FILE' no encontrado"
    echo "Archivos Excel disponibles:"
    ls -la *.xlsx 2>/dev/null || echo "No hay archivos Excel"
    exit 1
fi

echo "✓ Archivo de datos encontrado: $DATA_FILE"

# Verificar script principal V3
if [ ! -f "mlflow_experiments_v3.py" ]; then
    echo "✗ ERROR: Script 'mlflow_experiments_v3.py' no encontrado"
    exit 1
fi

echo "✓ Script V3 encontrado"

# Verificar MLflow UI
if ! lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Iniciando MLflow UI V3..."
    ./start_mlflow_v3.sh
    sleep 5
fi

# Verificar conectividad
if curl -s http://localhost:5000 >/dev/null; then
    echo "✓ MLflow UI accesible"
else
    echo "⚠ MLflow UI no responde, continuando..."
fi

# Configurar variables de entorno
export PYTHONPATH="${PYTHONPATH}:."
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Ejecutar experimentos V3
echo "=========================================="
echo "EJECUTANDO EXPERIMENTOS V3..."
echo "Timestamp: $(date)"
echo "=========================================="

python3 mlflow_experiments_v3.py

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ EXPERIMENTOS V3 COMPLETADOS EXITOSAMENTE"
    echo "Tiempo: $(date)"
    echo "Reportes generados:"
    ls -la *_v3.txt *_v3.json 2>/dev/null
    echo "MLflow UI: http://localhost:5000"
    if command -v curl >/dev/null; then
        PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")
        echo "EC2 URL: http://$PUBLIC_IP:5000"
    fi
else
    echo "✗ ERROR EN EXPERIMENTOS V3"
    echo "Código de salida: $EXIT_CODE"
    echo "Revisar logs: mlflow_experiments_v3.log"
fi
echo "=========================================="

exit $EXIT_CODE
"""
        
        # Script de despliegue EC2
        deploy_script = """#!/bin/bash
# Deploy MLflow V3 to EC2 - Complete Setup

echo "============================================"
echo "DESPLIEGUE COMPLETO MLFLOW V3 EN EC2"
echo "============================================"

# Verificar si estamos en EC2
if curl -s --max-time 5 http://169.254.169.254/latest/meta-data/instance-id >/dev/null 2>&1; then
    echo "✓ Instancia EC2 detectada"
    export PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
    export INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    echo "IP Pública: $PUBLIC_IP"
    echo "Instance ID: $INSTANCE_ID"
else
    echo "⚠ Entorno local detectado"
    export PUBLIC_IP="localhost"
fi

# Configurar firewall (si es EC2)
if [ "$PUBLIC_IP" != "localhost" ]; then
    echo "Configurando firewall EC2..."
    sudo ufw --force enable 2>/dev/null || true
    sudo ufw allow 22 2>/dev/null || true
    sudo ufw allow 5000 2>/dev/null || true
    sudo ufw allow 80 2>/dev/null || true
    echo "✓ Puertos configurados: 22, 5000, 80"
fi

# Ejecutar setup V3
echo "Ejecutando setup V3..."
python3 setup_mlflow_v3.py

if [ $? -eq 0 ]; then
    echo "✓ Setup V3 completado"
else
    echo "✗ Error en setup V3"
    exit 1
fi

# Iniciar MLflow
echo "Iniciando MLflow V3..."
./start_mlflow_v3.sh

# Verificar funcionamiento
sleep 5
if curl -s http://localhost:5000 >/dev/null; then
    echo "✓ MLflow V3 funcionando correctamente"
    echo "============================================"
    echo "DESPLIEGUE COMPLETADO"
    echo "Local: http://localhost:5000"
    echo "EC2: http://$PUBLIC_IP:5000"
    echo "============================================"
else
    echo "✗ Error: MLflow no responde"
    exit 1
fi
"""
        
        # Crear todos los scripts
        scripts = {
            "start_mlflow_v3.sh": start_script,
            "stop_mlflow_v3.sh": stop_script,
            "run_experiments_v3.sh": run_experiments_script,
            "deploy_ec2_v3.sh": deploy_script
        }
        
        for script_name, script_content in scripts.items():
            try:
                with open(script_name, "w") as f:
                    f.write(script_content)
                os.chmod(script_name, 0o755)
                print(f"✓ {script_name}")
            except Exception as e:
                print(f"✗ Error creando {script_name}: {e}")
                return False
        
        self.setup_results['scripts_created'] = True
        return True
    
    def setup_ec2_environment(self):
        """Configura entorno específico de EC2."""
        if not self.is_ec2:
            print("⚠ Entorno local - omitiendo configuración EC2")
            return True
        
        try:
            # Configurar firewall
            commands = [
                "sudo ufw --force enable",
                "sudo ufw allow 22",
                "sudo ufw allow 5000", 
                "sudo ufw allow 80"
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd.split(), check=True, capture_output=True)
                    print(f"✓ {cmd}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠ {cmd} falló: {e}")
            
            # Crear archivo de información EC2
            ec2_info = {
                "instance_id": self._get_instance_id(),
                "public_ip": self.public_ip,
                "private_ip": self._get_private_ip(),
                "region": self._get_region(),
                "setup_timestamp": datetime.now().isoformat(),
                "mlflow_url": f"http://{self.public_ip}:5000"
            }
            
            with open("ec2_info_v3.json", "w") as f:
                json.dump(ec2_info, f, indent=2)
            
            print(f"✓ Configuración EC2 completada - IP: {self.public_ip}")
            self.setup_results['ec2_configured'] = True
            return True
            
        except Exception as e:
            print(f"⚠ Error configurando EC2: {e}")
            return False
    
    def test_mlflow_installation(self):
        """Prueba instalación de MLflow V3."""
        try:
            import mlflow
            print(f"✓ MLflow {mlflow.__version__} importado correctamente")
            
            # Probar configuración
            mlflow.set_tracking_uri("http://localhost:5000")
            print("✓ Tracking URI configurado")
            
            # Verificar dependencias críticas
            critical_imports = ['sklearn', 'pandas', 'numpy']
            for module in critical_imports:
                try:
                    __import__(module)
                    print(f"✓ {module} disponible")
                except ImportError:
                    print(f"✗ {module} faltante")
                    return False
            
            self.setup_results['mlflow_tested'] = True
            return True
            
        except ImportError:
            print("✗ Error: MLflow no está instalado correctamente")
            return False
    
    def generate_setup_report_v3(self):
        """Genera reporte completo de configuración V3."""
        
        # Información del sistema
        system_info = {
            "setup_timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "public_ip": self.public_ip,
            "environment": "AWS_EC2" if self.is_ec2 else "LOCAL",
            "platform": platform.platform(),
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "version": "3.0"
        }
        
        # Configuración MLflow
        mlflow_config = {
            "tracking_uri": "http://localhost:5000",
            "backend_store": "./mlruns",
            "artifacts_location": "./mlflow_artifacts",
            "experiment_name": "segmentacion_clientes_v3",
            "algorithms": {
                "clustering": ["kmeans", "agglomerative", "dbscan"],
                "classification": ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree"]
            }
        }
        
        # Archivos y directorios creados
        files_created = [
            "mlflow_config_v3.json",
            "start_mlflow_v3.sh",
            "stop_mlflow_v3.sh", 
            "run_experiments_v3.sh",
            "deploy_ec2_v3.sh",
            "ec2_info_v3.json" if self.is_ec2 else None
        ]
        files_created = [f for f in files_created if f is not None]
        
        directories_created = [
            "mlruns", "mlflow_artifacts", "logs", "reports_v3",
            "models_v3", "data", "scripts_v3", "docs_v3"
        ]
        
        # Reporte completo
        setup_report = {
            "system_info": system_info,
            "mlflow_config": mlflow_config,
            "setup_results": self.setup_results,
            "files_created": files_created,
            "directories_created": directories_created
        }
        
        # Guardar JSON
        with open("setup_report_v3.json", "w") as f:
            json.dump(setup_report, f, indent=2)
        
        # Reporte en texto
        text_report = f"""
REPORTE DE CONFIGURACIÓN MLFLOW V3 - ENTREGA FINAL
=================================================

INFORMACIÓN DEL SISTEMA
-----------------------
Timestamp: {system_info['setup_timestamp']}
Hostname: {system_info['hostname']}
IP: {system_info['public_ip']}
Entorno: {system_info['environment']}
Plataforma: {system_info['platform'][:80]}...
Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
Directorio: {system_info['working_directory']}

CONFIGURACIÓN MLFLOW V3
-----------------------
Tracking URI: {mlflow_config['tracking_uri']}
Backend Store: {mlflow_config['backend_store']}
Artifacts: {mlflow_config['artifacts_location']}
Experimento: {mlflow_config['experiment_name']}

ALGORITMOS CONFIGURADOS
-----------------------
Clustering: {', '.join(mlflow_config['algorithms']['clustering'])}
Clasificación: {', '.join(mlflow_config['algorithms']['classification'])}

RESULTADOS DE CONFIGURACIÓN
---------------------------
✓ Verificación Python: {self.setup_results['python_check']}
✓ Dependencias instaladas: {self.setup_results['dependencies_installed']}
✓ Directorios creados: {self.setup_results['directories_created']}
✓ Scripts creados: {self.setup_results['scripts_created']}
✓ MLflow probado: {self.setup_results['mlflow_tested']}
✓ EC2 configurado: {self.setup_results['ec2_configured']}

COMANDOS DISPONIBLES V3
-----------------------
Configurar entorno: python3 setup_mlflow_v3.py
Iniciar MLflow: ./start_mlflow_v3.sh
Detener MLflow: ./stop_mlflow_v3.sh
Ejecutar experimentos: ./run_experiments_v3.sh
Desplegar en EC2: ./deploy_ec2_v3.sh

ACCESO MLFLOW UI V3
------------------
Local: http://localhost:5000
EC2: http://{self.public_ip}:5000

ARCHIVOS PRINCIPALES V3
-----------------------
Script principal: mlflow_experiments_v3.py
Configuración: mlflow_config_v3.json
Logs: logs/mlflow_ui_v3.log
Reportes: reports_v3/

SIGUIENTE PASO
--------------
1. Verificar datos: ls -la "resumen por item final.xlsx"
2. Ejecutar: ./run_experiments_v3.sh
3. Acceder UI: http://localhost:5000

VERIFICACIÓN FINAL
------------------
curl http://localhost:5000
python3 -c "import mlflow; print('MLflow V3 OK')"

NOTAS IMPORTANTES
-----------------
- Mantener instancia EC2 activa para revisión
- Verificar puertos 5000, 22, 80 abiertos en Security Groups
- Archivo de datos debe estar en directorio raíz
- Logs disponibles en directorio logs/
"""
        
        with open("CONFIGURACION_MLFLOW_V3.txt", "w", encoding='utf-8') as f:
            f.write(text_report)
        
        print("✓ Reportes de configuración V3 generados")
        return True
    
    def _detect_ec2(self):
        """Detecta si está ejecutándose en EC2."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def _get_public_ip(self):
        """Obtiene IP pública."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/public-ipv4',
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return "localhost"
    
    def _get_instance_id(self):
        """Obtiene Instance ID de EC2."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return "unknown"
    
    def _get_private_ip(self):
        """Obtiene IP privada."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/local-ipv4',
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return "unknown"
    
    def _get_region(self):
        """Obtiene región de EC2."""
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/placement/region',
                timeout=5
            )
            if response.status_code == 200:
                return response.text
        except:
            pass
        return "unknown"
    
    def _print_final_summary(self):
        """Imprime resumen final."""
        print("\n" + "="*70)
        print("CONFIGURACIÓN MLFLOW V3 COMPLETADA")
        print("="*70)
        
        success_count = sum(self.setup_results.values())
        total_steps = len(self.setup_results)
        
        print(f"Pasos completados: {success_count}/{total_steps}")
        
        if success_count == total_steps:
            print("✓ CONFIGURACIÓN EXITOSA")
            print("\nPróximos pasos:")
            print("1. ./start_mlflow_v3.sh")
            print("2. ./run_experiments_v3.sh")
            print(f"3. Acceder a http://{self.public_ip}:5000")
        else:
            print("⚠ CONFIGURACIÓN PARCIAL")
            print("Revisar errores arriba y volver a ejecutar")
        
        print("="*70)


def main():
    """Función principal de configuración V3."""
    setup = MLflowSetupV3()
    return setup.run_complete_setup()


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n✗ Setup falló. Revisar errores arriba.")
        sys.exit(1)
    else:
        print("\n✓ Setup V3 completado exitosamente.")
