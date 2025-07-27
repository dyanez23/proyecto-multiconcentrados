# Guía Completa de Despliegue EC2 - MLflow V3 (Entrega Final)

## Objetivo
Desplegar exitosamente el sistema MLflow V3 en AWS EC2 para cumplir con todos los requerimientos de la Entrega 3.

## PARTE 1: Configuración Inicial de EC2

### **Paso 1.1: Crear Instancia EC2**
```bash
# En AWS Console:
# - AMI: Ubuntu Server 22.04 LTS (64-bit x86)
# - Instance Type: t3.medium (recomendado) o t2.medium (mínimo)
# - Storage: 20GB gp3 (mínimo 15GB)
# - Key Pair: Crear nueva o usar existente
```

### **Paso 1.2: Configurar Security Group**
```bash
# Reglas de entrada requeridas:
# SSH     | TCP | 22   | Tu IP / 0.0.0.0/0
# HTTP    | TCP | 80   | 0.0.0.0/0
# Custom  | TCP | 5000 | 0.0.0.0/0  # MLflow UI
# Custom  | TCP | 8050 | 0.0.0.0/0  # Dashboard (si aplica)
```

### **Paso 1.3: Conectar a la Instancia**
```bash
# Desde tu máquina local
chmod 400 tu-key.pem
ssh -i "tu-key.pem" ubuntu@TU_IP_PUBLICA_EC2

# Una vez conectado, actualizar sistema
sudo apt update && sudo apt upgrade -y
```

## PARTE 2: Instalación Base del Sistema

### **Paso 2.1: Instalar Python y Herramientas**
```bash
# Instalar Python 3.9+ y herramientas esenciales
sudo apt install -y python3 python3-pip python3-venv git curl wget unzip
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

# Verificar instalaciones
python3 --version  # Debe ser 3.7+
pip3 --version
git --version
```

### **Paso 2.2: Crear Directorio de Trabajo**
```bash
# Crear directorio del proyecto
mkdir ~/mlflow_v3_project
cd ~/mlflow_v3_project

# Crear entorno virtual
python3 -m venv mlflow_v3_env
source mlflow_v3_env/bin/activate

# Verificar entorno virtual
which python3
which pip3
```

## PARTE 3: Subida de Archivos del Proyecto

### **Método A: Subir archivos vía SCP**
```bash
# Desde tu máquina local (ventana de terminal separada)
scp -i "tu-key.pem" mlflow_experiments_v3.py ubuntu@TU_IP:~/mlflow_v3_project/
scp -i "tu-key.pem" setup_mlflow_v3.py ubuntu@TU_IP:~/mlflow_v3_project/
scp -i "tu-key.pem" "resumen por item final.xlsx" ubuntu@TU_IP:~/mlflow_v3_project/

# Verificar archivos subidos
ssh -i "tu-key.pem" ubuntu@TU_IP "ls -la ~/mlflow_v3_project/"
```

### **Método B: Clonar desde Git (recomendado)**
```bash
# En EC2, clonar repositorio
git clone https://github.com/tu-usuario/proyecto-multiconcentrados.git
cd proyecto-multiconcentrados

# O si los archivos están en un directorio específico
cp ruta/a/archivos/* ~/mlflow_v3_project/
cd ~/mlflow_v3_project
```

### **Paso 3.1: Verificar Archivos Necesarios**
```bash
# En EC2, verificar que tienes estos archivos
ls -la ~/mlflow_v3_project/

# Archivos requeridos:
# ✓ mlflow_experiments_v3.py
# ✓ setup_mlflow_v3.py  
# ✓ resumen por item final.xlsx
```

## PARTE 4: Configuración Automática V3

### **Paso 4.1: Ejecutar Setup Automático**
```bash
# En EC2, activar entorno virtual
cd ~/mlflow_v3_project
source mlflow_v3_env/bin/activate

# Ejecutar configuración automática V3
python3 setup_mlflow_v3.py

# El script automáticamente:
# ✓ Verifica Python 3.7+
# ✓ Instala todas las dependencias
# ✓ Crea directorios necesarios
# ✓ Configura MLflow V3
# ✓ Crea scripts de ejecución
# ✓ Configura firewall EC2
# ✓ Genera reportes de configuración
```

### **Paso 4.2: Verificar Configuración**
```bash
# Verificar que todo se configuró correctamente
ls -la

# Deberías ver estos archivos nuevos:
# ✓ start_mlflow_v3.sh
# ✓ stop_mlflow_v3.sh
# ✓ run_experiments_v3.sh
# ✓ deploy_ec2_v3.sh
# ✓ mlflow_config_v3.json
# ✓ CONFIGURACION_MLFLOW_V3.txt

# Verificar permisos de ejecución
ls -la *.sh
```

### **Paso 4.3: Configuración Manual del Firewall (si es necesario)**
```bash
# Solo si el setup automático falló en esta parte
sudo ufw --force enable
sudo ufw allow 22      # SSH
sudo ufw allow 5000    # MLflow UI
sudo ufw allow 80      # HTTP

# Verificar reglas
sudo ufw status verbose
```

## PARTE 5: Ejecución del Sistema MLflow V3

### **Paso 5.1: Iniciar MLflow UI**
```bash
# Usar script automático
./start_mlflow_v3.sh

# O manualmente si es necesario:
# nohup mlflow ui --host 0.0.0.0 --port 5000 > logs/mlflow_ui_v3.log 2>&1 &
```

### **Paso 5.2: Verificar MLflow UI**
```bash
# Verificar que MLflow está ejecutándose
ps aux | grep mlflow

# Verificar conectividad local
curl http://localhost:5000

# Obtener IP pública para acceso externo
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Acceder desde navegador:
# http://TU_IP_PUBLICA:5000
```

### **Paso 5.3: Ejecutar Experimentos V3**
```bash
# Opción 1: Usar script automático (recomendado)
./run_experiments_v3.sh

# Opción 2: Ejecución manual
python3 mlflow_experiments_v3.py

# Monitorear progreso
tail -f mlflow_experiments_v3.log
```

## PARTE 6: Verificación de Resultados

### **Paso 6.1: Verificar Experimentos en MLflow UI**
```bash
# Acceder a MLflow UI desde navegador
echo "MLflow UI disponible en:"
echo "http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"

# Verificar desde línea de comandos
mlflow experiments list --tracking-uri http://localhost:5000
```

### **Paso 6.2: Verificar Archivos Generados**
```bash
# Verificar reportes generados
ls -la *_v3.txt *_v3.json

# Archivos esperados:
# ✓ REPORTE_EXPERIMENTOS_MLFLOW_V3.txt
# ✓ system_info_v3.json
# ✓ data_quality_report_v3.json
# ✓ experiment_summary_v3.json
# ✓ mlflow_experiments_v3.log

# Verificar contenido de reportes
head -20 REPORTE_EXPERIMENTOS_MLFLOW_V3.txt
```

### **Paso 6.3: Verificar Modelos en MLflow**
```bash
# Verificar que los modelos se guardaron
find mlruns -name "*.pkl" -o -name "model" | head -10

# Verificar experimentos
find mlruns -name "meta.yaml" | wc -l
```

## PARTE 7: Captura de Screenshots para Entrega

### **Screenshot 1: Información del Sistema EC2**
```bash
# Ejecutar comandos que muestren la información requerida
echo "=============================================="
echo "INFORMACIÓN DEL SISTEMA EC2 - MLFLOW V3"
echo "=============================================="
echo "Hostname: $(hostname)"
echo "Usuario: $(whoami)"
echo "IP Pública: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Región: $(curl -s http://169.254.169.254/latest/meta-data/placement/region)"
echo "Timestamp: $(date)"
echo "=============================================="

# Capturar esta pantalla completa
```

### **Screenshot 2: MLflow UI - Lista de Experimentos**
```bash
# Acceder a: http://TU_IP_PUBLICA:5000
# Capturar pantalla mostrando:
# ✓ URL con IP pública visible
# ✓ Experimento "segmentacion_clientes_v3"
# ✓ Múltiples runs visibles
```

### **Screenshot 3: MLflow UI - Detalle de Runs**
```bash
# Hacer clic en experimento "segmentacion_clientes_v3"
# Capturar pantalla mostrando:
# ✓ Lista de runs de clustering y clasificación
# ✓ Métricas como Silhouette Score, AUC-ROC
# ✓ Estados "FINISHED"
```

### **Screenshot 4: MLflow UI - Detalles de un Run**
```bash
# Hacer clic en un run específico
# Capturar pantalla mostrando:
# ✓ Parámetros del modelo
# ✓ Métricas de evaluación
# ✓ Tags (version: 3.0, status: completed)
```

### **Screenshot 5: MLflow UI - Artifacts del Modelo**
```bash
# En el mismo run, ir a pestaña "Artifacts"
# Capturar pantalla mostrando:
# ✓ Modelo guardado (model/)
# ✓ Archivos del modelo
# ✓ Metadatos
```

## PARTE 8: Generar Evidencia para Entrega

### **Paso 8.1: Crear Archivo de Evidencia**
```bash
# Crear evidencia completa para la entrega
cat > evidencia_entrega_v3.txt << 'EOF'
============================================
EVIDENCIA DE EJECUCIÓN MLFLOW V3 EN EC2
============================================
EOF

echo "Timestamp: $(date)" >> evidencia_entrega_v3.txt
echo "Hostname: $(hostname)" >> evidencia_entrega_v3.txt
echo "Usuario: $(whoami)" >> evidencia_entrega_v3.txt
echo "IP Pública: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)" >> evidencia_entrega_v3.txt
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)" >> evidencia_entrega_v3.txt
echo "Directorio: $(pwd)" >> evidencia_entrega_v3.txt
echo "Python: $(python3 --version)" >> evidencia_entrega_v3.txt
echo "MLflow: $(python3 -c 'import mlflow; print(mlflow.__version__)')" >> evidencia_entrega_v3.txt

echo "============================================" >> evidencia_entrega_v3.txt
echo "ARCHIVOS DEL PROYECTO V3" >> evidencia_entrega_v3.txt
echo "============================================" >> evidencia_entrega_v3.txt
ls -la *.py *.txt *.json *.xlsx *.sh >> evidencia_entrega_v3.txt

echo "============================================" >> evidencia_entrega_v3.txt
echo "PROCESOS MLFLOW ACTIVOS" >> evidencia_entrega_v3.txt
echo "============================================" >> evidencia_entrega_v3.txt
ps aux | grep mlflow >> evidencia_entrega_v3.txt

echo "============================================" >> evidencia_entrega_v3.txt
echo "PUERTOS ABIERTOS" >> evidencia_entrega_v3.txt
echo "============================================" >> evidencia_entrega_v3.txt
netstat -tlnp | grep :5000 >> evidencia_entrega_v3.txt

echo "============================================" >> evidencia_entrega_v3.txt
echo "EXPERIMENTOS MLFLOW" >> evidencia_entrega_v3.txt
echo "============================================" >> evidencia_entrega_v3.txt
find mlruns -name "*.yaml" | wc -l >> evidencia_entrega_v3.txt
ls -la mlruns/ >> evidencia_entrega_v3.txt

# Mostrar evidencia
cat evidencia_entrega_v3.txt
```

### **Paso 8.2: Verificar Acceso Externo**
```bash
# Verificar que MLflow UI es accesible externamente
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Verificando acceso externo a MLflow V3..."
echo "URL: http://$PUBLIC_IP:5000"

# Probar conectividad
curl -I http://localhost:5000
```

## PARTE 9: Troubleshooting Común

### **Problema 1: MLflow UI no inicia**
```bash
# Verificar logs
cat logs/mlflow_ui_v3.log

# Matar procesos existentes
pkill -f "mlflow ui"

# Verificar puerto libre
lsof -i :5000

# Reiniciar MLflow
./stop_mlflow_v3.sh
sleep 3
./start_mlflow_v3.sh
```

### **Problema 2: No se puede acceder desde navegador**
```bash
# Verificar Security Groups en AWS Console
# Debe permitir entrada en puerto 5000

# Verificar firewall local
sudo ufw status

# Verificar que MLflow está escuchando en todas las interfaces
netstat -tlnp | grep :5000
# Debe mostrar: 0.0.0.0:5000 (no 127.0.0.1:5000)
```

### **Problema 3: Experimentos fallan**
```bash
# Verificar archivo de datos
ls -la "resumen por item final.xlsx"

# Verificar logs de experimentos
tail -50 mlflow_experiments_v3.log

# Verificar dependencias
python3 -c "import mlflow, sklearn, pandas, numpy; print('Dependencias OK')"

# Ejecutar con más verbosidad
python3 mlflow_experiments_v3.py 2>&1 | tee debug_output.log
```

### **Problema 4: Permisos de archivos**
```bash
# Corregir permisos de scripts
chmod +x *.sh

# Corregir propiedad de archivos
sudo chown -R ubuntu:ubuntu ~/mlflow_v3_project

# Verificar espacio en disco
df -h
```

## PARTE 10: Checklist Final pre-Entrega

### **Verificaciones Técnicas**
```bash
# ✓ MLflow UI accesible externamente
curl -I http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000

# ✓ Experimentos ejecutados completamente
find mlruns -name "*.yaml" | wc -l  # Debe ser > 20

# ✓ Modelos guardados
find mlruns -name "model" | wc -l   # Debe ser > 5

# ✓ Reportes V3 generados
ls -la *_v3.txt *_v3.json

# ✓ Logs sin errores críticos
grep -i error mlflow_experiments_v3.log | wc -l  # Debe ser 0 o muy pocos
```

### Screenshots Requeridos
- Terminal EC2 con hostname + IP + usuario visible
- MLflow UI homepage con IP pública en URL
- Lista de experimentos mostrando runs completados
- Detalles de un run con parámetros y métricas
- Artifacts de modelo guardado
- Comparación de múltiples runs

### Archivos para Entregar
- Código fuente: `mlflow_experiments_v3.py`
- Script setup: `setup_mlflow_v3.py`
- Reportes: `REPORTE_EXPERIMENTOS_MLFLOW_V3.txt`
- Configuración: `mlflow_config_v3.json`
- Evidencia: `evidencia_entrega_v3.txt`
- Info EC2: `ec2_info_v3.json`

## COMANDOS DE EMERGENCIA

### **Reinicio Completo del Sistema**
```bash
# Si todo falla, reinicio completo
./stop_mlflow_v3.sh
pkill -f python3
pkill -f mlflow

# Limpiar entorno
rm -rf mlruns mlflow_artifacts logs

# Reconfigurar
python3 setup_mlflow_v3.py
./start_mlflow_v3.sh
./run_experiments_v3.sh
```

### **Backup de Emergencia**
```bash
# Crear backup de todo antes de la entrega
tar -czf backup_mlflow_v3_$(date +%Y%m%d_%H%M%S).tar.gz \
    mlruns/ mlflow_artifacts/ logs/ *.py *.sh *.json *.txt *.xlsx

# Subir backup a S3 (opcional)
# aws s3 cp backup_mlflow_v3_*.tar.gz s3://tu-bucket/
```

## Información de Contacto para la Entrega

### **URLs de Acceso**
```bash
# Obtener URLs finales para incluir en entrega
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "MLflow UI: http://$PUBLIC_IP:5000"
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "SSH Access: ssh -i tu-key.pem ubuntu@$PUBLIC_IP"
```

### **Credenciales para el Profesor**
```bash
# Información que necesita el profesor
cat > acceso_profesor.txt << EOF
ACCESO MLFLOW V3 - ENTREGA FINAL
================================
MLflow UI: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000
Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
Usuario: ubuntu
Directorio: ~/mlflow_v3_project
Experimento: segmentacion_clientes_v3
Timestamp: $(date)
================================
EOF

cat acceso_profesor.txt
```

## NOTAS IMPORTANTES FINALES

1. NO TERMINAR LA INSTANCIA EC2 hasta después de la calificación
2. MANTENER MLFLOW UI ACTIVO durante la revisión del profesor
3. DOCUMENTAR TODAS LAS IPs en screenshots y reportes
4. VERIFICAR ACCESO EXTERNO antes de entregar
5. TENER LOGS DISPONIBLES para consulta del profesor

## LISTO PARA LA ENTREGA

Si completaste todos los pasos de esta guía, tu sistema MLflow V3 está listo para la evaluación de la Entrega 3. Asegúrate de tener todos los screenshots, archivos y evidencias organizados según los requerimientos del profesor.