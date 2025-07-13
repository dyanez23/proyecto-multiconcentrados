# Instrucciones de Despliegue en AWS EC2

## Paso a Paso para Cumplir con la Entrega 2

### **PARTE 1: Configuración de EC2**

#### 1.1 Crear Instancia EC2
```bash
# En AWS Console:
# - AMI: Ubuntu Server 22.04 LTS
# - Instance Type: t3.medium (mínimo t2.micro)
# - Storage: 20GB GP3
# - Security Group: Permitir puertos 22, 5000, 80
```

#### 1.2 Conectar a la Instancia
```bash
# Desde tu máquina local
ssh -i "tu-key.pem" ubuntu@EC2_PUBLIC_IP

# Una vez conectado, actualizar sistema
sudo apt update && sudo apt upgrade -y
```

#### 1.3 Instalar Python y Dependencias
```bash
# Instalar Python 3.9+
sudo apt install python3 python3-pip python3-venv git -y

# Verificar instalación
python3 --version
pip3 --version
```

### **PARTE 2: Setup del Proyecto**

#### 2.1 Crear Directorio de Trabajo
```bash
# Crear directorio del proyecto
mkdir ~/mlflow_project
cd ~/mlflow_project

# Crear entorno virtual
python3 -m venv mlflow_env
source mlflow_env/bin/activate
```

#### 2.2 Subir Archivos al EC2
```bash
# Desde tu máquina local, subir archivos
scp -i "tu-key.pem" mlflow_experiments_final.py ubuntu@EC2_IP:~/mlflow_project/
scp -i "tu-key.pem" setup_mlflow_environment.py ubuntu@EC2_IP:~/mlflow_project/
scp -i "tu-key.pem" "resumen por item final.xlsx" ubuntu@EC2_IP:~/mlflow_project/

# O clonar desde repositorio Git
git clone https://github.com/tu-usuario/tu-repositorio.git
```

#### 2.3 Ejecutar Setup Automático
```bash
# En EC2, ejecutar configuración
cd ~/mlflow_project
source mlflow_env/bin/activate
python3 setup_mlflow_environment.py
```

### **PARTE 3: Ejecución de Experimentos**

#### 3.1 Iniciar MLflow UI
```bash
# Dar permisos a scripts
chmod +x *.sh

# Iniciar MLflow UI
./start_mlflow.sh

# Verificar que esté corriendo
curl http://localhost:5000
```

#### 3.2 Ejecutar Experimentos
```bash
# Ejecutar experimentos completos
python3 mlflow_experiments_final.py

# O usar script automatizado
./run_experiments.sh
```

#### 3.3 Verificar Acceso Externo
```bash
# Obtener IP pública
curl http://169.254.169.254/latest/meta-data/public-ipv4

# Acceder desde navegador
# http://TU_IP_PUBLICA:5000
```

### **PARTE 4: Captura de Screenshots para Entrega**

#### 4.1 Screenshots Requeridos

**Screenshot 1: Información del Sistema**
- Abrir terminal en EC2
- Ejecutar: `hostname && curl http://169.254.169.254/latest/meta-data/public-ipv4 && whoami`
- Capturar pantalla mostrando hostname, IP pública y usuario

**Screenshot 2: MLflow UI - Experiments List**
- Ir a `http://TU_IP_PUBLICA:5000`
- Capturar pantalla del listado de experimentos
- Debe mostrar: URL con IP pública, experimentos creados

**Screenshot 3: MLflow UI - Experiment Details**
- Hacer clic en experimento "segmentacion_clientes_v2"
- Capturar pantalla mostrando lista de runs
- Debe mostrar: múltiples runs con diferentes algoritmos

**Screenshot 4: MLflow UI - Run Details**
- Hacer clic en un run específico
- Capturar pantalla mostrando parámetros y métricas
- Debe mostrar: parámetros del modelo, métricas de evaluación

**Screenshot 5: MLflow UI - Model Artifacts**
- En el mismo run, ir a la pestaña "Artifacts"
- Capturar pantalla mostrando modelo guardado
- Debe mostrar: modelo pickle, archivos de configuración

#### 4.2 Comando para Generar Screenshots Automáticamente
```bash
# Crear script para documentar la ejecución
cat > capture_evidence.sh << 'EOF'
#!/bin/bash
echo "=== EVIDENCIA DE EJECUCIÓN MLFLOW EN EC2 ===" > evidence.txt
echo "Timestamp: $(date)" >> evidence.txt
echo "Hostname: $(hostname)" >> evidence.txt
echo "Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)" >> evidence.txt
echo "User: $(whoami)" >> evidence.txt
echo "Working Directory: $(pwd)" >> evidence.txt
echo "Python Version: $(python3 --version)" >> evidence.txt
echo "MLflow Version: $(python3 -c 'import mlflow; print(mlflow.__version__)')" >> evidence.txt
echo "=== ARCHIVOS GENERADOS ===" >> evidence.txt
ls -la *.py *.txt *.json *.log 2>/dev/null >> evidence.txt
echo "=== PROCESOS MLFLOW ===" >> evidence.txt
ps aux | grep mlflow >> evidence.txt
echo "=== PUERTOS ABIERTOS ===" >> evidence.txt
netstat -tlnp | grep :5000 >> evidence.txt
EOF

chmod +x capture_evidence.sh
./capture_evidence.sh
cat evidence.txt
```

### **PARTE 5: Documentación y Reportes**

#### 5.1 Generar Reporte Final
```bash
# El script automáticamente genera estos archivos:
ls -la REPORTE_EXPERIMENTOS_MLFLOW.txt
ls -la experiment_summary.json
ls -la system_info.json
ls -la mlflow_experiments.log
```

#### 5.2 Estructura de Entrega para Profesor
```
entrega_2/
├── screenshots/
│   ├── 01_sistema_ec2.png          # Hostname + IP + usuario
│   ├── 02_mlflow_experiments.png   # Lista de experimentos
│   ├── 03_mlflow_runs.png          # Detalle de runs
│   ├── 04_run_details.png          # Parámetros y métricas
│   └── 05_model_artifacts.png      # Artifacts del modelo
├── codigo/
│   ├── mlflow_experiments_final.py
│   ├── setup_mlflow_environment.py
│   └── scripts_auxiliares/
├── reportes/
│   ├── REPORTE_EXPERIMENTOS_MLFLOW.txt
│   ├── DOCUMENTACION_EXPERIMENTOS_MLFLOW.md
│   ├── experiment_summary.json
│   └── system_info.json
├── logs/
│   └── mlflow_experiments.log
└── README_ENTREGA.md
```

### **PARTE 6: Comandos de Verificación**

#### 6.1 Verificar Estado del Sistema
```bash
# Estado de MLflow
ps aux | grep mlflow
netstat -tlnp | grep :5000

# Verificar logs
tail -f mlflow_experiments.log

# Verificar experimentos desde CLI
mlflow experiments list --tracking-uri http://localhost:5000
```

#### 6.2 Troubleshooting Común
```bash
# Si MLflow no inicia
sudo ufw allow 5000
./stop_mlflow.sh
./start_mlflow.sh

# Si no se puede acceder externamente
# Verificar Security Groups en AWS Console
# Debe permitir inbound traffic en puerto 5000

# Si hay errores de permisos
chmod +x *.sh
sudo chown -R ubuntu:ubuntu ~/mlflow_project
```

### **PARTE 7: Checklist para la Entrega**

#### ✅ Código
- [ ] `mlflow_experiments_final.py` funcionando sin errores
- [ ] Script de setup ejecutado exitosamente
- [ ] Todos los archivos subidos al repositorio Git

#### ✅ MLflow en EC2
- [ ] Instancia EC2 activa y accesible
- [ ] MLflow UI corriendo en puerto 5000
- [ ] Experimentos ejecutados completamente
- [ ] Modelos registrados en MLflow

#### ✅ Screenshots
- [ ] Screenshot con IP pública visible en URL
- [ ] Screenshot con usuario de EC2 visible
- [ ] Screenshots de experimentos en MLflow UI
- [ ] Screenshots de métricas y parámetros
- [ ] Screenshots de artifacts/modelos

#### ✅ Documentación
- [ ] Reporte técnico generado
- [ ] Logs de ejecución disponibles
- [ ] Información del sistema documentada
- [ ] Instrucciones de reproducción

#### ✅ Repositorio Git
- [ ] Commits de todos los miembros del equipo
- [ ] Código fuente completo
- [ ] Documentación incluida
- [ ] README con instrucciones

### **COMANDOS FINALES PARA LA ENTREGA**

```bash
# 1. Verificar que todo esté funcionando
curl http://localhost:5000
python3 -c "import mlflow; print('MLflow OK')"

# 2. Generar evidencia final
./capture_evidence.sh

# 3. Crear backup de experimentos
tar -czf mlflow_backup_$(date +%Y%m%d_%H%M%S).tar.gz mlruns/ mlflow_artifacts/ *.log *.txt *.json

# 4. Mantener instancia activa
echo "Instancia EC2 configurada y lista para revisión del profesor"
echo "MLflow UI disponible en: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000"
```

### **NOTAS IMPORTANTES**

1. **NO TERMINAR LA INSTANCIA EC2** hasta después de la calificación
2. **MANTENER MLFLOW UI ACTIVO** durante la revisión
3. **DOCUMENTAR TODAS LAS IPs Y HOSTNAMES** en los screenshots
4. **VERIFICAR ACCESO EXTERNO** antes de entregar
5. **TENER LOGS Y REPORTES LISTOS** para consulta

### **CONTACTO DE EMERGENCIA**

Si hay problemas técnicos durante la entrega:
1. Verificar logs: `tail -f mlflow_experiments.log`
2. Reiniciar MLflow: `./stop_mlflow.sh && ./start_mlflow.sh`
3. Verificar conectividad: `curl http://localhost:5000`
4. Documentar el error y continuar con la entrega