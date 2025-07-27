# Guía de Despliegue Dashboard MLflow V3

## Archivos necesarios

### Archivos principales
```
proyecto/
├── dashboard_mlflow_v3.py          # Dashboard principal
├── requirements_v3.txt             # Dependencias
├── logo_src.py                     # Logo (opcional)
├── resumen por item final.xlsx     # Dataset
└── mlflow_experiments_v3.py        # Modelos MLflow (opcional)
```

## OPCIÓN 1: Despliegue Local (Desarrollo)

### Paso 1: Preparar entorno
```bash
# Crear directorio
mkdir dashboard_v3
cd dashboard_v3

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements_v3.txt
```

### Paso 3: Ejecutar dashboard
```bash
python dashboard_mlflow_v3.py
```

### Paso 4: Acceder
- URL: http://localhost:8050
- MLflow (si está activo): http://localhost:5000

## OPCIÓN 2: Despliegue EC2 (Producción)

### Paso 1: Crear instancia EC2
```bash
# AWS Console:
# - AMI: Ubuntu 22.04 LTS
# - Tipo: t3.medium
# - Storage: 20GB
# - Security Group: puertos 22, 8050, 5000
```

### Paso 2: Conectar y configurar
```bash
ssh -i "tu-key.pem" ubuntu@TU_IP_EC2

# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python
sudo apt install -y python3 python3-pip python3-venv
```

### Paso 3: Subir archivos
```bash
# Desde tu máquina local
scp -i "tu-key.pem" dashboard_mlflow_v3.py ubuntu@TU_IP:~/
scp -i "tu-key.pem" requirements_v3.txt ubuntu@TU_IP:~/
scp -i "tu-key.pem" logo_src.py ubuntu@TU_IP:~/
scp -i "tu-key.pem" "resumen por item final.xlsx" ubuntu@TU_IP:~/
```

### Paso 4: Configurar en EC2
```bash
# En EC2
mkdir dashboard_v3
mv *.py *.txt *.xlsx dashboard_v3/
cd dashboard_v3

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements_v3.txt
```

### Paso 5: Ejecutar dashboard
```bash
# Ejecución simple
python dashboard_mlflow_v3.py

# O con nohup para mantenerlo corriendo
nohup python dashboard_mlflow_v3.py > dashboard.log 2>&1 &
```

### Paso 6: Acceder externamente
- URL: http://TU_IP_EC2:8050

## OPCIÓN 3: Despliegue con Supervisor (Recomendado para producción)

### Paso 1: Instalar Supervisor
```bash
sudo apt install -y supervisor
```

### Paso 2: Crear configuración
```bash
sudo nano /etc/supervisor/conf.d/dashboard.conf
```

```ini
[program:dashboard_v3]
command=/home/ubuntu/dashboard_v3/venv/bin/python dashboard_mlflow_v3.py
directory=/home/ubuntu/dashboard_v3
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/dashboard_v3.log
```

### Paso 3: Activar servicio
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start dashboard_v3

# Verificar estado
sudo supervisorctl status dashboard_v3
```

### Paso 4: Ver logs
```bash
sudo tail -f /var/log/dashboard_v3.log
```

## Gestión y Mantenimiento

### Comandos útiles
```bash
# Ver estado del dashboard
sudo supervisorctl status dashboard_v3

# Reiniciar dashboard
sudo supervisorctl restart dashboard_v3

# Detener dashboard
sudo supervisorctl stop dashboard_v3

# Ver logs en tiempo real
sudo tail -f /var/log/dashboard_v3.log

# Verificar puertos abiertos
netstat -tlnp | grep :8050
```

### Actualizar dashboard
```bash
# Detener servicio
sudo supervisorctl stop dashboard_v3

# Hacer backup
cp dashboard_mlflow_v3.py dashboard_mlflow_v3.py.backup

# Subir nueva versión
# scp -i "tu-key.pem" nuevo_dashboard.py ubuntu@TU_IP:~/dashboard_v3/

# Reiniciar servicio
sudo supervisorctl start dashboard_v3
```

### Monitorear recursos
```bash
# Memoria y CPU
htop

# Espacio en disco
df -h

# Procesos Python
ps aux | grep python
```

## Integración con MLflow

### Si tienes MLflow corriendo:
```bash
# En otra terminal/sesión
cd ~/mlflow_v3_project
source mlflow_v3_env/bin/activate
mlflow ui --host 0.0.0.0 --port 5000

# El dashboard automáticamente intentará conectarse
```

### URLs de acceso:
- Dashboard: http://TU_IP:8050
- MLflow UI: http://TU_IP:5000

## Troubleshooting

### Dashboard no inicia
```bash
# Verificar logs
sudo tail -f /var/log/dashboard_v3.log

# Verificar archivo de datos
ls -la "resumen por item final.xlsx"

# Ejecutar manualmente para ver errores
cd dashboard_v3
source venv/bin/activate
python dashboard_mlflow_v3.py
```

### No se puede acceder externamente
```bash
# Verificar Security Groups en AWS
# Debe permitir puerto 8050

# Verificar que el dashboard esté escuchando en 0.0.0.0
netstat -tlnp | grep :8050
```

### Error de memoria
```bash
# Verificar memoria disponible
free -h

# Si es necesario, crear swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## URLs finales de acceso

### Desarrollo local:
- Dashboard: http://localhost:8050
- MLflow: http://localhost:5000

### Producción EC2:
- Dashboard: http://TU_IP_PUBLICA_EC2:8050
- MLflow: http://TU_IP_PUBLICA_EC2:5000

## Checklist de verificación

- [ ] Archivos subidos correctamente
- [ ] Dependencias instaladas
- [ ] Dashboard iniciando sin errores
- [ ] Acceso externo funcionando
- [ ] Logs sin errores críticos
- [ ] MLflow integrado (opcional)
- [ ] Supervisor configurado (recomendado)
- [ ] Backup de archivos importantes