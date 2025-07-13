# Documentación Técnica - Experimentos MLflow

## Resumen Ejecutivo

Este proyecto implementa un sistema completo de experimentación de machine learning utilizando MLflow para el tracking, versionado y comparación de modelos de segmentación y clasificación de clientes.

## Arquitectura del Sistema

### Componentes Principales

1. **DataProcessor**: Carga, valida y prepara datasets
2. **MLflowExperimentManager**: Gestiona experimentos y tracking
3. **Algoritmos de Clustering**: K-Means, Hierarchical, DBSCAN
4. **Algoritmos de Clasificación**: Logistic Regression, Random Forest, Gradient Boosting, SVM

### Flujo de Datos

```
Excel Data → Data Processing → Feature Engineering → Model Training → MLflow Tracking → Results Analysis
```

## Modelos Implementados

### Clustering Models

| Algoritmo | Parámetros Evaluados | Métrica Principal |
|-----------|---------------------|-------------------|
| K-Means | n_clusters (2-8), init, n_init | Silhouette Score |
| Hierarchical | n_clusters (2-8), linkage | Silhouette Score |
| DBSCAN | eps, min_samples | Silhouette Score |

### Classification Models

| Algoritmo | Parámetros Evaluados | Métrica Principal |
|-----------|---------------------|-------------------|
| Logistic Regression | C, penalty, solver | AUC-ROC |
| Random Forest | n_estimators, max_depth, min_samples | AUC-ROC |
| Gradient Boosting | n_estimators, learning_rate, max_depth | AUC-ROC |
| SVM | C, kernel, gamma | AUC-ROC |

## Métricas de Evaluación

### Clustering
- **Silhouette Score**: Medida de cohesión y separación de clusters
- **Davies-Bouldin Score**: Ratio de dispersión intra-cluster vs inter-cluster
- **Calinski-Harabasz Score**: Ratio de suma de cuadrados inter vs intra-cluster

### Clasificación
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Proporción de positivos verdaderos entre predicciones positivas
- **Recall**: Proporción de positivos verdaderos detectados
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC

## Estructura de Características

### Clustering Features
- `valor_total_sum`: Total gastado por cliente
- `valor_total_mean`: Gasto promedio por transacción
- `valor_total_count`: Número de transacciones
- `cantidad_sum`: Cantidad total de productos
- `ticket_promedio`: Valor promedio por factura
- `descuento_ratio`: Proporción de descuentos
- `codigo_producto_nunique`: Diversidad de productos
- `categoria_nunique`: Diversidad de categorías
- `diversidad_productos`: Productos únicos por transacción
- `diversidad_geografica`: Municipios únicos por transacción

### Classification Features
- `valor_total_sum`: Total gastado
- `valor_total_mean`: Gasto promedio
- `valor_total_count`: Número de transacciones
- `cantidad_sum`: Cantidad total
- `gasto_mensual_promedio`: Gasto promedio mensual
- `ticket_promedio`: Ticket promedio
- `codigo_producto_nunique`: Productos únicos
- `categoria_nunique`: Categorías únicas
- `departamento_nunique`: Departamentos únicos
- `municipio_nunique`: Municipios únicos

## Configuración de Experimentos

### Clustering Experiments
```python
clustering_configs = {
    'kmeans': {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20]
    },
    'hierarchical': {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8],
        'linkage': ['ward', 'complete', 'average']
    },
    'dbscan': {
        'eps': [0.3, 0.5, 0.7, 1.0],
        'min_samples': [3, 5, 7, 10]
    }
}
```

### Classification Experiments
```python
classification_configs = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10]
    }
}
```

## Tracking en MLflow

### Parámetros Registrados
- Hiperparámetros del modelo
- Configuración del dataset
- Semilla aleatoria
- Configuración de validación cruzada

### Métricas Registradas
- Métricas de evaluación principales
- Métricas de validación cruzada
- Feature importance (cuando disponible)
- Distribución de clusters/clases

### Artifacts Registrados
- Modelos entrenados (formato pickle)
- Configuraciones de experimentos
- Reportes de calidad de datos

## Instalación y Configuración

### Prerrequisitos
- Python 3.7+
- MLflow 2.0+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.20+

### Instalación Automática
```bash
python setup_mlflow_environment.py
```

### Instalación Manual
```bash
pip install mlflow scikit-learn pandas numpy boto3 requests matplotlib seaborn
mkdir -p mlruns mlflow_artifacts logs reports models
```

## Ejecución de Experimentos

### Paso 1: Configurar Entorno
```bash
python setup_mlflow_environment.py
```

### Paso 2: Iniciar MLflow UI
```bash
./start_mlflow.sh
# o manualmente:
mlflow ui --host 0.0.0.0 --port 5000
```

### Paso 3: Ejecutar Experimentos
```bash
python mlflow_experiments_final.py
```

### Paso 4: Verificar Resultados
- MLflow UI: `http://localhost:5000` (local) o `http://EC2_IP:5000` (EC2)
- Logs: `mlflow_experiments.log`
- Reportes: `REPORTE_EXPERIMENTOS_MLFLOW.txt`

## Estructura de Archivos

```
proyecto/
├── mlflow_experiments_final.py     # Script principal
├── setup_mlflow_environment.py     # Configuración automática
├── resumen por item final.xlsx     # Dataset
├── start_mlflow.sh                 # Iniciar MLflow UI
├── stop_mlflow.sh                  # Detener MLflow UI
├── run_experiments.sh              # Ejecutor de experimentos
├── mlruns/                         # Backend store MLflow
├── mlflow_artifacts/               # Artifacts storage
├── logs/                           # Logs de ejecución
├── reports/                        # Reportes generados
└── models/                         # Modelos guardados
```

## Salidas del Sistema

### Archivos Generados
- `mlflow_experiments.log`: Log detallado de ejecución
- `system_info.json`: Información del sistema
- `data_quality_report.json`: Reporte de calidad de datos
- `experiment_summary.json`: Resumen ejecutivo
- `REPORTE_EXPERIMENTOS_MLFLOW.txt`: Reporte principal

### MLflow UI Components
- **Experiments**: Lista de todos los experimentos
- **Runs**: Ejecuciones individuales con métricas y parámetros
- **Models**: Registro de modelos entrenados
- **Artifacts**: Archivos y objetos generados

## Mejores Prácticas Implementadas

### Reproducibilidad
- Semilla aleatoria fija (42)
- Versionado de código
- Tracking completo de parámetros

### Validación Robusta
- Validación cruzada estratificada
- División train/test estratificada
- Múltiples métricas de evaluación

### Escalabilidad
- Gestión automática de memoria
- Límites en combinaciones de parámetros
- Logging estructurado

### Calidad de Código
- Documentación completa
- Manejo de errores
- Validaciones de entrada

## Troubleshooting

### Problemas Comunes

1. **Puerto 5000 ocupado**
   ```bash
   ./stop_mlflow.sh
   ./start_mlflow.sh
   ```

2. **Archivo de datos no encontrado**
   ```bash
   ls -la *.xlsx
   # Verificar nombre exacto del archivo
   ```

3. **Errores de permisos en EC2**
   ```bash
   sudo ufw allow 5000
   chmod +x *.sh
   ```

4. **MLflow UI no accesible desde exterior**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   # Verificar security groups en EC2
   ```

## Consideraciones de Seguridad

### EC2 Configuration
- Abrir puerto 5000 en security groups
- Configurar firewall local (ufw)
- Usar HTTPS en producción

### Data Privacy
- No incluir datos sensibles en tracking
- Usar artifact store seguro
- Implementar control de acceso

## Métricas de Performance

### Tiempo de Ejecución Estimado
- Setup: 2-5 minutos
- Clustering experiments: 10-15 minutos
- Classification experiments: 15-25 minutos
- Total: 30-45 minutos

### Recursos Requeridos
- RAM: 4GB mínimo, 8GB recomendado
- CPU: 2 cores mínimo, 4 cores recomendado
- Disco: 5GB mínimo para artifacts

## Extensiones Futuras

### Algoritmos Adicionales
- Neural Networks
- XGBoost
- CatBoost
- Ensemble methods

### Features Avanzadas
- Automated hyperparameter tuning
- Model serving
- A/B testing framework
- Real-time monitoring

### Integrations
- Apache Airflow para scheduling
- Docker para containerización
- Kubernetes para escalabilidad
- AWS SageMaker para producción