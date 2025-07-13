"""
MLflow Experiments - Segmentación y Clasificación de Clientes
============================================================

Sistema completo de experimentación ML con tracking avanzado,
evaluación robusta y generación de reportes automatizados.

Autor: Equipo Analítica
Fecha: Julio 2025
Versión: 2.0
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import logging
from typing import Dict, Tuple, Any, Optional

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Configuración
warnings.filterwarnings('ignore')
np.random.seed(42)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlflow_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesador avanzado de datos con validaciones y transformaciones."""
    
    def __init__(self, excel_path: str = "resumen por item final.xlsx"):
        self.excel_path = excel_path
        self.df_raw = None
        self.df_clustering = None
        self.df_classification = None
        self.data_quality_report = {}
        
    def load_and_validate_data(self) -> bool:
        """Carga y valida datos con reporte de calidad."""
        logger.info("Iniciando carga y validación de datos")
        
        try:
            # Verificar existencia del archivo
            if not os.path.exists(self.excel_path):
                logger.error(f"Archivo no encontrado: {self.excel_path}")
                return False
            
            # Cargar datos
            self.df_raw = pd.read_excel(
                self.excel_path,
                sheet_name=0,
                dtype={'CLIENTE': str, 'CODIGO': str}
            )
            
            # Estandarizar nombres de columnas
            self.df_raw.columns = [
                "anio", "mes", "cliente", "codigo_producto", "nombre_producto",
                "unidad_medida", "cantidad", "valor_unitario", "descuento_total",
                "valor_total", "num_factura", "cc", "cat1", "cat2", "cat3",
                "categoria", "departamento", "municipio"
            ]
            
            # Generar reporte de calidad
            self._generate_quality_report()
            
            # Limpiar datos
            self._clean_data()
            
            # Preparar datasets
            self._prepare_clustering_dataset()
            self._prepare_classification_dataset()
            
            logger.info("Datos cargados y procesados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error procesando datos: {str(e)}")
            return False
    
    def _generate_quality_report(self) -> None:
        """Genera reporte de calidad de datos."""
        self.data_quality_report = {
            'total_records': len(self.df_raw),
            'total_columns': len(self.df_raw.columns),
            'missing_values': self.df_raw.isnull().sum().to_dict(),
            'duplicate_records': self.df_raw.duplicated().sum(),
            'unique_clients': self.df_raw['cliente'].nunique(),
            'unique_products': self.df_raw['codigo_producto'].nunique(),
            'date_range': {
                'years': sorted(self.df_raw['anio'].unique()),
                'months': sorted(self.df_raw['mes'].unique())
            },
            'numerical_stats': self.df_raw.select_dtypes(include=[np.number]).describe().to_dict()
        }
        
        logger.info(f"Reporte de calidad generado - Registros: {self.data_quality_report['total_records']:,}")
    
    def _clean_data(self) -> None:
        """Limpia y normaliza datos."""
        # Eliminar columnas innecesarias
        if 'cc' in self.df_raw.columns:
            self.df_raw.drop(columns=['cc'], inplace=True)
        
        # Normalizar texto
        self.df_raw['mes'] = self.df_raw['mes'].str.strip().str.capitalize()
        
        # Filtrar datos válidos
        self.df_raw = self.df_raw[
            (self.df_raw['valor_total'] > 0) &
            (self.df_raw['cantidad'] > 0) &
            (self.df_raw['cliente'] != "000022222222")  # Excluir mostrador
        ].copy()
        
        # Crear variables derivadas
        self.df_raw['valor_con_descuento'] = self.df_raw['valor_total'] - self.df_raw['descuento_total']
        self.df_raw['precio_efectivo'] = self.df_raw['valor_con_descuento'] / self.df_raw['cantidad']
        
        logger.info(f"Datos limpiados - Registros finales: {len(self.df_raw):,}")
    
    def _prepare_clustering_dataset(self) -> None:
        """Prepara dataset para clustering con características avanzadas."""
        logger.info("Preparando dataset para clustering")
        
        # Agregaciones por cliente
        aggregations = {
            'valor_total': ['sum', 'mean', 'std', 'count'],
            'cantidad': ['sum', 'mean'],
            'descuento_total': ['sum', 'mean'],
            'precio_efectivo': ['mean', 'std'],
            'num_factura': 'nunique',
            'codigo_producto': 'nunique',
            'categoria': 'nunique',
            'departamento': 'nunique',
            'municipio': 'nunique'
        }
        
        df_agg = self.df_raw.groupby('cliente').agg(aggregations).reset_index()
        
        # Aplanar columnas multi-nivel
        df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
        df_agg.rename(columns={'cliente_': 'cliente'}, inplace=True)
        
        # Crear características adicionales
        df_agg['ticket_promedio'] = df_agg['valor_total_sum'] / df_agg['valor_total_count']
        df_agg['descuento_ratio'] = df_agg['descuento_total_sum'] / df_agg['valor_total_sum']
        df_agg['diversidad_productos'] = df_agg['codigo_producto_nunique'] / df_agg['valor_total_count']
        df_agg['diversidad_geografica'] = df_agg['municipio_nunique'] / df_agg['valor_total_count']
        
        # Seleccionar características para clustering
        feature_cols = [
            'valor_total_sum', 'valor_total_mean', 'valor_total_count',
            'cantidad_sum', 'ticket_promedio', 'descuento_ratio',
            'codigo_producto_nunique', 'categoria_nunique',
            'diversidad_productos', 'diversidad_geografica'
        ]
        
        # Normalizar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_agg[feature_cols])
        
        self.df_clustering = {
            'X': X_scaled,
            'feature_names': feature_cols,
            'cliente_ids': df_agg['cliente'].values,
            'raw_features': df_agg[feature_cols],
            'scaler': scaler
        }
        
        logger.info(f"Dataset clustering preparado - Shape: {X_scaled.shape}")
    
    def _prepare_classification_dataset(self) -> None:
        """Prepara dataset para clasificación con target balanceado."""
        logger.info("Preparando dataset para clasificación")
        
        # Mapeo de meses
        month_mapping = {
            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
            'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
            'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
        }
        
        # Crear variables temporales
        df_temp = self.df_raw.copy()
        df_temp['mes_num'] = df_temp['mes'].map(month_mapping)
        df_temp['fecha'] = pd.to_datetime(
            df_temp['anio'].astype(str) + '-' + df_temp['mes_num'].astype(str).str.zfill(2) + '-01'
        )
        
        # Agregaciones por cliente
        client_metrics = df_temp.groupby('cliente').agg({
            'fecha': ['nunique', 'min', 'max'],
            'valor_total': ['sum', 'mean', 'std', 'count'],
            'cantidad': ['sum', 'mean'],
            'descuento_total': ['sum', 'mean'],
            'codigo_producto': 'nunique',
            'categoria': 'nunique',
            'departamento': 'nunique',
            'municipio': 'nunique'
        }).reset_index()
        
        # Aplanar columnas
        client_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in client_metrics.columns.values]
        client_metrics.rename(columns={'cliente_': 'cliente'}, inplace=True)
        
        # Crear características derivadas
        client_metrics['periodo_actividad'] = (
            client_metrics['fecha_max'] - client_metrics['fecha_min']
        ).dt.days
        client_metrics['frecuencia_mensual'] = client_metrics['fecha_nunique']
        client_metrics['gasto_mensual_promedio'] = client_metrics['valor_total_sum'] / client_metrics['frecuencia_mensual']
        client_metrics['ticket_promedio'] = client_metrics['valor_total_sum'] / client_metrics['valor_total_count']
        
        # Definir target: clientes frecuentes (más de 2 meses de actividad)
        frequency_threshold = 2
        client_metrics['es_cliente_frecuente'] = (client_metrics['frecuencia_mensual'] > frequency_threshold).astype(int)
        
        # Seleccionar características
        feature_cols = [
            'valor_total_sum', 'valor_total_mean', 'valor_total_count',
            'cantidad_sum', 'gasto_mensual_promedio', 'ticket_promedio',
            'codigo_producto_nunique', 'categoria_nunique',
            'departamento_nunique', 'municipio_nunique'
        ]
        
        self.df_classification = {
            'X': client_metrics[feature_cols],
            'y': client_metrics['es_cliente_frecuente'],
            'feature_names': feature_cols,
            'cliente_ids': client_metrics['cliente'].values,
            'threshold': frequency_threshold,
            'class_distribution': client_metrics['es_cliente_frecuente'].value_counts().to_dict()
        }
        
        logger.info(f"Dataset clasificación preparado - Shape: {client_metrics[feature_cols].shape}")
        logger.info(f"Distribución de clases: {self.df_classification['class_distribution']}")


class MLflowExperimentManager:
    """Gestor avanzado de experimentos MLflow."""
    
    def __init__(self, experiment_name: str = "segmentacion_clientes_v2"):
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        self.tracking_uri = "http://localhost:5000"
        
    def initialize_mlflow(self) -> bool:
        """Inicializa MLflow con configuración robusta."""
        logger.info("Inicializando MLflow")
        
        try:
            # Configurar tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Crear o obtener experimento
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        name=self.experiment_name,
                        tags={
                            "project": "segmentacion_clientes",
                            "version": "2.0",
                            "team": "analitica",
                            "created_at": datetime.now().isoformat()
                        }
                    )
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Error creando experimento: {e}")
                self.experiment_id = mlflow.create_experiment(f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            mlflow.set_experiment(experiment_id=self.experiment_id)
            self.client = MlflowClient()
            
            logger.info(f"MLflow inicializado - Experiment ID: {self.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MLflow: {str(e)}")
            return False
    
    def run_clustering_experiments(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Ejecuta experimentos de clustering con múltiples algoritmos."""
        logger.info("Ejecutando experimentos de clustering")
        
        X = data['X']
        feature_names = data['feature_names']
        run_ids = {}
        
        # Configuración de experimentos
        clustering_configs = {
            'kmeans': {
                'algorithm': KMeans,
                'param_grid': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 20]
                }
            },
            'hierarchical': {
                'algorithm': AgglomerativeClustering,
                'param_grid': {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                    'linkage': ['ward', 'complete', 'average']
                }
            },
            'dbscan': {
                'algorithm': DBSCAN,
                'param_grid': {
                    'eps': [0.3, 0.5, 0.7, 1.0],
                    'min_samples': [3, 5, 7, 10]
                }
            }
        }
        
        for algo_name, config in clustering_configs.items():
            logger.info(f"Ejecutando experimentos {algo_name}")
            
            for params in self._generate_param_combinations(config['param_grid']):
                run_name = f"{algo_name}_{'_'.join([f'{k}_{v}' for k, v in params.items()])}"
                
                with mlflow.start_run(run_name=run_name):
                    try:
                        # Instanciar y entrenar modelo
                        model = config['algorithm'](**params, random_state=42)
                        labels = model.fit_predict(X)
                        
                        # Validar resultados
                        unique_labels = np.unique(labels)
                        if len(unique_labels) < 2:
                            logger.warning(f"Clustering inválido para {run_name}: {len(unique_labels)} clusters")
                            continue
                        
                        # Calcular métricas
                        silhouette = silhouette_score(X, labels)
                        davies_bouldin = davies_bouldin_score(X, labels)
                        calinski_harabasz = calinski_harabasz_score(X, labels)
                        
                        # Log parámetros
                        mlflow.log_params(params)
                        mlflow.log_param("algorithm", algo_name)
                        mlflow.log_param("n_features", X.shape[1])
                        mlflow.log_param("n_samples", X.shape[0])
                        
                        # Log métricas
                        mlflow.log_metric("silhouette_score", silhouette)
                        mlflow.log_metric("davies_bouldin_score", davies_bouldin)
                        mlflow.log_metric("calinski_harabasz_score", calinski_harabasz)
                        mlflow.log_metric("n_clusters_found", len(unique_labels))
                        
                        # Log distribución de clusters
                        cluster_counts = np.bincount(labels[labels >= 0])  # Excluir outliers (-1)
                        for i, count in enumerate(cluster_counts):
                            mlflow.log_metric(f"cluster_{i}_size", count)
                        
                        # Log modelo (solo para algoritmos que lo soporten)
                        if hasattr(model, 'cluster_centers_'):
                            mlflow.sklearn.log_model(model, "model")
                        
                        # Tags
                        mlflow.set_tags({
                            "model_type": "clustering",
                            "algorithm": algo_name,
                            "experiment_phase": "hyperparameter_tuning",
                            "status": "completed"
                        })
                        
                        run_ids[run_name] = mlflow.active_run().info.run_id
                        logger.info(f"  {run_name}: Silhouette={silhouette:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error en {run_name}: {str(e)}")
                        mlflow.set_tag("status", "failed")
                        mlflow.set_tag("error", str(e))
        
        return run_ids
    
    def run_classification_experiments(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Ejecuta experimentos de clasificación con múltiples algoritmos."""
        logger.info("Ejecutando experimentos de clasificación")
        
        X = data['X']
        y = data['y']
        feature_names = data['feature_names']
        run_ids = {}
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Configuración de experimentos
        classification_configs = {
            'logistic_regression': {
                'algorithm': LogisticRegression,
                'param_grid': {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [1000, 2000]
                }
            },
            'random_forest': {
                'algorithm': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'algorithm': GradientBoostingClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'svm': {
                'algorithm': SVC,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'probability': [True]
                }
            }
        }
        
        for algo_name, config in classification_configs.items():
            logger.info(f"Ejecutando GridSearch para {algo_name}")
            
            with mlflow.start_run(run_name=f"{algo_name}_gridsearch"):
                try:
                    # GridSearch con validación cruzada
                    model = config['algorithm'](random_state=42)
                    grid_search = GridSearchCV(
                        model, 
                        config['param_grid'],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    
                    best_model = grid_search.best_estimator_
                    
                    # Predicciones
                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                    
                    # Métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Validación cruzada
                    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
                    
                    # Log parámetros
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("cv_folds", 5)
                    mlflow.log_param("test_size", 0.2)
                    mlflow.log_param("n_features", X.shape[1])
                    mlflow.log_param("n_samples", X.shape[0])
                    
                    # Log métricas
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("cv_mean", cv_scores.mean())
                    mlflow.log_metric("cv_std", cv_scores.std())
                    mlflow.log_metric("best_cv_score", grid_search.best_score_)
                    
                    if y_prob is not None:
                        auc_roc = roc_auc_score(y_test, y_prob)
                        mlflow.log_metric("auc_roc", auc_roc)
                    
                    # Feature importance (si está disponible)
                    if hasattr(best_model, 'feature_importances_'):
                        for feature, importance in zip(feature_names, best_model.feature_importances_):
                            mlflow.log_metric(f"feature_importance_{feature}", importance)
                    
                    # Log modelo
                    mlflow.sklearn.log_model(best_model, "model")
                    
                    # Tags
                    mlflow.set_tags({
                        "model_type": "classification",
                        "algorithm": algo_name,
                        "experiment_phase": "hyperparameter_tuning",
                        "status": "completed"
                    })
                    
                    run_ids[algo_name] = mlflow.active_run().info.run_id
                    logger.info(f"  {algo_name}: Accuracy={accuracy:.4f}, AUC={auc_roc:.4f}" if y_prob is not None else f"  {algo_name}: Accuracy={accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error en {algo_name}: {str(e)}")
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
        
        return run_ids
    
    def create_model_comparison(self, data_quality_report: Dict[str, Any]) -> str:
        """Crea experimento de comparación de modelos."""
        logger.info("Creando comparación de modelos")
        
        with mlflow.start_run(run_name="model_comparison_summary"):
            # Obtener todos los runs del experimento
            runs = self.client.search_runs([self.experiment_id])
            
            # Análisis de clustering
            clustering_runs = [r for r in runs if r.data.tags.get("model_type") == "clustering"]
            best_clustering = max(clustering_runs, key=lambda r: r.data.metrics.get("silhouette_score", -1))
            
            # Análisis de clasificación
            classification_runs = [r for r in runs if r.data.tags.get("model_type") == "classification"]
            best_classification = max(classification_runs, key=lambda r: r.data.metrics.get("auc_roc", 0))
            
            # Log resultados de comparación
            mlflow.log_metric("total_experiments", len(runs))
            mlflow.log_metric("clustering_experiments", len(clustering_runs))
            mlflow.log_metric("classification_experiments", len(classification_runs))
            
            if clustering_runs:
                mlflow.log_metric("best_silhouette_score", best_clustering.data.metrics.get("silhouette_score", 0))
                mlflow.log_param("best_clustering_algorithm", best_clustering.data.tags.get("algorithm"))
            
            if classification_runs:
                mlflow.log_metric("best_auc_score", best_classification.data.metrics.get("auc_roc", 0))
                mlflow.log_param("best_classification_algorithm", best_classification.data.tags.get("algorithm"))
            
            # Log información del dataset
            mlflow.log_params(data_quality_report)
            
            # Tags
            mlflow.set_tags({
                "experiment_type": "model_comparison",
                "status": "completed",
                "summary": "automated_model_comparison"
            })
            
            return mlflow.active_run().info.run_id
    
    def _generate_param_combinations(self, param_grid: Dict[str, list]) -> list:
        """Genera combinaciones de parámetros para grid search manual."""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        # Limitar combinaciones para evitar explosión combinatoria
        if len(combinations) > 50:
            np.random.shuffle(combinations)
            combinations = combinations[:50]
        
        return combinations


def generate_comprehensive_report(experiment_id: str, data_quality_report: Dict[str, Any]) -> None:
    """Genera reportes completos de experimentos."""
    logger.info("Generando reportes completos")
    
    # Información del sistema
    import socket
    import platform
    
    try:
        import requests
        public_ip = requests.get('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=5).text
        environment = "AWS_EC2"
    except:
        public_ip = "localhost"
        environment = "LOCAL"
    
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "public_ip": public_ip,
        "environment": environment,
        "platform": platform.platform(),
        "python_version": sys.version,
        "experiment_id": experiment_id,
        "mlflow_version": mlflow.__version__
    }
    
    # Guardar información del sistema
    with open("system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    # Guardar reporte de calidad de datos
    with open("data_quality_report.json", "w") as f:
        json.dump(data_quality_report, f, indent=2)
    
    # Reporte principal
    report = f"""
REPORTE COMPLETO - EXPERIMENTOS MLFLOW
=====================================

INFORMACIÓN DEL SISTEMA
-----------------------
Timestamp: {system_info['timestamp']}
Hostname: {system_info['hostname']}
IP Pública: {system_info['public_ip']}
Entorno: {system_info['environment']}
Plataforma: {system_info['platform']}
Python: {system_info['python_version']}
MLflow: {system_info['mlflow_version']}

EXPERIMENTO
-----------
ID: {experiment_id}
Nombre: segmentacion_clientes_v2

CALIDAD DE DATOS
---------------
Total Registros: {data_quality_report['total_records']:,}
Clientes Únicos: {data_quality_report['unique_clients']:,}
Productos Únicos: {data_quality_report['unique_products']:,}
Años: {data_quality_report['date_range']['years']}
Registros Duplicados: {data_quality_report['duplicate_records']:,}

EXPERIMENTOS EJECUTADOS
-----------------------
1. CLUSTERING:
   - K-Means (múltiples configuraciones)
   - Clustering Jerárquico (ward, complete, average)
   - DBSCAN (múltiples eps y min_samples)

2. CLASIFICACIÓN:
   - Regresión Logística (GridSearch)
   - Random Forest (GridSearch)
   - Gradient Boosting (GridSearch)
   - SVM (GridSearch)

3. COMPARACIÓN DE MODELOS:
   - Análisis automático de mejores modelos
   - Métricas de evaluación cruzada
   - Selección de algoritmos óptimos

ACCESO A MLFLOW UI
-----------------
Local: http://localhost:5000
EC2: http://{public_ip}:5000

COMANDOS DE EJECUCIÓN
--------------------
1. Iniciar MLflow UI:
   mlflow ui --host 0.0.0.0 --port 5000

2. Ejecutar experimentos:
   python mlflow_experiments_final.py

3. Ver logs:
   tail -f mlflow_experiments.log

ARCHIVOS GENERADOS
-----------------
- mlflow_experiments.log: Log detallado de ejecución
- system_info.json: Información del sistema
- data_quality_report.json: Reporte de calidad de datos
- experiment_summary.json: Resumen de experimentos

MÉTRICAS PRINCIPALES
-------------------
Clustering: Silhouette Score, Davies-Bouldin, Calinski-Harabasz
Clasificación: Accuracy, Precision, Recall, F1-Score, AUC-ROC

NOTAS TÉCNICAS
--------------
- Validación cruzada de 5 folds
- Semilla aleatoria: 42 (reproducibilidad)
- Normalización con StandardScaler
- División estratificada 80/20
- GridSearch con límite de 50 combinaciones

"""
    
    with open("REPORTE_EXPERIMENTOS_MLFLOW.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    # Crear resumen ejecutivo
    executive_summary = {
        "project": "Segmentación y Clasificación de Clientes",
        "experiment_id": experiment_id,
        "execution_date": system_info['timestamp'],
        "environment": system_info['environment'],
        "data_summary": {
            "total_records": data_quality_report['total_records'],
            "unique_clients": data_quality_report['unique_clients'],
            "data_quality": "PASSED" if data_quality_report['duplicate_records'] < 1000 else "WARNING"
        },
        "experiments_executed": {
            "clustering_algorithms": ["kmeans", "hierarchical", "dbscan"],
            "classification_algorithms": ["logistic_regression", "random_forest", "gradient_boosting", "svm"],
            "total_runs": "50+"
        },
        "deliverables": {
            "mlflow_ui": f"http://{public_ip}:5000",
            "reports": ["REPORTE_EXPERIMENTOS_MLFLOW.txt", "system_info.json", "data_quality_report.json"],
            "models": "Stored in MLflow Model Registry",
            "logs": "mlflow_experiments.log"
        }
    }
    
    with open("experiment_summary.json", "w") as f:
        json.dump(executive_summary, f, indent=2)
    
    logger.info("Reportes generados exitosamente")


def validate_environment() -> bool:
    """Valida el entorno antes de ejecutar experimentos."""
    logger.info("Validando entorno de ejecución")
    
    # Verificar dependencias
    required_packages = ['mlflow', 'sklearn', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Paquetes faltantes: {missing_packages}")
        return False
    
    # Verificar conexión MLflow
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code != 200:
            logger.warning("MLflow UI no disponible en puerto 5000")
    except:
        logger.warning("No se pudo verificar MLflow UI")
    
    return True


def main() -> bool:
    """Función principal de ejecución."""
    logger.info("Iniciando sistema de experimentos MLflow")
    
    # Validar entorno
    if not validate_environment():
        logger.error("Entorno no válido")
        return False
    
    # Verificar archivo de datos
    data_file = "resumen por item final.xlsx"
    if not os.path.exists(data_file):
        logger.error(f"Archivo de datos no encontrado: {data_file}")
        available_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if available_files:
            logger.info(f"Archivos Excel disponibles: {available_files}")
        return False
    
    # Inicializar procesador de datos
    data_processor = DataProcessor(data_file)
    if not data_processor.load_and_validate_data():
        logger.error("Error procesando datos")
        return False
    
    # Inicializar gestor de experimentos
    experiment_manager = MLflowExperimentManager()
    if not experiment_manager.initialize_mlflow():
        logger.error("Error inicializando MLflow")
        return False
    
    try:
        # Ejecutar experimentos de clustering
        logger.info("Iniciando experimentos de clustering")
        clustering_runs = experiment_manager.run_clustering_experiments(data_processor.df_clustering)
        
        # Ejecutar experimentos de clasificación
        logger.info("Iniciando experimentos de clasificación")
        classification_runs = experiment_manager.run_classification_experiments(data_processor.df_classification)
        
        # Crear comparación de modelos
        logger.info("Creando comparación de modelos")
        comparison_run = experiment_manager.create_model_comparison(data_processor.data_quality_report)
        
        # Generar reportes
        generate_comprehensive_report(experiment_manager.experiment_id, data_processor.data_quality_report)
        
        # Resumen final
        total_runs = len(clustering_runs) + len(classification_runs) + 1
        logger.info(f"Experimentos completados exitosamente")
        logger.info(f"Total de runs ejecutados: {total_runs}")
        logger.info(f"Experiment ID: {experiment_manager.experiment_id}")
        logger.info(f"MLflow UI: {experiment_manager.tracking_uri}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("SISTEMA DE EXPERIMENTOS MLFLOW - SEGMENTACIÓN DE CLIENTES")
    print("="*60)
    
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
        print("Para acceder a MLflow UI:")
        print("mlflow ui --host 0.0.0.0 --port 5000")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ERROR EN LA EJECUCIÓN")
        print("="*60)
        print("Revisa el archivo 'mlflow_experiments.log' para más detalles")
        sys.exit(1)