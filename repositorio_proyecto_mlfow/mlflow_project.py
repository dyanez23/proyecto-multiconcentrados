"""
MLflow Experiments - Segmentación y Clasificación de Clientes
============================================================

Script profesional para tracking de experimentos de machine learning
con modelos de clustering y clasificación usando datos reales.

Autor: Equipo Analítica
Fecha: Julio 2025
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    silhouette_score, accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')
np.random.seed(42)

print("MLflow Experiments - Segmentación de Clientes")
print("=" * 55)

class DataLoader:
    """Carga y prepara datos para experimentos ML."""
    
    def __init__(self, excel_path="resumen por item final.xlsx"):
        self.excel_path = excel_path
        self.df = None
        self.df_clustering = None
        self.df_classification = None
        
    def load_and_prepare_data(self):
        """Carga datos desde Excel y prepara datasets."""
        print("Cargando datos desde Excel...")
        
        try:
            # Cargar datos
            self.df = pd.read_excel(
                self.excel_path,
                sheet_name=0,
                dtype={'CLIENTE': str, 'CODIGO': str}
            )
            
            # Renombrar columnas
            self.df.columns = [
                "anio", "mes", "cliente", "codigo_producto", "nombre_producto",
                "unidad_medida", "cantidad", "valor_unitario", "descuento_total",
                "valor_total", "num_factura", "cc", "cat1", "cat2", "cat3",
                "categoria", "departamento", "municipio"
            ]
            
            # Limpiar datos
            self.df.drop(columns="cc", inplace=True)
            self.df["mes"] = self.df["mes"].str.strip().str.capitalize()
            
            # Filtrar clientes reales (excluir mostrador)
            cliente_mostrador = "000022222222"
            clientes_reales = self.df[self.df["cliente"] != cliente_mostrador]
            
            print(f"Datos cargados: {len(self.df):,} registros")
            print(f"Clientes reales: {clientes_reales['cliente'].nunique():,}")
            
            # Preparar datos para clustering
            self._prepare_clustering_data(clientes_reales)
            
            # Preparar datos para clasificación
            self._prepare_classification_data(clientes_reales)
            
            return True
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def _prepare_clustering_data(self, clientes_reales):
        """Prepara datos agregados por cliente para clustering."""
        print("Preparando datos clustering...")
        
        # Agregaciones por cliente
        aggregations = {
            'valor_total': 'sum',
            'cantidad': 'sum',
            'num_factura': 'sum',
            'codigo_producto': 'nunique',
            'categoria': 'nunique',
            'municipio': lambda x: x.mode().iloc[0],
            'departamento': lambda x: x.mode().iloc[0]
        }
        
        df_clustering = clientes_reales.groupby('cliente').agg(aggregations).reset_index()
        
        # Renombrar columnas
        df_clustering.columns = [
            'cliente', 'total_gastado', 'cantidad_total', 'total_facturas',
            'num_productos_distintos', 'num_categorias_distintas',
            'municipio_principal', 'departamento_principal'
        ]
        
        # Variables numéricas para normalización
        numeric_cols = [
            'total_gastado', 'cantidad_total', 'total_facturas',
            'num_productos_distintos', 'num_categorias_distintas'
        ]
        
        # Normalizar datos
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(df_clustering[numeric_cols])
        
        # One-hot encoding para variables categóricas
        X_categorical = pd.get_dummies(
            df_clustering[['municipio_principal', 'departamento_principal']],
            drop_first=True
        ).astype(int)
        
        # Combinar características
        X_combined = np.hstack([X_numeric, X_categorical.values])
        
        self.df_clustering = {
            'X': X_combined,
            'feature_names': numeric_cols + list(X_categorical.columns),
            'cliente_ids': df_clustering['cliente'].values,
            'scaler': scaler,
            'raw_data': df_clustering
        }
        
        print(f"Clustering dataset: {X_combined.shape}")
    
    def _prepare_classification_data(self, clientes_reales):
        """Prepara datos para clasificación de clientes frecuentes."""
        print("Preparando datos clasificación...")
        
        # Mapeo de meses
        meses_dict = {m: i for i, m in enumerate([
            "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ], start=1)}
        
        # Crear fechas
        clientes_reales = clientes_reales.copy()
        clientes_reales["mes_num"] = clientes_reales["mes"].map(meses_dict)
        clientes_reales["fecha"] = pd.to_datetime(
            clientes_reales["anio"] + "-" + clientes_reales["mes_num"].astype(str).str.zfill(2) + "-01"
        )
        
        # Calcular métricas por cliente
        cliente_stats = clientes_reales.groupby("cliente").agg({
            'fecha': 'nunique',
            'cantidad': 'sum',
            'valor_total': 'sum',
            'nombre_producto': 'nunique',
            'categoria': 'nunique',
            'municipio': 'nunique'
        }).reset_index()
        
        cliente_stats.columns = [
            'cliente', 'frecuencia_meses', 'total_compras', 'total_gastado',
            'num_productos', 'num_categorias', 'num_municipios'
        ]
        
        self.df_classification = cliente_stats
        print(f"Classification dataset: {cliente_stats.shape}")


class MLflowExperiments:
    """Maneja experimentos MLflow para clustering y clasificación."""
    
    def __init__(self, experiment_name="segmentacion_clientes_entrega2"):
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        
    def setup_mlflow(self):
        """Configura tracking MLflow."""
        print("Configurando MLflow...")
        
        # Configurar tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Crear experimento
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(experiment_id=self.experiment_id)
            self.client = MlflowClient()
            
            print(f"Experimento configurado: {self.experiment_id}")
            return True
            
        except Exception as e:
            print(f"Error configurando MLflow: {e}")
            return False
    
    def run_clustering_experiments(self, data):
        """Ejecuta experimentos de clustering."""
        print("\nEjecutando experimentos clustering...")
        
        X = data['X']
        run_ids = {}
        
        # K-Means experiments
        print("K-Means experiments...")
        for k in range(2, 8):
            with mlflow.start_run(run_name=f"kmeans_k{k}"):
                # Entrenar modelo
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Métricas
                silhouette = silhouette_score(X, labels)
                inertia = kmeans.inertia_
                
                # Log parámetros y métricas
                mlflow.log_param("algorithm", "kmeans")
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("random_state", 42)
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("inertia", inertia)
                
                # Log modelo
                mlflow.sklearn.log_model(kmeans, "model")
                
                # Tags
                mlflow.set_tags({
                    "model_type": "clustering",
                    "algorithm": "kmeans",
                    "experiment_phase": "hyperparameter_tuning"
                })
                
                run_ids[f"kmeans_k{k}"] = mlflow.active_run().info.run_id
                print(f"  K={k}: Silhouette={silhouette:.4f}")
        
        # Hierarchical clustering experiments  
        print("Clustering jerárquico experiments...")
        for k in range(2, 8):
            with mlflow.start_run(run_name=f"hierarchical_k{k}"):
                # Entrenar modelo
                hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = hierarchical.fit_predict(X)
                
                # Métricas
                silhouette = silhouette_score(X, labels)
                
                # Log parámetros y métricas
                mlflow.log_param("algorithm", "hierarchical")
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("linkage", "ward")
                mlflow.log_metric("silhouette_score", silhouette)
                
                # Tags
                mlflow.set_tags({
                    "model_type": "clustering",
                    "algorithm": "hierarchical",
                    "experiment_phase": "hyperparameter_tuning"
                })
                
                run_ids[f"hierarchical_k{k}"] = mlflow.active_run().info.run_id
                print(f"  K={k}: Silhouette={silhouette:.4f}")
        
        return run_ids
    
    def run_classification_experiments(self, data, frequency_threshold=3):
        """Ejecuta experimentos de clasificación."""
        print("\nEjecutando experimentos clasificación...")
        
        # Preparar variables
        data['cliente_frecuente'] = (data['frecuencia_meses'] >= frequency_threshold).astype(int)
        
        feature_cols = ['total_compras', 'total_gastado', 'num_productos', 'num_categorias', 'num_municipios']
        X = data[feature_cols]
        y = data['cliente_frecuente']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        run_ids = {}
        
        # Logistic Regression
        print("Regresión Logística experiment...")
        with mlflow.start_run(run_name="logistic_regression_gridsearch"):
            # GridSearch
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            lr = LogisticRegression(random_state=42, max_iter=1000)
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Predicciones y métricas
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
            
            # Log parámetros y métricas
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("algorithm", "logistic_regression")
            mlflow.log_param("frequency_threshold", frequency_threshold)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Log modelo
            mlflow.sklearn.log_model(best_model, "model")
            
            # Tags
            mlflow.set_tags({
                "model_type": "classification",
                "algorithm": "logistic_regression",
                "experiment_phase": "hyperparameter_tuning"
            })
            
            run_ids["logistic_regression"] = mlflow.active_run().info.run_id
            print(f"  Accuracy={accuracy:.4f}, AUC-ROC={auc_roc:.4f}")
        
        # Decision Tree
        print("Árbol de Decisión experiment...")
        with mlflow.start_run(run_name="decision_tree_gridsearch"):
            # GridSearch
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Predicciones y métricas
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
            
            # Log parámetros y métricas
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("algorithm", "decision_tree")
            mlflow.log_param("frequency_threshold", frequency_threshold)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Feature importance
            for feature, importance in zip(feature_cols, best_model.feature_importances_):
                mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Log modelo
            mlflow.sklearn.log_model(best_model, "model")
            
            # Tags
            mlflow.set_tags({
                "model_type": "classification",
                "algorithm": "decision_tree",
                "experiment_phase": "hyperparameter_tuning"
            })
            
            run_ids["decision_tree"] = mlflow.active_run().info.run_id
            print(f"  Accuracy={accuracy:.4f}, AUC-ROC={auc_roc:.4f}")
        
        return run_ids
    
    def create_comparison_experiment(self):
        """Crea experimento de comparación de modelos."""
        print("\nCreando experimento comparación...")
        
        with mlflow.start_run(run_name="model_comparison_summary"):
            # Obtener mejores runs de cada tipo
            runs = self.client.search_runs([self.experiment_id])
            
            best_clustering = None
            best_classification = None
            best_clustering_score = -1
            best_classification_score = -1
            
            for run in runs:
                tags = run.data.tags
                metrics = run.data.metrics
                
                if tags.get("model_type") == "clustering":
                    silhouette = metrics.get("silhouette_score", 0)
                    if silhouette > best_clustering_score:
                        best_clustering_score = silhouette
                        best_clustering = run
                
                elif tags.get("model_type") == "classification":
                    auc = metrics.get("auc_roc", 0)
                    if auc > best_classification_score:
                        best_classification_score = auc
                        best_classification = run
            
            # Log mejores métricas
            if best_clustering:
                mlflow.log_metric("best_clustering_silhouette", best_clustering_score)
                mlflow.log_param("best_clustering_algorithm", best_clustering.data.tags.get("algorithm"))
            
            if best_classification:
                mlflow.log_metric("best_classification_auc", best_classification_score)
                mlflow.log_param("best_classification_algorithm", best_classification.data.tags.get("algorithm"))
            
            # Tags
            mlflow.set_tags({
                "experiment_type": "comparison",
                "total_experiments": len(runs),
                "status": "completed"
            })
            
            print(f"Mejor clustering: {best_clustering_score:.4f}")
            print(f"Mejor clasificación: {best_classification_score:.4f}")


def generate_reports(experiment_id):
    """Genera reportes de experimentos."""
    print("\nGenerando reportes...")
    
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
        "experiment_id": experiment_id
    }
    
    # Guardar información del sistema
    with open("mlflow_system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    # Reporte principal
    report = f"""
REPORTE MLFLOW EXPERIMENTS - SEGMENTACIÓN CLIENTES
=================================================

INFORMACIÓN DEL SISTEMA:
-----------------------
Timestamp: {system_info['timestamp']}
Hostname: {system_info['hostname']}
IP: {system_info['public_ip']}
Environment: {system_info['environment']}
Experiment ID: {experiment_id}

EXPERIMENTOS EJECUTADOS:
-----------------------
- Clustering K-Means (k=2-7)
- Clustering Jerárquico (k=2-7)  
- Clasificación Regresión Logística (GridSearch)
- Clasificación Árbol de Decisión (GridSearch)
- Experimento de Comparación

ACCESO MLFLOW UI:
----------------
Local: http://localhost:5000
EC2: http://{public_ip}:5000

COMANDOS:
--------
Iniciar MLflow UI: mlflow ui --host 0.0.0.0 --port 5000
"""
    
    with open("REPORTE_MLFLOW_EXPERIMENTS.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("Reportes generados exitosamente")


def main():
    """Función principal."""
    # Verificar archivo de datos
    data_file = "resumen por item final.xlsx"
    if not os.path.exists(data_file):
        print(f"Error: Archivo '{data_file}' no encontrado")
        print("Archivos Excel disponibles:")
        for f in os.listdir('.'):
            if f.endswith('.xlsx'):
                print(f"  - {f}")
        return False
    
    # Cargar datos
    data_loader = DataLoader(data_file)
    if not data_loader.load_and_prepare_data():
        print("Error cargando datos")
        return False
    
    # Configurar MLflow
    ml_experiments = MLflowExperiments()
    if not ml_experiments.setup_mlflow():
        print("Error configurando MLflow")
        return False
    
    # Ejecutar experimentos
    clustering_runs = ml_experiments.run_clustering_experiments(data_loader.df_clustering)
    classification_runs = ml_experiments.run_classification_experiments(data_loader.df_classification)
    
    # Experimento de comparación
    ml_experiments.create_comparison_experiment()
    
    # Generar reportes
    generate_reports(ml_experiments.experiment_id)
    
    print("\n" + "=" * 55)
    print("EXPERIMENTOS COMPLETADOS")
    print("=" * 55)
    print(f"Total clustering runs: {len(clustering_runs)}")
    print(f"Total classification runs: {len(classification_runs)}")
    print(f"Experiment ID: {ml_experiments.experiment_id}")
    print("MLflow UI: http://localhost:5000")
    print("=" * 55)
    
    return True


if __name__ == "__main__":
    print("Iniciando experimentos MLflow...")
    
    success = main()
    
    if success:
        print("\nEjecución exitosa")
        print("Para acceder MLflow UI ejecuta: mlflow ui --host 0.0.0.0 --port 5000")
    else:
        print("\nError en ejecución")
        sys.exit(1)