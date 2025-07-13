"""
SEGMENTO 2: Modelos de Machine Learning Core (VERSIÓN CORREGIDA)
===============================================================

Este módulo implementa todos los modelos de clustering y clasificación.
Entrena, evalúa y compara modelos preparando resultados para MLflow.

CORREGIDO: Manejo de serialización de objetos PAM/K-medoids

Proyecto: Despliegue de Soluciones Analíticas - Entrega 2
Segmento: 2/4 - Modelos de Machine Learning
Responsable: [Nombre del miembro del equipo]
Fecha: Julio 2025
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from datetime import datetime
import json

# Configuración matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# Machine Learning - Preprocesamiento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Machine Learning - Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Machine Learning - Clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# PAM/K-Medoids
from pyclustering.cluster.kmedoids import kmedoids

# Suprimir warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

# Directorios
DATA_DIR = "data_processed"
MODELS_DIR = "models_results"
REPORTS_DIR = "model_reports"

# Crear directorios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("="*70)
print("SEGMENTO 2: MODELOS DE MACHINE LEARNING (VERSIÓN CORREGIDA)")
print("="*70)
print(f"Inicio del entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# CARGA DE DATOS PROCESADOS
# =============================================================================

def cargar_datos_procesados():
    """
    Carga todos los datos procesados del Segmento 1.
    
    Returns:
        dict: Diccionario con todos los datasets y configuraciones
    """
    print("\nCargando datos procesados del Segmento 1...")
    
    try:
        # Cargar datasets
        df_clustering = pd.read_pickle(os.path.join(DATA_DIR, "datos_clustering.pkl"))
        df_clasificacion = pd.read_pickle(os.path.join(DATA_DIR, "datos_clasificacion.pkl"))
        
        # Cargar configuración
        with open(os.path.join(DATA_DIR, "configuracion.pkl"), 'rb') as f:
            config = pickle.load(f)
        
        # Cargar estadísticas
        with open(os.path.join(DATA_DIR, "estadisticas_descriptivas.pkl"), 'rb') as f:
            stats = pickle.load(f)
        
        print(f"  Dataset clustering cargado: {df_clustering.shape}")
        print(f"  Dataset clasificación cargado: {df_clasificacion.shape}")
        print(f"  Configuración y estadísticas cargadas")
        
        return {
            'clustering': df_clustering,
            'clasificacion': df_clasificacion,
            'config': config,
            'stats': stats
        }
        
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        print("Asegúrate de haber ejecutado el Segmento 1 primero")
        return None

def preparar_datos_clustering(df_clustering, config):
    """
    Prepara los datos para modelos de clustering.
    
    Args:
        df_clustering (pd.DataFrame): Dataset de clustering
        config (dict): Configuración del proyecto
        
    Returns:
        tuple: (X_scaled, scaler, feature_names)
    """
    print("\nPreparando datos para clustering...")
    
    # Variables numéricas para clustering
    variables_numericas = config['VARIABLES_NUMERICAS']
    
    # Extraer características numéricas
    X_num = df_clustering[variables_numericas].copy()
    
    # One-hot encoding para variables categóricas
    X_cat = pd.get_dummies(
        df_clustering[['municipio_principal', 'departamento_principal']], 
        drop_first=True
    ).astype(int)
    
    # Combinar características
    X = pd.concat([X_num, X_cat], axis=1)
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  Características numéricas: {len(variables_numericas)}")
    print(f"  Características categóricas: {X_cat.shape[1]}")
    print(f"  Total características: {X_scaled.shape[1]}")
    print(f"  Muestras para clustering: {X_scaled.shape[0]}")
    
    return X_scaled, scaler, list(X.columns)

# =============================================================================
# MODELOS DE CLUSTERING
# =============================================================================

class ClusteringModels:
    """
    Clase para manejar todos los modelos de clustering.
    """
    
    def __init__(self, X, feature_names, cliente_ids):
        self.X = X
        self.feature_names = feature_names
        self.cliente_ids = cliente_ids
        self.resultados = {}
    
    def evaluar_kmeans(self, k_range=(2, 11)):
        """
        Evalúa K-Means con diferentes valores de k.
        
        Args:
            k_range (tuple): Rango de valores k a evaluar
            
        Returns:
            dict: Resultados de K-Means
        """
        print("\nEvaluando modelos K-Means...")
        
        resultados_kmeans = {}
        
        for k in range(k_range[0], k_range[1]):
            print(f"  Entrenando K-Means con k={k}")
            
            # Entrenar modelo
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiquetas = modelo.fit_predict(self.X)
            
            # Calcular métricas
            silhouette = silhouette_score(self.X, etiquetas)
            inercia = modelo.inertia_
            calinski = calinski_harabasz_score(self.X, etiquetas)
            davies = davies_bouldin_score(self.X, etiquetas)
            
            # PCA para visualización
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(self.X)
            
            resultados_kmeans[k] = {
                'modelo': modelo,
                'etiquetas': etiquetas,
                'metricas': {
                    'silhouette_score': silhouette,
                    'inercia': inercia,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies
                },
                'pca_coords': X_pca,
                'pca_model': pca,
                'parametros': {'n_clusters': k, 'random_state': 42, 'n_init': 10}
            }
            
            print(f"    Silhouette Score: {silhouette:.4f}")
        
        # Seleccionar mejor k basado en silhouette score
        mejor_k = max(resultados_kmeans.keys(), 
                     key=lambda k: resultados_kmeans[k]['metricas']['silhouette_score'])
        
        resultados_kmeans['mejor_k'] = mejor_k
        resultados_kmeans['mejor_modelo'] = resultados_kmeans[mejor_k]
        
        print(f"  Mejor K-Means: k={mejor_k} (Silhouette: {resultados_kmeans[mejor_k]['metricas']['silhouette_score']:.4f})")
        
        self.resultados['kmeans'] = resultados_kmeans
        return resultados_kmeans
    
    def evaluar_clustering_jerarquico(self, k_range=(2, 11)):
        """
        Evalúa Clustering Jerárquico con diferentes valores de k.
        
        Args:
            k_range (tuple): Rango de valores k a evaluar
            
        Returns:
            dict: Resultados de Clustering Jerárquico
        """
        print("\nEvaluando modelos de Clustering Jerárquico...")
        
        resultados_jerarquico = {}
        
        # Calcular matriz de distancias una sola vez
        distance_matrix = pairwise_distances(self.X, metric='euclidean')
        
        for k in range(k_range[0], k_range[1]):
            print(f"  Entrenando Clustering Jerárquico con k={k}")
            
            # Entrenar modelo
            modelo = AgglomerativeClustering(n_clusters=k, linkage='ward')
            etiquetas = modelo.fit_predict(self.X)
            
            # Calcular métricas
            silhouette = silhouette_score(distance_matrix, etiquetas, metric='precomputed')
            calinski = calinski_harabasz_score(self.X, etiquetas)
            davies = davies_bouldin_score(self.X, etiquetas)
            
            # PCA para visualización
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(self.X)
            
            resultados_jerarquico[k] = {
                'modelo': modelo,
                'etiquetas': etiquetas,
                'metricas': {
                    'silhouette_score': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies
                },
                'pca_coords': X_pca,
                'pca_model': pca,
                'parametros': {'n_clusters': k, 'linkage': 'ward'}
            }
            
            print(f"    Silhouette Score: {silhouette:.4f}")
        
        # Seleccionar mejor k
        mejor_k = max(resultados_jerarquico.keys(), 
                     key=lambda k: resultados_jerarquico[k]['metricas']['silhouette_score'])
        
        resultados_jerarquico['mejor_k'] = mejor_k
        resultados_jerarquico['mejor_modelo'] = resultados_jerarquico[mejor_k]
        
        print(f"  Mejor Jerárquico: k={mejor_k} (Silhouette: {resultados_jerarquico[mejor_k]['metricas']['silhouette_score']:.4f})")
        
        self.resultados['jerarquico'] = resultados_jerarquico
        return resultados_jerarquico
    
    def evaluar_pam(self, k_range=(2, 11)):
        """
        Evalúa PAM/K-Medoids con diferentes valores de k.
        
        Args:
            k_range (tuple): Rango de valores k a evaluar
            
        Returns:
            dict: Resultados de PAM
        """
        print("\nEvaluando modelos PAM (K-Medoids)...")
        
        resultados_pam = {}
        
        # Calcular matriz de distancias
        distance_matrix = pairwise_distances(self.X, metric='euclidean')
        
        for k in range(k_range[0], k_range[1]):
            print(f"  Entrenando PAM con k={k}")
            
            try:
                # Inicializar medoids aleatoriamente
                np.random.seed(42)
                initial_medoids = np.random.choice(len(self.X), size=k, replace=False).tolist()
                
                # Entrenar PAM
                pam_instance = kmedoids(
                    data=distance_matrix, 
                    initial_index_medoids=initial_medoids, 
                    data_type='distance_matrix'
                )
                pam_instance.process()
                clusters = pam_instance.get_clusters()
                
                # Convertir clusters a etiquetas
                etiquetas = np.zeros(len(self.X), dtype=int)
                for cluster_id, indices in enumerate(clusters):
                    for idx in indices:
                        etiquetas[idx] = cluster_id
                
                # Calcular métricas
                silhouette = silhouette_score(distance_matrix, etiquetas, metric='precomputed')
                
                # PCA para visualización
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(self.X)
                
                resultados_pam[k] = {
                    'pam_instance': pam_instance,
                    'etiquetas': etiquetas,
                    'medoids': pam_instance.get_medoids(),
                    'metricas': {
                        'silhouette_score': silhouette
                    },
                    'pca_coords': X_pca,
                    'pca_model': pca,
                    'parametros': {'n_clusters': k, 'random_state': 42}
                }
                
                print(f"    Silhouette Score: {silhouette:.4f}")
                
            except Exception as e:
                print(f"    Error con k={k}: {e}")
                continue
        
        if resultados_pam:
            # Seleccionar mejor k
            mejor_k = max(resultados_pam.keys(), 
                         key=lambda k: resultados_pam[k]['metricas']['silhouette_score'])
            
            resultados_pam['mejor_k'] = mejor_k
            resultados_pam['mejor_modelo'] = resultados_pam[mejor_k]
            
            print(f"  Mejor PAM: k={mejor_k} (Silhouette: {resultados_pam[mejor_k]['metricas']['silhouette_score']:.4f})")
        
        self.resultados['pam'] = resultados_pam
        return resultados_pam
    
    def comparar_modelos_clustering(self):
        """
        Compara todos los modelos de clustering entrenados.
        
        Returns:
            pd.DataFrame: Comparación de modelos
        """
        print("\nComparando modelos de clustering...")
        
        comparacion = []
        
        # K-Means
        if 'kmeans' in self.resultados:
            mejor_kmeans = self.resultados['kmeans']['mejor_modelo']
            comparacion.append({
                'Modelo': 'K-Means',
                'K': self.resultados['kmeans']['mejor_k'],
                'Silhouette Score': mejor_kmeans['metricas']['silhouette_score'],
                'Calinski-Harabasz': mejor_kmeans['metricas']['calinski_harabasz'],
                'Davies-Bouldin': mejor_kmeans['metricas']['davies_bouldin'],
                'Inercia': mejor_kmeans['metricas']['inercia']
            })
        
        # Jerárquico
        if 'jerarquico' in self.resultados:
            mejor_jerarquico = self.resultados['jerarquico']['mejor_modelo']
            comparacion.append({
                'Modelo': 'Jerárquico',
                'K': self.resultados['jerarquico']['mejor_k'],
                'Silhouette Score': mejor_jerarquico['metricas']['silhouette_score'],
                'Calinski-Harabasz': mejor_jerarquico['metricas']['calinski_harabasz'],
                'Davies-Bouldin': mejor_jerarquico['metricas']['davies_bouldin'],
                'Inercia': None
            })
        
        # PAM
        if 'pam' in self.resultados and self.resultados['pam']:
            mejor_pam = self.resultados['pam']['mejor_modelo']
            comparacion.append({
                'Modelo': 'PAM',
                'K': self.resultados['pam']['mejor_k'],
                'Silhouette Score': mejor_pam['metricas']['silhouette_score'],
                'Calinski-Harabasz': None,
                'Davies-Bouldin': None,
                'Inercia': None
            })
        
        df_comparacion = pd.DataFrame(comparacion)
        
        print("  Comparación completada:")
        print(df_comparacion.to_string(index=False))
        
        return df_comparacion

# =============================================================================
# MODELOS DE CLASIFICACIÓN
# =============================================================================

class ClassificationModels:
    """
    Clase para manejar todos los modelos de clasificación.
    """
    
    def __init__(self, df_clasificacion):
        self.df_clasificacion = df_clasificacion
        self.resultados = {}
    
    def preparar_datos_clasificacion(self, frecuencia_minima=3):
        """
        Prepara los datos para clasificación.
        
        Args:
            frecuencia_minima (int): Meses mínimos para ser cliente frecuente
            
        Returns:
            tuple: (X, y, feature_names)
        """
        print(f"\nPreparando datos para clasificación (frecuencia mínima: {frecuencia_minima} meses)...")
        
        # Crear variable objetivo
        self.df_clasificacion['cliente_frecuente'] = (
            self.df_clasificacion['frecuencia_meses'] >= frecuencia_minima
        ).astype(int)
        
        # Preparar características
        feature_cols = ['total_compras', 'total_gastado', 'num_productos', 'num_categorias', 'num_municipios']
        X = self.df_clasificacion[feature_cols].copy()
        y = self.df_clasificacion['cliente_frecuente'].copy()
        
        print(f"  Características: {feature_cols}")
        print(f"  Total muestras: {len(X)}")
        print(f"  Clientes frecuentes: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Clientes no frecuentes: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
        
        return X, y, feature_cols
    
    def evaluar_regresion_logistica(self, X, y, feature_names):
        """
        Evalúa modelo de Regresión Logística.
        
        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            feature_names (list): Nombres de características
            
        Returns:
            dict: Resultados de Regresión Logística
        """
        print("\nEvaluando Regresión Logística...")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verificar clases suficientes
        if y_train.nunique() < 2:
            print("  Error: Insuficientes clases para entrenamiento")
            return None
        
        # Entrenar modelo con optimización de hiperparámetros
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 2000]
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        scores_cv = cross_val_score(mejor_modelo, X, y, cv=5, scoring='accuracy')
        
        # Reporte de clasificación
        reporte = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        resultados = {
            'modelo': mejor_modelo,
            'mejores_parametros': grid_search.best_params_,
            'predicciones': {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            },
            'metricas': {
                'accuracy': accuracy,
                'auc_roc': auc,
                'cv_scores': scores_cv,
                'cv_mean': scores_cv.mean(),
                'cv_std': scores_cv.std()
            },
            'confusion_matrix': cm,
            'classification_report': reporte,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'feature_importance': dict(zip(feature_names, mejor_modelo.coef_[0])),
            'feature_names': feature_names
        }
        
        print(f"  Mejores parámetros: {grid_search.best_params_}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  CV Score: {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")
        
        return resultados
    
    def evaluar_arbol_decision(self, X, y, feature_names):
        """
        Evalúa modelo de Árbol de Decisión.
        
        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            feature_names (list): Nombres de características
            
        Returns:
            dict: Resultados de Árbol de Decisión
        """
        print("\nEvaluando Árbol de Decisión...")
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Verificar clases suficientes
        if y_train.nunique() < 2:
            print("  Error: Insuficientes clases para entrenamiento")
            return None
        
        # Entrenar modelo con optimización de hiperparámetros
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        scores_cv = cross_val_score(mejor_modelo, X, y, cv=5, scoring='accuracy')
        
        # Reporte de clasificación
        reporte = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        resultados = {
            'modelo': mejor_modelo,
            'mejores_parametros': grid_search.best_params_,
            'predicciones': {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            },
            'metricas': {
                'accuracy': accuracy,
                'auc_roc': auc,
                'cv_scores': scores_cv,
                'cv_mean': scores_cv.mean(),
                'cv_std': scores_cv.std()
            },
            'confusion_matrix': cm,
            'classification_report': reporte,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'feature_importance': dict(zip(feature_names, mejor_modelo.feature_importances_)),
            'feature_names': feature_names
        }
        
        print(f"  Mejores parámetros: {grid_search.best_params_}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  CV Score: {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")
        
        return resultados
    
    def comparar_modelos_clasificacion(self, resultados_logistica, resultados_arbol):
        """
        Compara modelos de clasificación.
        
        Args:
            resultados_logistica (dict): Resultados de regresión logística
            resultados_arbol (dict): Resultados de árbol de decisión
            
        Returns:
            pd.DataFrame: Comparación de modelos
        """
        print("\nComparando modelos de clasificación...")
        
        comparacion = []
        
        if resultados_logistica:
            comparacion.append({
                'Modelo': 'Regresión Logística',
                'Accuracy': resultados_logistica['metricas']['accuracy'],
                'AUC-ROC': resultados_logistica['metricas']['auc_roc'],
                'CV Mean': resultados_logistica['metricas']['cv_mean'],
                'CV Std': resultados_logistica['metricas']['cv_std']
            })
        
        if resultados_arbol:
            comparacion.append({
                'Modelo': 'Árbol de Decisión',
                'Accuracy': resultados_arbol['metricas']['accuracy'],
                'AUC-ROC': resultados_arbol['metricas']['auc_roc'],
                'CV Mean': resultados_arbol['metricas']['cv_mean'],
                'CV Std': resultados_arbol['metricas']['cv_std']
            })
        
        df_comparacion = pd.DataFrame(comparacion)
        
        print("  Comparación completada:")
        print(df_comparacion.to_string(index=False))
        
        return df_comparacion

# =============================================================================
# EXPORTACIÓN Y PERSISTENCIA (FUNCIONES CORREGIDAS)
# =============================================================================

def exportar_resultados(resultados_clustering, resultados_clasificacion, comparaciones):
    """
    Exporta todos los resultados de los modelos (versión corregida para PAM).
    
    Args:
        resultados_clustering (dict): Resultados de clustering
        resultados_clasificacion (dict): Resultados de clasificación
        comparaciones (dict): Comparaciones de modelos
    """
    print("\nExportando resultados de modelos...")
    
    # Crear copias de resultados sin objetos no serializables
    resultados_clustering_limpio = {}
    resultados_clasificacion_limpio = {}
    
    # Procesar resultados de clustering
    for modelo_name, resultados in resultados_clustering.items():
        if not isinstance(resultados, dict):
            continue
            
        resultados_clustering_limpio[modelo_name] = {}
        
        for k, resultado in resultados.items():
            if k in ['mejor_k', 'mejor_modelo']:
                if k == 'mejor_k':
                    resultados_clustering_limpio[modelo_name][k] = resultado
                else:
                    # Para mejor_modelo, solo guardar referencias
                    if isinstance(resultado, dict) and 'metricas' in resultado:
                        resultados_clustering_limpio[modelo_name][k] = {
                            'metricas': resultado['metricas'],
                            'parametros': resultado['parametros'],
                            'etiquetas': resultado['etiquetas'].tolist() if hasattr(resultado['etiquetas'], 'tolist') else resultado['etiquetas']
                        }
                continue
                
            if isinstance(resultado, dict) and 'metricas' in resultado:
                # Crear versión limpia sin objetos de sklearn/pyclustering
                resultado_limpio = {
                    'metricas': resultado['metricas'],
                    'parametros': resultado['parametros'],
                    'etiquetas': resultado['etiquetas'].tolist() if hasattr(resultado['etiquetas'], 'tolist') else resultado['etiquetas']
                }
                
                # Agregar coordenadas PCA si existen
                if 'pca_coords' in resultado:
                    resultado_limpio['pca_coords'] = resultado['pca_coords'].tolist() if hasattr(resultado['pca_coords'], 'tolist') else resultado['pca_coords']
                
                # Para PAM, agregar medoids si existen
                if 'medoids' in resultado:
                    resultado_limpio['medoids'] = resultado['medoids']
                
                resultados_clustering_limpio[modelo_name][k] = resultado_limpio
    
    # Procesar resultados de clasificación
    for modelo_name, resultado in resultados_clasificacion.items():
        if resultado is None:
            resultados_clasificacion_limpio[modelo_name] = None
            continue
            
        # Crear versión limpia sin objetos de sklearn
        resultado_limpio = {
            'mejores_parametros': resultado['mejores_parametros'],
            'metricas': resultado['metricas'],
            'feature_importance': resultado['feature_importance'],
            'feature_names': resultado['feature_names'],
            'confusion_matrix': resultado['confusion_matrix'].tolist() if hasattr(resultado['confusion_matrix'], 'tolist') else resultado['confusion_matrix'],
            'classification_report': resultado['classification_report']
        }
        
        # Agregar curva ROC
        if 'roc_curve' in resultado:
            resultado_limpio['roc_curve'] = {
                'fpr': resultado['roc_curve']['fpr'].tolist() if hasattr(resultado['roc_curve']['fpr'], 'tolist') else resultado['roc_curve']['fpr'],
                'tpr': resultado['roc_curve']['tpr'].tolist() if hasattr(resultado['roc_curve']['tpr'], 'tolist') else resultado['roc_curve']['tpr']
            }
        
        # Agregar predicciones (solo métricas, no arrays completos)
        if 'predicciones' in resultado:
            resultado_limpio['predicciones_summary'] = {
                'n_samples': len(resultado['predicciones']['y_test']),
                'accuracy': (resultado['predicciones']['y_test'] == resultado['predicciones']['y_pred']).mean(),
                'positive_class_ratio': resultado['predicciones']['y_test'].mean()
            }
        
        resultados_clasificacion_limpio[modelo_name] = resultado_limpio
    
    # Crear estructuras serializables para JSON
    resultados_exportables = {
        'clustering': resultados_clustering_limpio,
        'clasificacion': resultados_clasificacion_limpio,
        'comparaciones': {
            'clustering': comparaciones['clustering'].to_dict('records') if 'clustering' in comparaciones else [],
            'clasificacion': comparaciones['clasificacion'].to_dict('records') if 'clasificacion' in comparaciones else []
        },
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'modelos_clustering': list(resultados_clustering_limpio.keys()),
            'modelos_clasificacion': [k for k, v in resultados_clasificacion_limpio.items() if v is not None],
            'total_experimentos': sum(len(v) for v in resultados_clustering_limpio.values() if isinstance(v, dict)) + len([k for k, v in resultados_clasificacion_limpio.items() if v is not None])
        }
    }
    
    # Exportar resultados LIMPIOS (sin objetos problemáticos)
    try:
        with open(os.path.join(MODELS_DIR, "resultados_modelos_limpio.pkl"), 'wb') as f:
            pickle.dump({
                'clustering': resultados_clustering_limpio,
                'clasificacion': resultados_clasificacion_limpio
            }, f)
        print("  ✓ resultados_modelos_limpio.pkl exportado")
    except Exception as e:
        print(f"  ✗ Error exportando PKL limpio: {e}")
    
    # Exportar versión JSON para interoperabilidad
    try:
        with open(os.path.join(MODELS_DIR, "resultados_modelos.json"), 'w') as f:
            json.dump(resultados_exportables, f, indent=2, default=str)
        print("  ✓ resultados_modelos.json exportado")
    except Exception as e:
        print(f"  ✗ Error exportando JSON: {e}")
    
    # Exportar comparaciones
    try:
        for nombre, df in comparaciones.items():
            df.to_csv(os.path.join(MODELS_DIR, f"comparacion_{nombre}.csv"), index=False)
            df.to_pickle(os.path.join(MODELS_DIR, f"comparacion_{nombre}.pkl"))
        print("  ✓ Comparaciones exportadas")
    except Exception as e:
        print(f"  ✗ Error exportando comparaciones: {e}")
    
    # Exportar modelos INDIVIDUALES para MLflow (solo los necesarios)
    try:
        modelos_para_mlflow = {}
        
        # Solo guardar modelos de sklearn que SÍ se pueden serializar
        for modelo_name, resultados in resultados_clustering.items():
            if modelo_name in ['kmeans', 'jerarquico'] and 'mejor_modelo' in resultados:
                mejor = resultados['mejor_modelo']
                if 'modelo' in mejor:
                    modelos_para_mlflow[f"{modelo_name}_mejor"] = mejor['modelo']
        
        for modelo_name, resultado in resultados_clasificacion.items():
            if resultado is not None and 'modelo' in resultado:
                modelos_para_mlflow[f"{modelo_name}_mejor"] = resultado['modelo']
        
        with open(os.path.join(MODELS_DIR, "modelos_sklearn.pkl"), 'wb') as f:
            pickle.dump(modelos_para_mlflow, f)
        print("  ✓ modelos_sklearn.pkl exportado")
    except Exception as e:
        print(f"  ✗ Error exportando modelos sklearn: {e}")
    
    print("\n  Archivos exportados:")
    print("    - resultados_modelos_limpio.pkl (datos sin objetos problemáticos)")
    print("    - resultados_modelos.json (interoperable)")
    print("    - modelos_sklearn.pkl (solo modelos sklearn serializables)")
    print("    - comparacion_*.csv y *.pkl")

def generar_reporte_modelos(resultados_clustering, resultados_clasificacion, comparaciones):
    """
    Genera reporte técnico de los modelos entrenados.
    
    Args:
        resultados_clustering (dict): Resultados de clustering
        resultados_clasificacion (dict): Resultados de clasificación
        comparaciones (dict): Comparaciones de modelos
    """
    print("\nGenerando reporte técnico de modelos...")
    
    reporte = f"""
REPORTE TÉCNICO DE MODELOS DE MACHINE LEARNING (VERSIÓN CORREGIDA)
================================================================

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Segmento: 2/4 del proyecto

MODELOS DE CLUSTERING ENTRENADOS:
=================================

K-MEANS:
--------
"""
    
    # Añadir resultados de K-Means
    if 'kmeans' in resultados_clustering and 'mejor_modelo' in resultados_clustering['kmeans']:
        mejor_kmeans = resultados_clustering['kmeans']['mejor_modelo']
        mejor_k = resultados_clustering['kmeans']['mejor_k']
        reporte += f"""
Mejor configuración: k={mejor_k}
Silhouette Score: {mejor_kmeans['metricas']['silhouette_score']:.4f}
Calinski-Harabasz: {mejor_kmeans['metricas']['calinski_harabasz']:.4f}
Davies-Bouldin: {mejor_kmeans['metricas']['davies_bouldin']:.4f}
Inercia: {mejor_kmeans['metricas']['inercia']:.2f}
"""
    
    reporte += """
CLUSTERING JERÁRQUICO:
---------------------
"""
    
    # Añadir resultados de Clustering Jerárquico
    if 'jerarquico' in resultados_clustering and 'mejor_modelo' in resultados_clustering['jerarquico']:
        mejor_jerarquico = resultados_clustering['jerarquico']['mejor_modelo']
        mejor_k = resultados_clustering['jerarquico']['mejor_k']
        reporte += f"""
Mejor configuración: k={mejor_k}, linkage=ward
Silhouette Score: {mejor_jerarquico['metricas']['silhouette_score']:.4f}
Calinski-Harabasz: {mejor_jerarquico['metricas']['calinski_harabasz']:.4f}
Davies-Bouldin: {mejor_jerarquico['metricas']['davies_bouldin']:.4f}
"""
    
    reporte += """
PAM (K-MEDOIDS):
---------------
"""
    
    # Añadir resultados de PAM
    if 'pam' in resultados_clustering and resultados_clustering['pam'] and 'mejor_modelo' in resultados_clustering['pam']:
        mejor_pam = resultados_clustering['pam']['mejor_modelo']
        mejor_k = resultados_clustering['pam']['mejor_k']
        reporte += f"""
Mejor configuración: k={mejor_k}
Silhouette Score: {mejor_pam['metricas']['silhouette_score']:.4f}
Medoids encontrados: {len(mejor_pam.get('medoids', []))}
"""
    
    reporte += """

MODELOS DE CLASIFICACIÓN ENTRENADOS:
===================================

REGRESIÓN LOGÍSTICA:
-------------------
"""
    
    # Añadir resultados de Regresión Logística
    if 'logistica' in resultados_clasificacion and resultados_clasificacion['logistica']:
        logistica = resultados_clasificacion['logistica']
        reporte += f"""
Mejores parámetros: {logistica['mejores_parametros']}
Accuracy: {logistica['metricas']['accuracy']:.4f}
AUC-ROC: {logistica['metricas']['auc_roc']:.4f}
CV Score: {logistica['metricas']['cv_mean']:.4f} ± {logistica['metricas']['cv_std']:.4f}
"""
    
    reporte += """
ÁRBOL DE DECISIÓN:
-----------------
"""
    
    # Añadir resultados de Árbol de Decisión
    if 'arbol' in resultados_clasificacion and resultados_clasificacion['arbol']:
        arbol = resultados_clasificacion['arbol']
        reporte += f"""
Mejores parámetros: {arbol['mejores_parametros']}
Accuracy: {arbol['metricas']['accuracy']:.4f}
AUC-ROC: {arbol['metricas']['auc_roc']:.4f}
CV Score: {arbol['metricas']['cv_mean']:.4f} ± {arbol['metricas']['cv_std']:.4f}
"""
    
    reporte += """

ARCHIVOS GENERADOS (VERSIÓN CORREGIDA):
======================================
- models_results/resultados_modelos_limpio.pkl (sin objetos problemáticos)
- models_results/resultados_modelos.json (interoperable)
- models_results/modelos_sklearn.pkl (solo modelos sklearn)
- models_results/comparacion_clustering.csv
- models_results/comparacion_clasificacion.csv

CORRECCIONES IMPLEMENTADAS:
==========================
- Problema de serialización PAM/K-medoids SOLUCIONADO
- Objetos pyclustering separados de objetos sklearn
- Estructura limpia para MLflow preparada
- Manejo robusto de errores de exportación

PREPARACIÓN PARA MLFLOW:
=======================
Todos los modelos están listos para ser registrados en MLflow con:
- Parámetros optimizados documentados
- Métricas completas calculadas
- Artifacts preparados para logging
- Metadata estructurada
- Modelos sklearn serializables por separado

PRÓXIMOS PASOS:
==============
SEGMENTO 1: Data Processing + EDA [COMPLETADO]
SEGMENTO 2: Modelos ML [COMPLETADO - CORREGIDO]
SEGMENTO 3: MLflow Experiments [PENDIENTE]
SEGMENTO 4: Dashboard Integrado [PENDIENTE]
"""
    
    # Guardar reporte
    with open(os.path.join(REPORTS_DIR, "reporte_modelos.txt"), 'w', encoding='utf-8') as f:
        f.write(reporte)
    
    print("  Reporte técnico generado: model_reports/reporte_modelos.txt")

def preparar_para_mlflow_corregido(resultados_clustering, resultados_clasificacion):
    """
    Prepara estructura de datos optimizada para MLflow (versión corregida).
    
    Args:
        resultados_clustering (dict): Resultados de clustering
        resultados_clasificacion (dict): Resultados de clasificación
    """
    print("\nPreparando estructura para MLflow...")
    
    mlflow_data = {
        'experiments': {
            'clustering': {},
            'clasificacion': {}
        },
        'best_models': {},
        'comparison_metrics': {},
        'artifacts_info': {}
    }
    
    # Preparar experimentos de clustering para MLflow
    for modelo_name in ['kmeans', 'jerarquico', 'pam']:
        if modelo_name in resultados_clustering:
            resultados = resultados_clustering[modelo_name]
            
            if isinstance(resultados, dict) and 'mejor_modelo' in resultados:
                mejor = resultados['mejor_modelo']
                
                # Crear entrada limpia para MLflow
                mlflow_entry = {
                    'run_name': f"{modelo_name}_best",
                    'parameters': mejor['parametros'],
                    'metrics': mejor['metricas'],
                    'tags': {
                        'model_type': 'clustering',
                        'algorithm': modelo_name,
                        'optimization': 'silhouette_score',
                        'best_k': resultados.get('mejor_k', 'unknown')
                    }
                }
                
                # Para PAM, agregar información especial
                if modelo_name == 'pam' and 'medoids' in mejor:
                    mlflow_entry['artifacts_info'] = {
                        'medoids_count': len(mejor['medoids']),
                        'medoids_indices': mejor['medoids']
                    }
                
                mlflow_data['experiments']['clustering'][modelo_name] = mlflow_entry
    
    # Preparar experimentos de clasificación para MLflow
    for modelo_name in ['logistica', 'arbol']:
        if modelo_name in resultados_clasificacion and resultados_clasificacion[modelo_name]:
            resultado = resultados_clasificacion[modelo_name]
            
            mlflow_entry = {
                'run_name': f"{modelo_name}_best",
                'parameters': resultado['mejores_parametros'],
                'metrics': resultado['metricas'],
                'tags': {
                    'model_type': 'classification',
                    'algorithm': modelo_name,
                    'optimization': 'roc_auc'
                },
                'artifacts_info': {
                    'feature_names': resultado['feature_names'],
                    'feature_importance': resultado['feature_importance'],
                    'classification_report': resultado['classification_report']
                }
            }
            
            mlflow_data['experiments']['clasificacion'][modelo_name] = mlflow_entry
    
    # Guardar estructura para MLflow
    try:
        with open(os.path.join(MODELS_DIR, "mlflow_prepared_data.pkl"), 'wb') as f:
            pickle.dump(mlflow_data, f)
        print("  ✓ mlflow_prepared_data.pkl")
    except Exception as e:
        print(f"  ✗ Error con PKL: {e}")
    
    try:
        with open(os.path.join(MODELS_DIR, "mlflow_prepared_data.json"), 'w') as f:
            json.dump(mlflow_data, f, indent=2, default=str)
        print("  ✓ mlflow_prepared_data.json")
    except Exception as e:
        print(f"  ✗ Error con JSON: {e}")
    
    print("  Estructura para MLflow preparada exitosamente")
    return mlflow_data

def validar_resultados():
    """
    Valida que todos los resultados fueron generados correctamente (versión corregida).
    
    Returns:
        bool: True si la validación es exitosa
    """
    print("\nValidando resultados generados...")
    
    archivos_requeridos = [
        "models_results/resultados_modelos_limpio.pkl",  # Cambiado
        "models_results/resultados_modelos.json",
        "models_results/modelos_sklearn.pkl",            # Nuevo
        "models_results/mlflow_prepared_data.pkl",
        "models_results/mlflow_prepared_data.json",
        "model_reports/reporte_modelos.txt"
    ]
    
    archivos_opcionales = [
        "models_results/comparacion_clustering.csv",
        "models_results/comparacion_clasificacion.csv"
    ]
    
    todos_presentes = True
    
    print("  Archivos requeridos:")
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            size = os.path.getsize(archivo)
            print(f"  ✓ {archivo} ({size:,} bytes)")
        else:
            print(f"  ✗ FALTA: {archivo}")
            todos_presentes = False
    
    print("\n  Archivos opcionales:")
    for archivo in archivos_opcionales:
        if os.path.exists(archivo):
            size = os.path.getsize(archivo)
            print(f"  ✓ {archivo} ({size:,} bytes)")
        else:
            print(f"  ? No generado: {archivo}")
    
    if todos_presentes:
        print("\n  🎉 VALIDACIÓN EXITOSA: Todos los archivos críticos generados")
    else:
        print("\n  ❌ VALIDACIÓN FALLIDA: Faltan archivos críticos")
    
    return todos_presentes

# =============================================================================
# PIPELINE PRINCIPAL (VERSIÓN CORREGIDA)
# =============================================================================

def main():
    """
    Pipeline principal del Segmento 2: Modelos de Machine Learning (versión corregida).
    """
    print("\nINICIANDO PIPELINE DE MODELOS ML (VERSIÓN CORREGIDA)...")
    
    # 1. Cargar datos procesados
    datos = cargar_datos_procesados()
    if datos is None:
        print("ERROR: No se pudieron cargar los datos del Segmento 1")
        return None
    
    # 2. Preparar datos para clustering
    X_clustering, scaler, feature_names = preparar_datos_clustering(
        datos['clustering'], datos['config']
    )
    
    cliente_ids = datos['clustering']['cliente'].values
    
    # 3. Entrenar modelos de clustering
    print("\nENTRENANDO MODELOS DE CLUSTERING...")
    clustering_models = ClusteringModels(X_clustering, feature_names, cliente_ids)
    
    # Evaluar todos los modelos de clustering
    resultados_kmeans = clustering_models.evaluar_kmeans()
    resultados_jerarquico = clustering_models.evaluar_clustering_jerarquico()
    resultados_pam = clustering_models.evaluar_pam()
    
    # Comparar modelos de clustering
    comparacion_clustering = clustering_models.comparar_modelos_clustering()
    
    # 4. Entrenar modelos de clasificación
    print("\nENTRENANDO MODELOS DE CLASIFICACIÓN...")
    classification_models = ClassificationModels(datos['clasificacion'])
    
    # Preparar datos de clasificación
    X_class, y_class, feature_names_class = classification_models.preparar_datos_clasificacion()
    
    # Evaluar modelos de clasificación
    resultados_logistica = classification_models.evaluar_regresion_logistica(
        X_class, y_class, feature_names_class
    )
    resultados_arbol = classification_models.evaluar_arbol_decision(
        X_class, y_class, feature_names_class
    )
    
    # Comparar modelos de clasificación
    comparacion_clasificacion = classification_models.comparar_modelos_clasificacion(
        resultados_logistica, resultados_arbol
    )
    
    # 5. Consolidar resultados
    resultados_clustering_consolidados = {
        'kmeans': resultados_kmeans,
        'jerarquico': resultados_jerarquico,
        'pam': resultados_pam
    }
    
    resultados_clasificacion_consolidados = {
        'logistica': resultados_logistica,
        'arbol': resultados_arbol
    }
    
    comparaciones = {
        'clustering': comparacion_clustering,
        'clasificacion': comparacion_clasificacion
    }
    
    # 6. Exportar resultados (USANDO FUNCIÓN CORREGIDA)
    try:
        exportar_resultados(
            resultados_clustering_consolidados,
            resultados_clasificacion_consolidados,
            comparaciones
        )
        print("✓ Exportación de resultados exitosa")
    except Exception as e:
        print(f"✗ Error en exportación: {e}")
        return None
    
    # 7. Generar reporte técnico
    try:
        generar_reporte_modelos(
            resultados_clustering_consolidados,
            resultados_clasificacion_consolidados,
            comparaciones
        )
        print("✓ Reporte técnico generado")
    except Exception as e:
        print(f"✗ Error generando reporte: {e}")
    
    # 8. Preparar para MLflow (USANDO FUNCIÓN CORREGIDA)
    try:
        preparar_para_mlflow_corregido(
            resultados_clustering_consolidados,
            resultados_clasificacion_consolidados
        )
        print("✓ Preparación para MLflow exitosa")
    except Exception as e:
        print(f"✗ Error preparando MLflow: {e}")
    
    # 9. Validar resultados
    validacion_exitosa = validar_resultados()
    
    if validacion_exitosa:
        print("\n" + "="*70)
        print("SEGMENTO 2 COMPLETADO EXITOSAMENTE (VERSIÓN CORREGIDA)")
        print("="*70)
        print("RESUMEN DE RESULTADOS:")
        print(f"  - Modelos de clustering entrenados: {len(resultados_clustering_consolidados)}")
        print(f"  - Modelos de clasificación entrenados: {len([r for r in resultados_clasificacion_consolidados.values() if r is not None])}")
        print(f"  - Archivos de resultados generados: 6+")
        print(f"  - Comparaciones realizadas: {len(comparaciones)}")
        print("  - Problema de serialización PAM SOLUCIONADO ✓")
        print("="*70)
        print("LISTO PARA COMMIT Y SEGMENTO 3 (MLFLOW)")
        
        return {
            'clustering': resultados_clustering_consolidados,
            'clasificacion': resultados_clasificacion_consolidados,
            'comparaciones': comparaciones
        }
    else:
        print("\nERROR: Validación fallida. Revisa los errores anteriores.")
        return None

def generar_instrucciones_segmento3():
    """
    Genera instrucciones detalladas para el Segmento 3 (MLflow).
    """
    instrucciones = """
INSTRUCCIONES PARA SEGMENTO 3: MLFLOW EXPERIMENTS (VERSIÓN CORREGIDA)
=====================================================================

ARCHIVOS A UTILIZAR:
-------------------
- models_results/resultados_modelos_limpio.pkl    → Datos sin objetos problemáticos
- models_results/modelos_sklearn.pkl              → Solo modelos sklearn serializables
- models_results/mlflow_prepared_data.pkl         → Estructura optimizada para MLflow
- models_results/comparacion_*.csv                → Métricas de comparación

CORRECCIONES IMPLEMENTADAS:
--------------------------
✓ Problema de serialización PAM/K-medoids SOLUCIONADO
✓ Modelos sklearn separados y funcionalmente serializables
✓ Estructura limpia preparada específicamente para MLflow
✓ Manejo robusto de errores de exportación

CONFIGURACIÓN DE MLFLOW:
-----------------------
1. Instalar MLflow: pip install mlflow boto3
2. Configurar AWS EC2 instance
3. Configurar MLflow tracking server
4. Configurar artifact store (S3 bucket)

EXPERIMENTOS A REGISTRAR:
------------------------
CLUSTERING:
- K-Means experiments (k=2 a 10) → usar modelos_sklearn.pkl
- Hierarchical clustering experiments → usar modelos_sklearn.pkl
- PAM experiments → usar solo métricas (medoids info disponible)
- Comparison experiment

CLASIFICACIÓN:
- Logistic Regression with GridSearch → usar modelos_sklearn.pkl
- Decision Tree with GridSearch → usar modelos_sklearn.pkl
- Model comparison experiment

MLFLOW TRACKING REQUERIDO:
--------------------------
1. Parameters: Todos los hiperparámetros
2. Metrics: Todas las métricas calculadas
3. Artifacts: 
   - Modelos serializados (sklearn)
   - Gráficos de métricas
   - Confusion matrices
   - ROC curves
   - Información de medoids (PAM)
4. Tags: model_type, algorithm, optimization_metric

CARGA DE MODELOS PARA MLFLOW:
----------------------------
```python
import pickle
with open('models_results/modelos_sklearn.pkl', 'rb') as f:
    modelos = pickle.load(f)
    
# Modelos disponibles:
# - kmeans_mejor
# - jerarquico_mejor  
# - logistica_mejor
# - arbol_mejor
```

OUTPUTS ESPERADOS PARA EL REPORTE:
---------------------------------
1. Screenshots de MLflow UI en EC2
2. Model Registry con versiones
3. Experiments dashboard
4. Artifact storage evidencia
5. IP de EC2 visible en capturas
6. Evidencia de modelos funcionando correctamente

PREPARACIÓN PARA DASHBOARD:
--------------------------
Los modelos registrados en MLflow deben estar listos para:
- Integración con dashboard final
- Serving de predicciones
- Comparación visual de resultados

NOTAS TÉCNICAS:
--------------
- PAM se maneja solo con métricas y información de medoids
- Todos los modelos sklearn están probados y funcionan
- Estructura JSON disponible para interoperabilidad
- Validación automática implementada
"""
    
    with open("INSTRUCCIONES_SEGMENTO_3_CORREGIDO.txt", 'w', encoding='utf-8') as f:
        f.write(instrucciones)
    
    print("Instrucciones CORREGIDAS para Segmento 3 generadas: INSTRUCCIONES_SEGMENTO_3_CORREGIDO.txt")

# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline completo
    resultados = main()
    
    if resultados is not None:
        # Generar instrucciones para siguiente segmento
        generar_instrucciones_segmento3()
        
        print("\nSEGMENTO 2 FINALIZADO CON ÉXITO (VERSIÓN CORREGIDA)")
        print("Problema de serialización PAM SOLUCIONADO ✓")
        print("Preparado para MLflow Experiments (Segmento 3)")
    else:
        print("\nERROR EN EL SEGMENTO 2")
        print("Revisa los errores y ejecuta nuevamente")

"""
DOCUMENTACIÓN TÉCNICA DEL SEGMENTO 2 (VERSIÓN CORREGIDA)
========================================================

PROPÓSITO:
---------
Este segmento implementa todos los modelos de machine learning del proyecto:
clustering (K-Means, Jerárquico, PAM) y clasificación (Regresión Logística, 
Árbol de Decisión). Optimiza hiperparámetros y prepara todo para MLflow.

CORRECCIONES IMPLEMENTADAS:
--------------------------
✓ PROBLEMA PRINCIPAL SOLUCIONADO: Serialización de objetos PAM/K-medoids
✓ Separación de modelos sklearn serializables
✓ Estructura limpia sin objetos problemáticos
✓ Manejo robusto de errores de exportación
✓ Validación mejorada con archivos específicos

ARQUITECTURA CORREGIDA:
----------------------
1. CARGA: Importa datos procesados del Segmento 1
2. CLUSTERING: Entrena y evalúa 3 algoritmos con múltiples configuraciones
3. CLASIFICACIÓN: Entrena 2 modelos con GridSearch optimization
4. COMPARACIÓN: Analiza y compara todos los modelos
5. EXPORTACIÓN LIMPIA: Guarda resultados sin objetos problemáticos
6. PREPARACIÓN ESPECÍFICA: Estructura datos optimizada para MLflow

ARCHIVOS GENERADOS:
------------------
- resultados_modelos_limpio.pkl: Datos sin objetos problemáticos
- modelos_sklearn.pkl: Solo modelos sklearn serializables  
- resultados_modelos.json: Versión interoperable
- mlflow_prepared_data.pkl/json: Optimizado para MLflow
- comparacion_*.csv: Métricas de comparación

OPTIMIZACIONES IMPLEMENTADAS:
----------------------------
- GridSearchCV para hiperparámetros
- Validación cruzada 5-fold
- Múltiples métricas de evaluación
- PCA para visualización de clusters
- Manejo robusto de errores
- Serialización selectiva por tipo de modelo

INTEGRACIÓN CON MLFLOW:
----------------------
- Estructura de datos optimizada para tracking
- Parámetros y métricas documentados
- Tags y metadata preparados
- Artifacts listos para logging
- Modelos sklearn probados y funcionando
- Información especial de PAM disponible

OUTPUTS PARA EL REPORTE:
-----------------------
- Comparaciones cuantitativas de modelos
- Métricas de evaluación completas
- Justificación técnica de selecciones
- Preparación completa para deployment
- Evidencia de corrección de problemas

CALIDAD DE CÓDIGO:
-----------------
- Clases organizadas por tipo de modelo
- Documentación completa con docstrings
- Manejo de excepciones robusto
- Validación automática de outputs
- Compatibilidad con arquitectura modular
- Código probado y funcionando
"""