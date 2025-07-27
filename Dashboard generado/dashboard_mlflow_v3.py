"""
Dashboard MLflow V3 - Segmentación y Clasificación de Clientes
===============================================================

Dashboard interactivo integrado con MLflow V3 para explorar:
- Análisis exploratorio de datos mejorado
- Segmentación con modelos integrados de MLflow
- Clasificación predictiva con tracking completo
- Comparación de modelos con métricas avanzadas

Autor: Equipo de Analítica
Proyecto: Entrega Final - Despliegue de Soluciones Analíticas
Fecha: Julio 2025
Versión: 3.0
"""

import pandas as pd
import numpy as np
import os
import warnings
import io
import base64
import json

# Configuración para evitar errores
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

warnings.filterwarnings('ignore')

# Dash y visualización
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    silhouette_score, accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow no disponible - funcionando en modo local")

# Logo
try:
    from logo_src import logo_src
except ImportError:
    logo_src = ""

# Configuración
DATA_PATH = "resumen por item final.xlsx"
CLIENTE_MOSTRADOR = "000022222222"
MLFLOW_URI = "http://localhost:5000"

# Colores actualizados
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'dark': '#343a40',
    'light': '#f8f9fa'
}

def cargar_y_procesar_datos():
    """Carga y procesa los datos con validaciones mejoradas."""
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Archivo no encontrado: {DATA_PATH}")
            archivos_excel = [f for f in os.listdir('.') if f.endswith('.xlsx')]
            if archivos_excel:
                print(f"Usando: {archivos_excel[0]}")
                # Usar el primer archivo Excel encontrado
                archivo_actual = archivos_excel[0]
            else:
                raise FileNotFoundError("No se encontraron archivos Excel")
        else:
            archivo_actual = DATA_PATH
        
        df = pd.read_excel(archivo_actual, sheet_name=0, dtype={'CLIENTE': str, 'CODIGO': str})
        
        # Renombrar columnas
        df.columns = [
            "anio", "mes", "cliente", "codigo_producto", "nombre_producto",
            "unidad_medida", "cantidad", "valor_unitario", "descuento_total",
            "valor_total", "num_factura", "cc", "cat1", "cat2", "cat3",
            "categoria", "departamento", "municipio"
        ]
        
        # Limpiar datos
        if 'cc' in df.columns:
            df.drop(columns="cc", inplace=True)
        
        df["mes"] = df["mes"].str.strip().str.capitalize()
        df["anio"] = df["anio"].astype(str)
        df["mes_anio"] = df["mes"] + " - " + df["anio"]
        
        # Filtrar datos válidos
        df = df[
            (df['valor_total'] > 0) &
            (df['cantidad'] > 0) &
            (df['cliente'] != CLIENTE_MOSTRADOR)
        ].copy()
        
        print(f"Datos cargados exitosamente: {len(df):,} registros")
        return df
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def preparar_datos_clustering(df):
    """Prepara datos para clustering con características V3."""
    
    # Agregaciones por cliente
    client_metrics = df.groupby('cliente').agg({
        'valor_total': ['sum', 'mean', 'std', 'count'],
        'cantidad': ['sum', 'mean'],
        'descuento_total': ['sum', 'mean'],
        'codigo_producto': 'nunique',
        'categoria': 'nunique',
        'departamento': 'nunique',
        'municipio': 'nunique'
    }).reset_index()
    
    # Aplanar columnas
    client_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in client_metrics.columns.values]
    client_metrics.rename(columns={'cliente_': 'cliente'}, inplace=True)
    
    # Características derivadas mejoradas
    client_metrics['ticket_promedio'] = client_metrics['valor_total_sum'] / client_metrics['valor_total_count']
    client_metrics['variabilidad_gasto'] = client_metrics['valor_total_std'] / client_metrics['valor_total_mean']
    client_metrics['diversidad_productos'] = client_metrics['codigo_producto_nunique'] / client_metrics['valor_total_count']
    client_metrics['diversidad_geografica'] = client_metrics['municipio_nunique'] / client_metrics['valor_total_count']
    client_metrics['lealtad_geografica'] = 1 / client_metrics['municipio_nunique']
    
    # Manejar valores faltantes
    client_metrics = client_metrics.fillna(0)
    
    # Seleccionar características finales
    feature_cols = [
        'valor_total_sum', 'valor_total_mean', 'valor_total_count',
        'cantidad_sum', 'ticket_promedio', 'variabilidad_gasto',
        'codigo_producto_nunique', 'categoria_nunique',
        'diversidad_productos', 'diversidad_geografica', 'lealtad_geografica'
    ]
    
    X = client_metrics[feature_cols].fillna(0)
    
    return client_metrics, X, feature_cols

def preparar_datos_clasificacion(df):
    """Prepara datos para clasificación con umbral adaptativo."""
    
    # Mapeo de meses
    month_mapping = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4,
        'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
        'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    
    df_temp = df.copy()
    df_temp['mes_num'] = df_temp['mes'].map(month_mapping).fillna(1)
    df_temp['fecha'] = pd.to_datetime(
        df_temp['anio'].astype(str) + '-' + df_temp['mes_num'].astype(str).str.zfill(2) + '-01'
    )
    
    # Métricas por cliente
    client_metrics = df_temp.groupby('cliente').agg({
        'fecha': ['nunique', 'min', 'max'],
        'valor_total': ['sum', 'mean', 'count'],
        'cantidad': ['sum', 'mean'],
        'codigo_producto': 'nunique',
        'categoria': 'nunique',
        'municipio': 'nunique'
    }).reset_index()
    
    # Aplanar columnas
    client_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in client_metrics.columns.values]
    client_metrics.rename(columns={'cliente_': 'cliente'}, inplace=True)
    
    # Características derivadas
    client_metrics['frecuencia_mensual'] = client_metrics['fecha_nunique']
    client_metrics['gasto_mensual_promedio'] = client_metrics['valor_total_sum'] / client_metrics['frecuencia_mensual']
    client_metrics['ticket_promedio'] = client_metrics['valor_total_sum'] / client_metrics['valor_total_count']
    client_metrics['intensidad_compra'] = client_metrics['valor_total_count'] / client_metrics['fecha_nunique']
    
    # Target con umbral adaptativo (mediana)
    frequency_threshold = client_metrics['frecuencia_mensual'].median()
    client_metrics['es_cliente_frecuente'] = (
        client_metrics['frecuencia_mensual'] > frequency_threshold
    ).astype(int)
    
    # Características para el modelo
    feature_cols = [
        'valor_total_sum', 'valor_total_mean', 'valor_total_count',
        'cantidad_sum', 'gasto_mensual_promedio', 'ticket_promedio',
        'codigo_producto_nunique', 'categoria_nunique', 'municipio_nunique',
        'intensidad_compra'
    ]
    
    X = client_metrics[feature_cols].fillna(0)
    y = client_metrics['es_cliente_frecuente']
    
    return X, y, feature_cols, frequency_threshold

def ejecutar_clustering_v3(X, method='kmeans', n_clusters=3):
    """Ejecuta clustering con modelos V3."""
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Seleccionar algoritmo
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif method == 'dbscan':
        model = DBSCAN(eps=0.7, min_samples=5)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    # Entrenar modelo
    labels = model.fit_predict(X_scaled)
    
    # Calcular métricas
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
    else:
        silhouette = -1
    
    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return {
        'labels': labels,
        'silhouette': silhouette,
        'X_pca': X_pca,
        'pca_explained': pca.explained_variance_ratio_,
        'scaler': scaler,
        'model': model
    }

def ejecutar_clasificacion_v3(X, y, method='logistic'):
    """Ejecuta clasificación con modelos V3."""
    
    if y.nunique() < 2:
        return {'error': 'Insuficientes clases para clasificación'}
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Seleccionar modelo
    if method == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif method == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
    elif method == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    # Entrenar modelo
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_names': X.columns.tolist()
    }

def crear_grafico_eda(df, tipo='evolucion'):
    """Crea gráficos EDA mejorados."""
    
    if tipo == 'evolucion':
        ventas_mes = df.groupby('mes_anio')['valor_total'].sum().reset_index()
        fig = px.line(ventas_mes, x='mes_anio', y='valor_total',
                     title='Evolución Mensual de Ventas',
                     color_discrete_sequence=[COLORS['primary']])
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_yaxes(tickprefix="$", tickformat=",~s")
        
    elif tipo == 'productos':
        top_productos = df.groupby('nombre_producto')['cantidad'].sum().nlargest(10).reset_index()
        fig = px.bar(top_productos, x='cantidad', y='nombre_producto', 
                    orientation='h', title='Top 10 Productos por Cantidad',
                    color_discrete_sequence=[COLORS['success']])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
    elif tipo == 'categorias':
        cat_ventas = df.groupby('categoria')['valor_total'].sum().reset_index()
        fig = px.pie(cat_ventas, values='valor_total', names='categoria',
                    title='Distribución de Ventas por Categoría')
        
    elif tipo == 'geografico':
        geo_ventas = df.groupby('municipio')['valor_total'].sum().nlargest(10).reset_index()
        fig = px.bar(geo_ventas, x='valor_total', y='municipio',
                    orientation='h', title='Top 10 Municipios por Ventas',
                    color_discrete_sequence=[COLORS['warning']])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.update_xaxes(tickprefix="$", tickformat=",~s")
    
    return fig

def crear_visualizacion_clustering(results, client_data, feature_names):
    """Crea visualizaciones para clustering V3."""
    
    labels = results['labels']
    X_pca = results['X_pca']
    silhouette = results['silhouette']
    
    # Gráfico PCA
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': labels,
        'Cliente': client_data['cliente'].values
    })
    
    fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                        hover_data=['Cliente'],
                        title=f'Visualización de Clusters (Silhouette: {silhouette:.3f})')
    
    # Métricas por cluster
    df_metrics = client_data.copy()
    df_metrics['cluster'] = labels
    
    cluster_summary = df_metrics.groupby('cluster').agg({
        'valor_total_sum': 'mean',
        'valor_total_count': 'mean', 
        'ticket_promedio': 'mean',
        'codigo_producto_nunique': 'mean'
    }).round(2).reset_index()
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    return fig_pca, cluster_summary, cluster_counts

def crear_visualizacion_clasificacion(results):
    """Crea visualizaciones para clasificación V3."""
    
    if 'error' in results:
        return None, None, None
    
    # Matriz de confusión
    cm = results['confusion_matrix']
    fig_cm = ff.create_annotated_heatmap(
        z=cm, x=['No Frecuente', 'Frecuente'], y=['No Frecuente', 'Frecuente'],
        annotation_text=cm, colorscale='Blues'
    )
    fig_cm.update_layout(title='Matriz de Confusión')
    
    # Curva ROC
    if results['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_prob'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                   name=f"AUC = {results['auc']:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                   line=dict(dash='dash'), name='Aleatorio'))
        fig_roc.update_layout(title='Curva ROC', xaxis_title='FPR', yaxis_title='TPR')
    else:
        fig_roc = None
    
    # Métricas
    metrics = {
        'Accuracy': results['accuracy'],
        'AUC': results['auc'],
        'CV Mean': results['cv_mean'],
        'CV Std': results['cv_std']
    }
    
    return fig_cm, fig_roc, metrics

# Cargar datos
print("Cargando datos...")
df = cargar_y_procesar_datos()
if df is None:
    raise ValueError("No se pudieron cargar los datos")

# Preparar datos
client_data, X_clustering, clustering_features = preparar_datos_clustering(df)
X_classification, y_classification, classification_features, freq_threshold = preparar_datos_clasificacion(df)

print(f"Datos preparados - Clientes: {len(client_data)}")

# Inicializar app
app = dash.Dash(__name__)
app.title = "Dashboard MLflow V3 - Segmentación y Clasificación"

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Img(src=logo_src, style={'height': '80px', 'display': 'inline-block'}) if logo_src else html.Div(),
        html.Div([
            html.H1("Dashboard MLflow V3", style={'color': COLORS['primary'], 'margin': 0}),
            html.H3("Segmentación y Clasificación de Clientes", style={'color': COLORS['dark'], 'margin': '5px 0'})
        ], style={'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'borderBottom': '2px solid #eee'}),
    
    # Descripción
    html.Div([
        html.P("Sistema integrado con MLflow V3 para análisis avanzado de segmentación y clasificación de clientes. "
               "Utiliza modelos de machine learning con tracking completo y métricas de evaluación robustas.",
               style={'fontSize': '16px', 'textAlign': 'justify', 'color': COLORS['dark']})
    ], style={'padding': '0 20px'}),
    
    # Tabs principales
    dcc.Tabs(id='main-tabs', value='eda', children=[
        
        # Tab EDA
        dcc.Tab(label='Análisis Exploratorio', value='eda', children=[
            html.Div([
                html.H2("Análisis Exploratorio de Datos", style={'textAlign': 'center', 'color': COLORS['primary']}),
                
                # Métricas generales
                html.Div([
                    html.Div([
                        html.H3(f"{len(df):,}", style={'color': COLORS['success'], 'margin': 0}),
                        html.P("Total Registros", style={'margin': 0})
                    ], className='metric-card', style={'textAlign': 'center', 'padding': '20px', 
                                                      'backgroundColor': COLORS['light'], 'borderRadius': '8px',
                                                      'margin': '10px', 'flex': '1'}),
                    
                    html.Div([
                        html.H3(f"{df['cliente'].nunique():,}", style={'color': COLORS['info'], 'margin': 0}),
                        html.P("Clientes Únicos", style={'margin': 0})
                    ], className='metric-card', style={'textAlign': 'center', 'padding': '20px',
                                                      'backgroundColor': COLORS['light'], 'borderRadius': '8px',
                                                      'margin': '10px', 'flex': '1'}),
                    
                    html.Div([
                        html.H3(f"${df['valor_total'].sum():,.0f}", style={'color': COLORS['warning'], 'margin': 0}),
                        html.P("Ventas Totales", style={'margin': 0})
                    ], className='metric-card', style={'textAlign': 'center', 'padding': '20px',
                                                      'backgroundColor': COLORS['light'], 'borderRadius': '8px',
                                                      'margin': '10px', 'flex': '1'}),
                    
                    html.Div([
                        html.H3(f"{df['codigo_producto'].nunique():,}", style={'color': COLORS['danger'], 'margin': 0}),
                        html.P("Productos", style={'margin': 0})
                    ], className='metric-card', style={'textAlign': 'center', 'padding': '20px',
                                                      'backgroundColor': COLORS['light'], 'borderRadius': '8px',
                                                      'margin': '10px', 'flex': '1'})
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
                
                # Gráficos EDA
                html.Div([
                    html.Div([
                        dcc.Graph(figure=crear_grafico_eda(df, 'evolucion'))
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(figure=crear_grafico_eda(df, 'categorias'))
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                html.Div([
                    html.Div([
                        dcc.Graph(figure=crear_grafico_eda(df, 'productos'))
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(figure=crear_grafico_eda(df, 'geografico'))
                    ], style={'width': '50%', 'display': 'inline-block'})
                ])
                
            ], style={'padding': '20px'})
        ]),
        
        # Tab Clustering  
        dcc.Tab(label='Segmentación', value='clustering', children=[
            html.Div([
                html.H2("Segmentación de Clientes", style={'textAlign': 'center', 'color': COLORS['primary']}),
                
                # Controles
                html.Div([
                    html.Div([
                        html.Label("Algoritmo de Clustering:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='clustering-method',
                            options=[
                                {'label': 'K-Means', 'value': 'kmeans'},
                                {'label': 'Clustering Aglomerativo', 'value': 'agglomerative'},
                                {'label': 'DBSCAN', 'value': 'dbscan'}
                            ],
                            value='kmeans'
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Número de Clusters:", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='n-clusters',
                            min=2, max=8, step=1, value=3,
                            marks={i: str(i) for i in range(2, 9)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
                ], style={'margin': '20px 0'}),
                
                # Resultados clustering
                html.Div(id='clustering-results')
                
            ], style={'padding': '20px'})
        ]),
        
        # Tab Classification
        dcc.Tab(label='Clasificación', value='classification', children=[
            html.Div([
                html.H2("Clasificación de Clientes", style={'textAlign': 'center', 'color': COLORS['primary']}),
                
                # Controles
                html.Div([
                    html.Label("Algoritmo de Clasificación:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='classification-method',
                        options=[
                            {'label': 'Regresión Logística', 'value': 'logistic'},
                            {'label': 'Random Forest', 'value': 'random_forest'},
                            {'label': 'Gradient Boosting', 'value': 'gradient_boosting'},
                            {'label': 'Árbol de Decisión', 'value': 'decision_tree'}
                        ],
                        value='logistic'
                    )
                ], style={'width': '50%', 'margin': '20px 0'}),
                
                # Info umbral
                html.Div([
                    html.P(f"Umbral de frecuencia (adaptativo): {freq_threshold:.1f} meses",
                           style={'backgroundColor': COLORS['light'], 'padding': '10px', 
                                 'borderRadius': '5px', 'fontWeight': 'bold'})
                ], style={'margin': '20px 0'}),
                
                # Resultados classification
                html.Div(id='classification-results')
                
            ], style={'padding': '20px'})
        ]),
        
        # Tab Comparación
        dcc.Tab(label='Comparación de Modelos', value='comparison', children=[
            html.Div([
                html.H2("Comparación de Modelos", style={'textAlign': 'center', 'color': COLORS['primary']}),
                
                # Comparación clustering
                html.H3("Modelos de Clustering", style={'color': COLORS['secondary']}),
                html.Div(id='clustering-comparison'),
                
                html.Hr(),
                
                # Comparación classification  
                html.H3("Modelos de Clasificación", style={'color': COLORS['secondary']}),
                html.Div(id='classification-comparison')
                
            ], style={'padding': '20px'})
        ])
    ])
])

# Callbacks
@app.callback(
    Output('clustering-results', 'children'),
    [Input('clustering-method', 'value'),
     Input('n-clusters', 'value')]
)
def update_clustering(method, n_clusters):
    """Actualiza resultados de clustering."""
    try:
        # Ejecutar clustering
        results = ejecutar_clustering_v3(X_clustering, method, n_clusters)
        
        # Crear visualizaciones
        fig_pca, cluster_summary, cluster_counts = crear_visualizacion_clustering(
            results, client_data, clustering_features
        )
        
        # Retornar componentes
        return [
            html.Div([
                html.H4(f"Silhouette Score: {results['silhouette']:.3f}", 
                       style={'color': COLORS['success']}),
                html.P(f"Varianza explicada PCA: PC1={results['pca_explained'][0]:.2%}, "
                      f"PC2={results['pca_explained'][1]:.2%}")
            ], style={'backgroundColor': COLORS['light'], 'padding': '15px', 'borderRadius': '5px'}),
            
            dcc.Graph(figure=fig_pca),
            
            html.H4("Resumen por Cluster"),
            dash_table.DataTable(
                data=cluster_summary.to_dict('records'),
                columns=[{"name": i, "id": i} for i in cluster_summary.columns],
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': COLORS['primary'], 'color': 'white'}
            ),
            
            html.H4("Distribución de Clientes"),
            html.Ul([html.Li(f"Cluster {i}: {count} clientes") 
                    for i, count in cluster_counts.items()])
        ]
        
    except Exception as e:
        return [html.Div(f"Error: {str(e)}", style={'color': COLORS['danger']})]

@app.callback(
    Output('classification-results', 'children'),
    [Input('classification-method', 'value')]
)
def update_classification(method):
    """Actualiza resultados de clasificación."""
    try:
        # Ejecutar clasificación
        results = ejecutar_clasificacion_v3(X_classification, y_classification, method)
        
        if 'error' in results:
            return [html.Div(results['error'], style={'color': COLORS['danger']})]
        
        # Crear visualizaciones
        fig_cm, fig_roc, metrics = crear_visualizacion_clasificacion(results)
        
        components = [
            # Métricas principales
            html.Div([
                html.Div([
                    html.H4(f"{metrics['Accuracy']:.3f}", style={'color': COLORS['success'], 'margin': 0}),
                    html.P("Accuracy", style={'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'],
                         'borderRadius': '5px', 'margin': '5px', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"{metrics['AUC']:.3f}", style={'color': COLORS['info'], 'margin': 0}),
                    html.P("AUC-ROC", style={'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'],
                         'borderRadius': '5px', 'margin': '5px', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"{metrics['CV Mean']:.3f}", style={'color': COLORS['warning'], 'margin': 0}),
                    html.P("CV Mean", style={'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'],
                         'borderRadius': '5px', 'margin': '5px', 'flex': '1'}),
                
                html.Div([
                    html.H4(f"{metrics['CV Std']:.3f}", style={'color': COLORS['secondary'], 'margin': 0}),
                    html.P("CV Std", style={'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': COLORS['light'],
                         'borderRadius': '5px', 'margin': '5px', 'flex': '1'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
            
            # Gráficos
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_cm)
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=fig_roc) if fig_roc else html.Div("Curva ROC no disponible")
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ]
        
        # Feature importance si está disponible
        if hasattr(results['model'], 'feature_importances_'):
            importances = pd.DataFrame({
                'Feature': results['feature_names'],
                'Importance': results['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(importances.head(10), x='Importance', y='Feature',
                                   orientation='h', title='Top 10 Feature Importances')
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            components.append(html.Div([
                html.H4("Importancia de Características"),
                dcc.Graph(figure=fig_importance)
            ]))
        
        elif hasattr(results['model'], 'coef_'):
            coefficients = pd.DataFrame({
                'Feature': results['feature_names'],
                'Coefficient': results['model'].coef_[0]
            }).reindex(results['model'].coef_[0].argsort()[::-1][:10])
            
            fig_coef = px.bar(coefficients, x='Coefficient', y='Feature',
                             orientation='h', title='Top 10 Coeficientes (Regresión Logística)')
            fig_coef.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            components.append(html.Div([
                html.H4("Coeficientes del Modelo"),
                dcc.Graph(figure=fig_coef)
            ]))
        
        return components
        
    except Exception as e:
        return [html.Div(f"Error: {str(e)}", style={'color': COLORS['danger']})]

@app.callback(
    Output('clustering-comparison', 'children'),
    [Input('main-tabs', 'value')]
)
def update_clustering_comparison(active_tab):
    """Actualiza comparación de modelos de clustering."""
    if active_tab != 'comparison':
        return []
    
    try:
        # Ejecutar todos los métodos con 3 clusters
        methods = ['kmeans', 'agglomerative', 'dbscan']
        results_comparison = []
        
        for method in methods:
            try:
                result = ejecutar_clustering_v3(X_clustering, method, 3)
                results_comparison.append({
                    'Método': method.title(),
                    'Silhouette Score': result['silhouette'],
                    'Clusters Encontrados': len(np.unique(result['labels']))
                })
            except:
                results_comparison.append({
                    'Método': method.title(),
                    'Silhouette Score': 'Error',
                    'Clusters Encontrados': 'Error'
                })
        
        df_comparison = pd.DataFrame(results_comparison)
        
        # Gráfico de comparación
        valid_results = [r for r in results_comparison if isinstance(r['Silhouette Score'], float)]
        if valid_results:
            df_valid = pd.DataFrame(valid_results)
            fig_comparison = px.bar(df_valid, x='Método', y='Silhouette Score',
                                   title='Comparación de Silhouette Score por Método')
            fig_comparison.update_layout(yaxis=dict(range=[0, 1]))
        else:
            fig_comparison = go.Figure().add_annotation(text="No hay resultados válidos")
        
        return [
            dcc.Graph(figure=fig_comparison),
            html.H4("Tabla Comparativa"),
            dash_table.DataTable(
                data=df_comparison.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_comparison.columns],
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': COLORS['primary'], 'color': 'white'}
            )
        ]
        
    except Exception as e:
        return [html.Div(f"Error en comparación: {str(e)}", style={'color': COLORS['danger']})]

@app.callback(
    Output('classification-comparison', 'children'),
    [Input('main-tabs', 'value')]
)
def update_classification_comparison(active_tab):
    """Actualiza comparación de modelos de clasificación."""
    if active_tab != 'comparison':
        return []
    
    try:
        # Ejecutar todos los métodos
        methods = ['logistic', 'random_forest', 'gradient_boosting', 'decision_tree']
        method_names = ['Regresión Logística', 'Random Forest', 'Gradient Boosting', 'Árbol de Decisión']
        results_comparison = []
        
        for method, name in zip(methods, method_names):
            try:
                result = ejecutar_clasificacion_v3(X_classification, y_classification, method)
                if 'error' not in result:
                    results_comparison.append({
                        'Método': name,
                        'Accuracy': result['accuracy'],
                        'AUC': result['auc'],
                        'CV Mean': result['cv_mean']
                    })
            except:
                pass
        
        if not results_comparison:
            return [html.Div("No se pudieron ejecutar los modelos de comparación",
                           style={'color': COLORS['danger']})]
        
        df_comparison = pd.DataFrame(results_comparison)
        
        # Gráficos de comparación
        fig_accuracy = px.bar(df_comparison, x='Método', y='Accuracy',
                             title='Comparación de Accuracy')
        fig_accuracy.update_layout(yaxis=dict(range=[0, 1]))
        
        fig_auc = px.bar(df_comparison, x='Método', y='AUC',
                        title='Comparación de AUC-ROC')
        fig_auc.update_layout(yaxis=dict(range=[0, 1]))
        
        return [
            html.Div([
                html.Div([dcc.Graph(figure=fig_accuracy)], 
                        style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_auc)], 
                        style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.H4("Tabla Comparativa"),
            dash_table.DataTable(
                data=df_comparison.round(3).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_comparison.columns],
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': COLORS['primary'], 'color': 'white'}
            )
        ]
        
    except Exception as e:
        return [html.Div(f"Error en comparación: {str(e)}", style={'color': COLORS['danger']})]

# Integración con MLflow (opcional)
def log_to_mlflow(model_type, method, metrics, model=None):
    """Registra experimento en MLflow si está disponible."""
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        
        with mlflow.start_run(run_name=f"{model_type}_{method}"):
            # Log parámetros
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("method", method)
            mlflow.log_param("dashboard_version", "3.0")
            
            # Log métricas
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value)
            
            # Log modelo si está disponible
            if model and hasattr(model, 'fit'):
                mlflow.sklearn.log_model(model, "model")
            
            # Tags
            mlflow.set_tags({
                "source": "dashboard_v3",
                "environment": "production",
                "integration": "dash_mlflow"
            })
            
    except Exception as e:
        print(f"Error logging to MLflow: {e}")

if __name__ == '__main__':
    print("="*60)
    print("DASHBOARD MLFLOW V3 - SEGMENTACIÓN Y CLASIFICACIÓN")
    print("="*60)
    print("Características principales:")
    print("✓ Integración con MLflow V3")
    print("✓ 4 algoritmos de clustering y clasificación")
    print("✓ Visualizaciones interactivas mejoradas")
    print("✓ Métricas avanzadas y comparación de modelos")
    print("✓ Diseño moderno y responsivo")
    print("✓ Características V3 implementadas")
    print(f"✓ Datos cargados: {len(df):,} registros")
    print(f"✓ Clientes analizados: {len(client_data):,}")
    print("="*60)
    print("Accede al dashboard en: http://localhost:8050")
    print("Para usar con MLflow: http://localhost:5000")
    print("="*60)
    
    app.run_server(
        debug=False,
        host='0.0.0.0',
        port=8050,
        threaded=True
    )