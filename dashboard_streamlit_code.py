# Dashboard Multiconcentrados Boyacá - Segmentación y Análisis Comercial
# Desarrollado por: Daniela Yañez & Eduar Riaño
# Universidad de los Andes - MIIA

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Multiconcentrados Boyacá",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E4057 0%, #048A81 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #048A81;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .custom-subheader {
        color: #2E4057;
        font-weight: 600;
        border-bottom: 2px solid #048A81;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares para cargar y procesar datos
@st.cache_data
def load_data():
    """Carga y procesa los datos de ventas"""
    try:
        # Cargar datos principales - igual que en EDA
        df_ventas = pd.read_excel('resumen por item final.xlsx', sheet_name='Hoja1')
        
        # Limpiar y preparar datos - igual que en EDA
        df_ventas = df_ventas.dropna(subset=['CLIENTE', 'VALOR TOTAL'])
        df_ventas['VALOR TOTAL'] = pd.to_numeric(df_ventas['VALOR TOTAL'], errors='coerce')
        df_ventas['CANTIDAD'] = pd.to_numeric(df_ventas['CANTIDAD'], errors='coerce')
        
        # Crear columna de fecha - EXACTAMENTE igual que en EDA
        mes_numero = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 
                      'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 
                      'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}
        
        df_ventas['mes_num'] = df_ventas['MES'].map(mes_numero)
        
        # NO usar pd.to_datetime aquí - crear fecha simple solo para ordenamiento
        df_ventas['año_mes'] = df_ventas['AÑO'].astype(str) + '-' + df_ventas['MES']
        
        # Cargar datos de clientes si existe
        try:
            df_clientes = pd.read_excel('resumen por item final.xlsx', sheet_name='Hoja2')
        except:
            df_clientes = None
            st.warning("No se pudo cargar la hoja de clientes (Hoja2)")
        
        return df_ventas, df_clientes
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None

def calculate_rfm_metrics(df_ventas):
    """Calcula métricas RFM para segmentación"""
    try:
        # Usar fecha de referencia simple
        fecha_referencia = datetime(2025, 4, 30)
        
        # Agrupar por cliente y calcular métricas
        rfm = df_ventas.groupby('CLIENTE').agg({
            'AÑO': 'max',
            'mes_num': 'max',
            'VALOR TOTAL': ['count', 'sum'],
            'CANTIDAD': 'sum'
        }).reset_index()
        
        rfm.columns = ['CLIENTE', 'ultimo_año', 'ultimo_mes', 'frecuencia', 'valor_monetario', 'unidades']
        
        # Calcular recency en meses
        rfm['recency'] = (2025 - rfm['ultimo_año']) * 12 + (4 - rfm['ultimo_mes'].fillna(4))
        rfm['recency'] = rfm['recency'].clip(lower=0)  # No permitir valores negativos
        
        # Calcular quintiles para segmentación (evitando errores)
        try:
            rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        except:
            rfm['R_score'] = pd.cut(rfm['recency'], 5, labels=[5,4,3,2,1])
            
        try:
            rfm['F_score'] = pd.qcut(rfm['frecuencia'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            rfm['F_score'] = pd.cut(rfm['frecuencia'], 5, labels=[1,2,3,4,5])
            
        try:
            rfm['M_score'] = pd.qcut(rfm['valor_monetario'], 5, labels=[1,2,3,4,5], duplicates='drop')
        except:
            rfm['M_score'] = pd.cut(rfm['valor_monetario'], 5, labels=[1,2,3,4,5])
        
        # Crear segmentos
        rfm['RFM_Score'] = (rfm['R_score'].astype(str) + 
                           rfm['F_score'].astype(str) + 
                           rfm['M_score'].astype(str))
        
        def segment_customers(row):
            score = str(row['RFM_Score'])
            if score in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Campeones'
            elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Leales'
            elif score in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potencial Lealtad'
            elif score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Nuevos Clientes'
            elif score in ['155', '154', '144', '214', '215', '115', '114']:
                return 'En Riesgo'
            elif score in ['155', '254', '245']:
                return 'No se pueden perder'
            else:
                return 'Otros'
        
        rfm['Segmento'] = rfm.apply(segment_customers, axis=1)
        
        return rfm
        
    except Exception as e:
        st.error(f"Error en cálculo RFM: {str(e)}")
        return pd.DataFrame()

def create_temporal_chart(df_ventas, periodo_filtro):
    """Crea gráfico de evolución temporal"""
    df_temp = df_ventas.copy()
    
    # Aplicar filtro de período sin usar fechas complejas
    if periodo_filtro == "2024":
        df_temp = df_temp[df_temp['AÑO'] == 2024]
    elif periodo_filtro == "2025":
        df_temp = df_temp[df_temp['AÑO'] == 2025]
    # Para "Todos" y "Último año" usar todos los datos disponibles
    
    # Agrupar por año y mes
    ventas_mensual = df_temp.groupby(['AÑO', 'MES']).agg({
        'VALOR TOTAL': 'sum',
        'CLIENTE': 'nunique'
    }).reset_index()
    
    # Crear etiqueta de período
    ventas_mensual['periodo'] = ventas_mensual['AÑO'].astype(str) + '-' + ventas_mensual['MES']
    
    # Crear gráfico
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=ventas_mensual['periodo'], 
                  y=ventas_mensual['VALOR TOTAL'],
                  mode='lines+markers',
                  name='Ventas',
                  line=dict(color='#048A81', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=ventas_mensual['periodo'], 
                  y=ventas_mensual['CLIENTE'],
                  mode='lines+markers',
                  name='Clientes Únicos',
                  line=dict(color='#2E4057', width=2)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Período")
    fig.update_yaxes(title_text="Ventas (Pesos)", secondary_y=False)
    fig.update_yaxes(title_text="Número de Clientes", secondary_y=True)
    
    fig.update_layout(
        title="Evolución Temporal de Ventas y Clientes",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_geographic_chart(df_ventas):
    """Crea gráfico de distribución geográfica"""
    geo_data = df_ventas.groupby(['DEPARTAMENTO', 'MUNICIPIOS']).agg({
        'VALOR TOTAL': 'sum',
        'CLIENTE': 'nunique'
    }).reset_index()
    
    geo_data = geo_data.sort_values('VALOR TOTAL', ascending=False).head(15)
    
    fig = px.bar(geo_data, 
                 x='VALOR TOTAL', 
                 y='MUNICIPIOS',
                 color='DEPARTAMENTO',
                 title="Top 15 Municipios por Ventas",
                 labels={'VALOR TOTAL': 'Ventas (Pesos)', 'MUNICIPIOS': 'Municipio'},
                 height=500)
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig

def create_product_analysis(df_ventas):
    """Análisis de productos y categorías"""
    # Por categorías
    cat_data = df_ventas.groupby('CATEGORIA').agg({
        'VALOR TOTAL': 'sum',
        'CLIENTE': 'nunique'
    }).reset_index()
    
    cat_data = cat_data.sort_values('VALOR TOTAL', ascending=False)
    
    fig_cat = px.pie(cat_data, 
                     values='VALOR TOTAL', 
                     names='CATEGORIA',
                     title="Distribución de Ventas por Categoría")
    
    # Top productos - CORREGIR AQUÍ
    prod_data = df_ventas.groupby(['CODIGO', 'NOMBRE DE ARTICULO']).agg({
        'VALOR TOTAL': 'sum'
    }).reset_index()
    
    prod_data = prod_data.sort_values('VALOR TOTAL', ascending=False).head(10)
    
    # Convertir CODIGO a string antes de concatenar
    prod_data['CODIGO_str'] = prod_data['CODIGO'].astype(str)
    prod_data['NOMBRE_corto'] = prod_data['NOMBRE DE ARTICULO'].astype(str).str[:30]
    prod_data['producto'] = prod_data['CODIGO_str'] + ' - ' + prod_data['NOMBRE_corto']
    
    fig_prod = px.bar(prod_data,
                      x='VALOR TOTAL',
                      y='producto',
                      title="Top 10 Productos por Ventas",
                      labels={'VALOR TOTAL': 'Ventas (Pesos)', 'producto': 'Producto'})
    
    fig_prod.update_layout(yaxis={'categoryorder':'total ascending'})
    
    return fig_cat, fig_prod

def create_rfm_visualization(rfm_data):
    """Visualización de segmentación RFM"""
    # Gráfico de segmentos
    segment_counts = rfm_data['Segmento'].value_counts()
    
    fig_segments = px.bar(x=segment_counts.values,
                         y=segment_counts.index,
                         title="Distribución de Clientes por Segmento RFM",
                         labels={'x': 'Número de Clientes', 'y': 'Segmento'})
    
    fig_segments.update_layout(yaxis={'categoryorder':'total ascending'})
    
    # Análisis de valor por segmento
    segment_value = rfm_data.groupby('Segmento').agg({
        'valor_monetario': 'sum',
        'frecuencia': 'mean',
        'recency': 'mean'
    }).reset_index()
    
    fig_value = px.scatter(segment_value,
                          x='frecuencia',
                          y='valor_monetario',
                          size='valor_monetario',
                          color='Segmento',
                          title="Análisis de Valor por Segmento (Frecuencia vs Valor Monetario)")
    
    return fig_segments, fig_value

# Función principal del dashboard
def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>Dashboard Multiconcentrados Boyacá</h1>
        <p style="text-align: center; color: white; margin: 0;">
            Segmentación y Clasificación de Clientes para Optimización Comercial
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        df_ventas, df_clientes = load_data()
    
    if df_ventas is None:
        st.error("No se pudieron cargar los datos. Verifique que el archivo 'resumen por item final.xlsx' esté disponible.")
        return
    
    # Sidebar para filtros
    st.sidebar.header("Filtros de Análisis")
    
    # Filtros principales
    periodo_filtro = st.sidebar.selectbox(
        "Período de Análisis:",
        ["Todos", "Último año", "2025", "2024"]
    )
    
    departamentos_disponibles = ["Todos"] + sorted(df_ventas['DEPARTAMENTO'].unique().tolist())
    depto_filtro = st.sidebar.selectbox(
        "Departamento:",
        departamentos_disponibles
    )
    
    categorias_disponibles = ["Todas"] + sorted(df_ventas['CATEGORIA'].unique().tolist())
    categoria_filtro = st.sidebar.selectbox(
        "Categoría de Producto:",
        categorias_disponibles
    )
    
    # Aplicar filtros
    df_filtered = df_ventas.copy()
    
    if depto_filtro != "Todos":
        df_filtered = df_filtered[df_filtered['DEPARTAMENTO'] == depto_filtro]
    
    if categoria_filtro != "Todas":
        df_filtered = df_filtered[df_filtered['CATEGORIA'] == categoria_filtro]
    
    # Calcular métricas principales
    total_ventas = df_filtered['VALOR TOTAL'].sum()
    total_clientes = df_filtered['CLIENTE'].nunique()
    ticket_promedio = df_filtered['VALOR TOTAL'].mean()
    total_transacciones = len(df_filtered)
    
    # Calcular clientes recurrentes
    freq_clientes = df_filtered.groupby('CLIENTE').size()
    clientes_recurrentes = (freq_clientes > 1).sum()
    pct_recurrentes = (clientes_recurrentes / total_clientes) * 100 if total_clientes > 0 else 0
    
    # KPIs principales
    st.markdown('<p class="custom-subheader">Métricas Principales</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Ventas Totales",
            value=f"${total_ventas:,.0f}",
            delta=f"{total_transacciones:,} transacciones"
        )
    
    with col2:
        st.metric(
            label="Clientes Únicos",
            value=f"{total_clientes:,}",
            delta=f"{clientes_recurrentes:,} recurrentes"
        )
    
    with col3:
        st.metric(
            label="Ticket Promedio",
            value=f"${ticket_promedio:,.0f}",
            delta=f"{total_transacciones/total_clientes:.1f} trans./cliente" if total_clientes > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            label="% Clientes Recurrentes",
            value=f"{pct_recurrentes:.1f}%",
            delta="Meta: >70%"
        )
    
    # Gráficos principales
    st.markdown('<p class="custom-subheader">Análisis Temporal y Geográfico</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temporal = create_temporal_chart(df_filtered, periodo_filtro)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with col2:
        fig_geo = create_geographic_chart(df_filtered)
        st.plotly_chart(fig_geo, use_container_width=True)
    
    # Análisis de productos
    st.markdown('<p class="custom-subheader">Análisis de Productos y Categorías</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Por Categoría", "Top Productos"])
    
    with tab1:
        fig_cat, fig_prod = create_product_analysis(df_filtered)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with tab2:
        st.plotly_chart(fig_prod, use_container_width=True)
    
    # Segmentación RFM
    st.markdown('<p class="custom-subheader">Segmentación de Clientes (RFM)</p>', unsafe_allow_html=True)
    
    with st.spinner('Calculando segmentación RFM...'):
        rfm_data = calculate_rfm_metrics(df_filtered)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_segments, fig_value = create_rfm_visualization(rfm_data)
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_value, use_container_width=True)
    
    # Tabla de segmentos
    st.markdown('<p class="custom-subheader">Análisis Detallado por Segmento</p>', unsafe_allow_html=True)
    
    segment_summary = rfm_data.groupby('Segmento').agg({
        'CLIENTE': 'count',
        'valor_monetario': ['sum', 'mean'],
        'frecuencia': 'mean',
        'recency': 'mean'
    }).round(2)
    
    segment_summary.columns = ['Cantidad_Clientes', 'Valor_Total', 'Valor_Promedio', 'Freq_Promedio', 'Recency_Promedio']
    segment_summary['Participacion_Ventas'] = (segment_summary['Valor_Total'] / segment_summary['Valor_Total'].sum() * 100).round(1)
    
    st.dataframe(segment_summary, use_container_width=True)
    
    # Top clientes por segmento
    st.markdown('<p class="custom-subheader">Top Clientes por Segmento</p>', unsafe_allow_html=True)
    
    segmento_seleccionado = st.selectbox(
        "Seleccionar segmento:",
        rfm_data['Segmento'].unique()
    )
    
    top_clientes_segmento = rfm_data[rfm_data['Segmento'] == segmento_seleccionado].nlargest(10, 'valor_monetario')
    
    st.dataframe(
        top_clientes_segmento[['CLIENTE', 'valor_monetario', 'frecuencia', 'recency', 'RFM_Score']],
        use_container_width=True
    )
    
    # Insights y recomendaciones
    st.markdown('<p class="custom-subheader">Insights y Recomendaciones</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Oportunidades Detectadas:**
        
        - Alto porcentaje de clientes únicos (oportunidad de fidelización)
        - Concentración geográfica permite estrategias focalizadas
        - Segmentos de alto valor identificados para programas VIP
        - Productos estrella con potencial de cross-selling
        """)
    
    with col2:
        st.success("""
        **Recomendaciones Comerciales:**
        
        - Implementar programa de fidelización para clientes "En Riesgo"
        - Desarrollar ofertas personalizadas por segmento
        - Reforzar presencia en municipios de alto potencial
        - Crear campañas de reactivación para clientes inactivos
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Dashboard de Segmentación y Análisis Comercial v1.0</strong></p>
        <p>Desarrollado por: Daniela Yañez & Eduar Riaño | Universidad de los Andes - MIIA</p>
        <p>Última actualización: {}</p>
    </div>
    """.format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()