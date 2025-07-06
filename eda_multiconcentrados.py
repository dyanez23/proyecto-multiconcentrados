# Análisis Exploratorio de Datos - Multiconcentrados Boyacá
# Proyecto: Segmentación y Clasificación de Clientes para Optimización Comercial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("=" * 80)
print("ANÁLISIS EXPLORATORIO DE DATOS - MULTICONCENTRADOS BOYACÁ")
print("=" * 80)

# ============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

# Cargar datos principales de ventas
df_ventas = pd.read_excel('resumen por item final.xlsx', sheet_name='Hoja1')

# Cargar datos de clientes
df_clientes = pd.read_excel('resumen por item final.xlsx', sheet_name='Hoja2')

print(f"\nDatos cargados exitosamente:")
print(f"- Registros de ventas: {df_ventas.shape[0]:,} transacciones")
print(f"- Información de clientes: {df_clientes.shape[0]:,} clientes únicos")

# ============================================================================
# 2. EXPLORACIÓN INICIAL DE LA ESTRUCTURA DE DATOS
# ============================================================================

print("\n" + "=" * 50)
print("ESTRUCTURA Y CALIDAD DE LOS DATOS")
print("=" * 50)

# Información básica del dataset de ventas
print("\nEstructura del dataset de ventas:")
print(f"Dimensiones: {df_ventas.shape[0]} filas × {df_ventas.shape[1]} columnas")
print(f"Período analizado: {df_ventas['AÑO'].min()} - {df_ventas['AÑO'].max()}")

# Verificar tipos de datos
print("\nTipos de datos:")
for col in df_ventas.columns:
    dtype = str(df_ventas[col].dtype)
    null_count = df_ventas[col].isnull().sum()
    null_pct = (null_count / len(df_ventas)) * 100
    print(f"  {col:<25} : {dtype:<12} | Nulos: {null_count:>5} ({null_pct:>5.1f}%)")

# Período temporal cubierto
meses_unicos = df_ventas.groupby(['AÑO', 'MES']).size().reset_index(name='transacciones')
print(f"\nPeríodo temporal:")
print(f"- Años: {sorted(df_ventas['AÑO'].unique())}")
print(f"- Meses con transacciones: {len(meses_unicos)} meses")

# ============================================================================
# 3. ANÁLISIS DE CLIENTES
# ============================================================================

print("\n" + "=" * 50)
print("ANÁLISIS DE BASE DE CLIENTES")
print("=" * 50)

# Estadísticas generales de clientes
clientes_unicos = df_ventas['CLIENTE'].nunique()
total_transacciones = len(df_ventas)

print(f"\nMétricas generales:")
print(f"- Total de clientes únicos: {clientes_unicos:,}")
print(f"- Total de transacciones: {total_transacciones:,}")
print(f"- Promedio de transacciones por cliente: {total_transacciones/clientes_unicos:.1f}")

# Análisis de frecuencia de compra por cliente
freq_clientes = df_ventas.groupby('CLIENTE').agg({
    'VALOR TOTAL': ['count', 'sum', 'mean'],
    '# FACT.': 'sum'
}).round(2)

freq_clientes.columns = ['transacciones', 'valor_total', 'ticket_promedio', 'facturas']
freq_clientes = freq_clientes.reset_index()

print(f"\nDistribución de clientes por frecuencia de compra:")
freq_bins = [1, 2, 5, 10, 20, float('inf')]
freq_labels = ['1 transacción', '2-4 trans.', '5-9 trans.', '10-19 trans.', '20+ trans.']
freq_clientes['categoria_frecuencia'] = pd.cut(freq_clientes['transacciones'], 
                                              bins=freq_bins, labels=freq_labels, right=False)

freq_dist = freq_clientes['categoria_frecuencia'].value_counts()
for categoria, count in freq_dist.items():
    pct = (count / len(freq_clientes)) * 100
    print(f"  {categoria:<15}: {count:>4} clientes ({pct:>5.1f}%)")

# Top 10 clientes por valor
top_clientes = freq_clientes.nlargest(10, 'valor_total')
print(f"\nTop 10 clientes por valor total de compras:")
for i, row in top_clientes.iterrows():
    print(f"  Cliente {row['CLIENTE']}: ${row['valor_total']:>12,.0f} "
          f"({row['transacciones']:>2} trans. - ${row['ticket_promedio']:>8,.0f} promedio)")

# Concentración de ingresos
total_ingresos = freq_clientes['valor_total'].sum()
top_20_pct = freq_clientes.nlargest(int(len(freq_clientes) * 0.2), 'valor_total')['valor_total'].sum()
top_10_pct = freq_clientes.nlargest(int(len(freq_clientes) * 0.1), 'valor_total')['valor_total'].sum()

print(f"\nConcentración de ingresos:")
print(f"- Top 20% de clientes genera: {(top_20_pct/total_ingresos)*100:.1f}% de los ingresos")
print(f"- Top 10% de clientes genera: {(top_10_pct/total_ingresos)*100:.1f}% de los ingresos")

# ============================================================================
# 4. ANÁLISIS TEMPORAL
# ============================================================================

print("\n" + "=" * 50)
print("ANÁLISIS TEMPORAL DE VENTAS")
print("=" * 50)

# Crear columna de fecha para análisis temporal
df_ventas['fecha_periodo'] = df_ventas['AÑO'].astype(str) + '-' + df_ventas['MES']

# Ventas por mes
ventas_mensual = df_ventas.groupby(['AÑO', 'MES']).agg({
    'VALOR TOTAL': 'sum',
    'CANTIDAD': 'sum',
    'CLIENTE': 'nunique',
    '# FACT.': 'sum'
}).round(0)

ventas_mensual.columns = ['ventas_pesos', 'unidades_vendidas', 'clientes_activos', 'facturas']

print(f"\nEvolución mensual de ventas:")
print(ventas_mensual.to_string())

# Estadísticas mensuales
promedio_mensual = ventas_mensual['ventas_pesos'].mean()
mes_mayor_venta = ventas_mensual['ventas_pesos'].idxmax()
mes_menor_venta = ventas_mensual['ventas_pesos'].idxmin()

print(f"\nEstadísticas temporales:")
print(f"- Promedio mensual de ventas: ${promedio_mensual:,.0f}")
print(f"- Mes con mayores ventas: {mes_mayor_venta[1]} {mes_mayor_venta[0]} (${ventas_mensual.loc[mes_mayor_venta, 'ventas_pesos']:,.0f})")
print(f"- Mes con menores ventas: {mes_menor_venta[1]} {mes_menor_venta[0]} (${ventas_mensual.loc[mes_menor_venta, 'ventas_pesos']:,.0f})")

# Análisis de estacionalidad
estacionalidad = df_ventas.groupby('MES')['VALOR TOTAL'].sum().sort_values(ascending=False)
print(f"\nRanking de meses por volumen de ventas:")
for mes, valor in estacionalidad.items():
    pct = (valor / estacionalidad.sum()) * 100
    print(f"  {mes:<10}: ${valor:>12,.0f} ({pct:>5.1f}%)")

# ============================================================================
# 5. ANÁLISIS DE PRODUCTOS Y CATEGORÍAS
# ============================================================================

print("\n" + "=" * 50)
print("ANÁLISIS DE PRODUCTOS Y CATEGORÍAS")
print("=" * 50)

# Análisis por categorías principales
cat_ventas = df_ventas.groupby('CATEGORIA').agg({
    'VALOR TOTAL': 'sum',
    'CANTIDAD': 'sum',
    'CLIENTE': 'nunique'
}).round(0)

cat_ventas.columns = ['ventas_pesos', 'unidades', 'clientes']
cat_ventas['participacion'] = (cat_ventas['ventas_pesos'] / cat_ventas['ventas_pesos'].sum() * 100).round(1)
cat_ventas = cat_ventas.sort_values('ventas_pesos', ascending=False)

print(f"\nVentas por categoría de producto:")
for categoria, row in cat_ventas.iterrows():
    categoria_str = str(categoria).strip() if isinstance(categoria, str) else str(categoria)
    print(f"  {categoria_str:<30}: ${row['ventas_pesos']:>12,.0f} "
          f"({row['participacion']:>5.1f}%) - {row['clientes']:>3} clientes")

# Top productos por ventas
top_productos = df_ventas.groupby(['CODIGO', 'NOMBRE DE ARTICULO']).agg({
    'VALOR TOTAL': 'sum',
    'CANTIDAD': 'sum',
    'CLIENTE': 'nunique'
}).round(0)

top_productos.columns = ['ventas_pesos', 'unidades', 'clientes']
top_productos = top_productos.sort_values('ventas_pesos', ascending=False).head(10)

print(f"\nTop 10 productos por valor de ventas:")
for (codigo, nombre), row in top_productos.iterrows():
    codigo_str = str(codigo).strip() if isinstance(codigo, str) else str(codigo)
    nombre_str = str(nombre).strip() if isinstance(nombre, str) else str(nombre)
    print(f"  {codigo_str}: {nombre_str[:35]:<35} ${row['ventas_pesos']:>10,.0f}")

# Análisis de diversidad de productos por cliente
productos_por_cliente = df_ventas.groupby('CLIENTE')['CODIGO'].nunique().reset_index()
productos_por_cliente.columns = ['CLIENTE', 'productos_unicos']

print(f"\nDiversidad de productos por cliente:")
print(f"- Promedio de productos únicos por cliente: {productos_por_cliente['productos_unicos'].mean():.1f}")
print(f"- Cliente con mayor diversidad: {productos_por_cliente['productos_unicos'].max()} productos únicos")
print(f"- Clientes que compran un solo producto: {(productos_por_cliente['productos_unicos'] == 1).sum()} ({(productos_por_cliente['productos_unicos'] == 1).mean()*100:.1f}%)")

# ============================================================================
# 6. ANÁLISIS GEOGRÁFICO
# ============================================================================

print("\n" + "=" * 50)
print("ANÁLISIS GEOGRÁFICO")
print("=" * 50)

# Ventas por departamento
dept_ventas = df_ventas.groupby('DEPARTAMENTO').agg({
    'VALOR TOTAL': 'sum',
    'CLIENTE': 'nunique',
    '# FACT.': 'sum'
}).round(0)

dept_ventas.columns = ['ventas_pesos', 'clientes', 'facturas']
dept_ventas['participacion'] = (dept_ventas['ventas_pesos'] / dept_ventas['ventas_pesos'].sum() * 100).round(1)
dept_ventas = dept_ventas.sort_values('ventas_pesos', ascending=False)

print(f"\nVentas por departamento:")
for depto, row in dept_ventas.iterrows():
    depto_str = str(depto).strip() if isinstance(depto, str) else str(depto)
    print(f"  {depto_str:<15}: ${row['ventas_pesos']:>12,.0f} "
          f"({row['participacion']:>5.1f}%) - {row['clientes']:>3} clientes")

# Top municipios por ventas
mun_ventas = df_ventas.groupby(['DEPARTAMENTO', 'MUNICIPIOS']).agg({
    'VALOR TOTAL': 'sum',
    'CLIENTE': 'nunique'
}).round(0)

mun_ventas.columns = ['ventas_pesos', 'clientes']
mun_ventas = mun_ventas.sort_values('ventas_pesos', ascending=False).head(10)

print(f"\nTop 10 municipios por valor de ventas:")
for (depto, municipio), row in mun_ventas.iterrows():
    depto_str = str(depto).strip() if isinstance(depto, str) else str(depto)
    municipio_str = str(municipio).strip() if isinstance(municipio, str) else str(municipio)
    print(f"  {municipio_str}, {depto_str}: ${row['ventas_pesos']:>10,.0f} - {row['clientes']:>2} clientes")

# ============================================================================
# 7. ANÁLISIS FINANCIERO DETALLADO
# ============================================================================

print("\n" + "=" * 50)
print("ANÁLISIS FINANCIERO")
print("=" * 50)

# Estadísticas generales
total_ventas = df_ventas['VALOR TOTAL'].sum()
total_descuentos = df_ventas['TOTAL DESCUENTO'].sum()
ticket_promedio = df_ventas['VALOR TOTAL'].mean()

print(f"\nMétricas financieras generales:")
print(f"- Total ventas período: ${total_ventas:,.0f}")
print(f"- Total descuentos otorgados: ${total_descuentos:,.0f}")
print(f"- Ticket promedio por transacción: ${ticket_promedio:,.0f}")
print(f"- Descuento promedio como % de ventas: {(total_descuentos/total_ventas)*100:.2f}%")

# Distribución de tickets
print(f"\nDistribución de valores de transacción:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    valor = np.percentile(df_ventas['VALOR TOTAL'], p)
    print(f"  Percentil {p:>2}: ${valor:>10,.0f}")

# Análisis de tickets por rango
ticket_bins = [0, 50000, 100000, 250000, 500000, 1000000, float('inf')]
ticket_labels = ['<$50K', '$50K-$100K', '$100K-$250K', '$250K-$500K', '$500K-$1M', '>$1M']
df_ventas['rango_ticket'] = pd.cut(df_ventas['VALOR TOTAL'], bins=ticket_bins, labels=ticket_labels)

ticket_dist = df_ventas.groupby('rango_ticket').agg({
    'VALOR TOTAL': ['count', 'sum']
}).round(0)

ticket_dist.columns = ['transacciones', 'valor_total']
ticket_dist['participacion_trans'] = (ticket_dist['transacciones'] / len(df_ventas) * 100).round(1)
ticket_dist['participacion_valor'] = (ticket_dist['valor_total'] / total_ventas * 100).round(1)

print(f"\nDistribución por rangos de ticket:")
for rango, row in ticket_dist.iterrows():
    print(f"  {rango:<12}: {row['transacciones']:>5} trans. ({row['participacion_trans']:>5.1f}%) "
          f"- ${row['valor_total']:>12,.0f} ({row['participacion_valor']:>5.1f}%)")

# ============================================================================
# 8. IDENTIFICACIÓN DE PATRONES CLAVE
# ============================================================================

print("\n" + "=" * 50)
print("PATRONES Y INSIGHTS CLAVE IDENTIFICADOS")
print("=" * 50)

# Análisis de recurrencia de clientes
clientes_multiples = freq_clientes[freq_clientes['transacciones'] > 1]
pct_recurrentes = (len(clientes_multiples) / len(freq_clientes)) * 100
valor_recurrentes = clientes_multiples['valor_total'].sum()
pct_valor_recurrentes = (valor_recurrentes / total_ingresos) * 100

print(f"\n1. RECURRENCIA DE CLIENTES:")
print(f"   - {pct_recurrentes:.1f}% de los clientes son recurrentes (más de 1 compra)")
print(f"   - Los clientes recurrentes generan {pct_valor_recurrentes:.1f}% del valor total")
print(f"   - Ticket promedio clientes recurrentes: ${clientes_multiples['ticket_promedio'].mean():,.0f}")
print(f"   - Ticket promedio clientes únicos: ${freq_clientes[freq_clientes['transacciones']==1]['ticket_promedio'].mean():,.0f}")

# Análisis de comportamiento estacional
cv_estacional = estacionalidad.std() / estacionalidad.mean()
print(f"\n2. ESTACIONALIDAD:")
print(f"   - Coeficiente de variación mensual: {cv_estacional:.3f}")
if cv_estacional > 0.3:
    print(f"   - Las ventas muestran alta estacionalidad")
else:
    print(f"   - Las ventas muestran relativa estabilidad mensual")

# Concentración geográfica
pct_boyaca = (dept_ventas.loc['Boyaca', 'ventas_pesos'] / total_ventas * 100) if 'Boyaca' in dept_ventas.index else 0
print(f"\n3. CONCENTRACIÓN GEOGRÁFICA:")
print(f"   - Boyacá representa {pct_boyaca:.1f}% de las ventas totales")
print(f"   - Presencia en {df_ventas['DEPARTAMENTO'].nunique()} departamentos")
print(f"   - Presencia en {df_ventas['MUNICIPIOS'].nunique()} municipios")

# Análisis de portafolio
productos_top_80 = top_productos['ventas_pesos'].cumsum() / top_productos['ventas_pesos'].sum()
productos_80_20 = (productos_top_80 <= 0.8).sum()
print(f"\n4. CONCENTRACIÓN DE PRODUCTOS:")
print(f"   - {productos_80_20} productos representan ~80% de las ventas")
print(f"   - Total de productos únicos: {df_ventas['CODIGO'].nunique()}")
print(f"   - Concentración del portafolio: ALTA" if productos_80_20 < 20 else "   - Concentración del portafolio: MEDIA")

# ============================================================================
# 9. VISUALIZACIONES PRINCIPALES
# ============================================================================

print(f"\n" + "=" * 50)
print("GENERANDO VISUALIZACIONES PRINCIPALES")
print("=" * 50)

# Configurar el layout de subplots para múltiples gráficos
fig = plt.figure(figsize=(20, 24))

# 1. Evolución temporal de ventas
plt.subplot(4, 2, 1)
ventas_plot = ventas_mensual.reset_index()
ventas_plot['periodo'] = ventas_plot['AÑO'].astype(str) + '-' + ventas_plot['MES']
plt.plot(range(len(ventas_plot)), ventas_plot['ventas_pesos'], 'b-', linewidth=2, marker='o')
plt.title('Evolución Mensual de Ventas', fontsize=14, fontweight='bold')
plt.xlabel('Período')
plt.ylabel('Ventas (Pesos)')
plt.xticks(range(len(ventas_plot)), ventas_plot['periodo'], rotation=45)
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='y')

# 2. Distribución de clientes por frecuencia
plt.subplot(4, 2, 2)
freq_dist_values = freq_dist.values
freq_dist_labels = freq_dist.index
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.pie(freq_dist_values, labels=freq_dist_labels, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Distribución de Clientes por Frecuencia de Compra', fontsize=14, fontweight='bold')

# 3. Top 10 productos por ventas
plt.subplot(4, 2, 3)
top_10_productos = top_productos.head(10)
nombres_cortos = []
for _, nombre in top_10_productos.index:
    nombre_str = str(nombre) if nombre is not None else "Sin nombre"
    nombres_cortos.append(nombre_str[:25] + '...' if len(nombre_str) > 25 else nombre_str)

plt.barh(range(len(top_10_productos)), top_10_productos['ventas_pesos'], color='skyblue')
plt.yticks(range(len(top_10_productos)), nombres_cortos)
plt.xlabel('Ventas (Pesos)')
plt.title('Top 10 Productos por Valor de Ventas', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 4. Ventas por categoría
plt.subplot(4, 2, 4)
cat_names = []
for cat in cat_ventas.index:
    cat_str = str(cat).strip() if isinstance(cat, str) else str(cat)
    cat_names.append(cat_str[:20])

plt.bar(range(len(cat_ventas)), cat_ventas['ventas_pesos'], color='lightcoral')
plt.xticks(range(len(cat_ventas)), cat_names, rotation=45)
plt.ylabel('Ventas (Pesos)')
plt.title('Ventas por Categoría de Producto', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 5. Distribución de tickets
plt.subplot(4, 2, 5)
plt.hist(df_ventas['VALOR TOTAL'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
plt.xlabel('Valor de Transacción (Pesos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Valores de Transacción', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 6. Top municipios por ventas
plt.subplot(4, 2, 6)
top_municipios = mun_ventas.head(8)
municipios_labels = [f"{mun}, {dep}" for (dep, mun), _ in top_municipios.iterrows()]
municipios_labels = [label[:20] + '...' if len(label) > 20 else label for label in municipios_labels]
plt.barh(range(len(top_municipios)), top_municipios['ventas_pesos'], color='orange')
plt.yticks(range(len(top_municipios)), municipios_labels)
plt.xlabel('Ventas (Pesos)')
plt.title('Top 8 Municipios por Valor de Ventas', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 7. Concentración de clientes (Curva de Pareto)
plt.subplot(4, 2, 7)
freq_clientes_sorted = freq_clientes.sort_values('valor_total', ascending=False).reset_index(drop=True)
freq_clientes_sorted['pct_acumulado'] = (freq_clientes_sorted['valor_total'].cumsum() / 
                                        freq_clientes_sorted['valor_total'].sum() * 100)
freq_clientes_sorted['cliente_pct'] = (freq_clientes_sorted.index + 1) / len(freq_clientes_sorted) * 100

plt.plot(freq_clientes_sorted['cliente_pct'], freq_clientes_sorted['pct_acumulado'], 
         'b-', linewidth=2)
plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% de ventas')
plt.axvline(x=20, color='r', linestyle='--', alpha=0.7, label='20% de clientes')
plt.xlabel('Porcentaje de Clientes (%)')
plt.ylabel('Porcentaje Acumulado de Ventas (%)')
plt.title('Concentración de Ventas por Cliente (Pareto)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 8. Distribución por rangos de ticket
plt.subplot(4, 2, 8)
rangos = ticket_dist.index
valores = ticket_dist['participacion_valor']
plt.bar(range(len(rangos)), valores, color='purple', alpha=0.7)
plt.xticks(range(len(rangos)), rangos, rotation=45)
plt.ylabel('Participación en Ventas (%)')
plt.title('Participación por Rangos de Ticket', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analisis_eda_multiconcentrados.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. MÉTRICAS PARA SEGMENTACIÓN (Preparación RFM)
# ============================================================================

print(f"\n" + "=" * 50)
print("PREPARACIÓN DE MÉTRICAS PARA SEGMENTACIÓN")
print("=" * 50)

# Calcular métricas RFM básicas para cada cliente
fecha_referencia = pd.to_datetime('2025-04-30')  # Última fecha del dataset

# Crear una fecha aproximada para cada transacción (usando año y mes)
mes_numero = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 
              'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 
              'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}

# Crear columna de fecha aproximada para análisis temporal (sin usar pd.to_datetime)
mes_numero = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 
              'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 
              'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12}

df_ventas['mes_num'] = df_ventas['MES'].map(mes_numero)

# Crear fecha de referencia simple para recency
fecha_referencia = datetime(2025, 4, 30)  # Última fecha del dataset

# Métricas RFM por cliente (sin usar fechas complejas)
rfm_data = df_ventas.groupby('CLIENTE').agg({
    'AÑO': 'max',                    # Año de última compra
    'mes_num': 'max',                # Mes de última compra  
    'VALOR TOTAL': ['count', 'sum'], # Frecuencia y Valor Monetario
    'CANTIDAD': 'sum'                # Unidades totales
}).round(2)

rfm_data.columns = ['ultimo_año', 'ultimo_mes', 'frecuencia', 'valor_monetario', 'unidades_totales']

# Calcular recency aproximada en meses desde la última compra
rfm_data['recency_meses'] = (2025 - rfm_data['ultimo_año']) * 12 + (4 - rfm_data['ultimo_mes'])

# Estadísticas de las métricas RFM
print(f"\nEstadísticas de métricas para segmentación:")
print(f"\nRecency (meses desde última compra):")
print(f"  Promedio: {rfm_data['recency_meses'].mean():.1f} meses")
print(f"  Mediana: {rfm_data['recency_meses'].median():.1f} meses")
print(f"  Rango: {rfm_data['recency_meses'].min()} - {rfm_data['recency_meses'].max()} meses")

print(f"\nFrecuencia (número de transacciones):")
print(f"  Promedio: {rfm_data['frecuencia'].mean():.1f} transacciones")
print(f"  Mediana: {rfm_data['frecuencia'].median():.1f} transacciones")
print(f"  Rango: {rfm_data['frecuencia'].min()} - {rfm_data['frecuencia'].max()} transacciones")

print(f"\nValor Monetario (total de compras):")
print(f"  Promedio: ${rfm_data['valor_monetario'].mean():,.0f}")
print(f"  Mediana: ${rfm_data['valor_monetario'].median():,.0f}")
print(f"  Rango: ${rfm_data['valor_monetario'].min():,.0f} - ${rfm_data['valor_monetario'].max():,.0f}")

# Guardar datos preparados para modelado
rfm_data.to_csv('datos_rfm_clientes.csv')
df_ventas.to_csv('datos_ventas_procesados.csv', index=False)

print(f"\nArchivos generados para modelado:")
print(f"- datos_rfm_clientes.csv: Métricas RFM por cliente")
print(f"- datos_ventas_procesados.csv: Dataset de ventas procesado")
print(f"- analisis_eda_multiconcentrados.png: Visualizaciones principales")

# ============================================================================
# 11. RESUMEN EJECUTIVO DE HALLAZGOS
# ============================================================================

print(f"\n" + "=" * 80)
print("RESUMEN EJECUTIVO - HALLAZGOS PRINCIPALES")
print("=" * 80)

print(f"\n📊 DATOS GENERALES:")
print(f"   • Período analizado: Mayo 2024 - Abril 2025 ({len(meses_unicos)} meses)")
print(f"   • Total transacciones: {total_transacciones:,}")
print(f"   • Clientes únicos: {clientes_unicos:,}")
print(f"   • Valor total de ventas: ${total_ventas:,.0f}")

print(f"\n🎯 CONCENTRACIÓN DEL NEGOCIO:")
pareto_20 = freq_clientes_sorted[freq_clientes_sorted['cliente_pct'] <= 20]['pct_acumulado'].iloc[-1]
print(f"   • El 20% de los clientes genera {pareto_20:.1f}% de las ventas (Principio de Pareto)")
print(f"   • {pct_recurrentes:.1f}% de clientes son recurrentes (más de 1 compra)")
print(f"   • Concentración geográfica: {pct_boyaca:.1f}% de ventas en Boyacá")

print(f"\n📈 COMPORTAMIENTO DE COMPRA:")
print(f"   • Ticket promedio: ${ticket_promedio:,.0f}")
print(f"   • Productos únicos: {df_ventas['CODIGO'].nunique():,}")
print(f"   • Categorías principales: {len(cat_ventas)} categorías de productos")
print(f"   • Diversidad promedio: {productos_por_cliente['productos_unicos'].mean():.1f} productos por cliente")

print(f"\n🌍 ALCANCE GEOGRÁFICO:")
print(f"   • Departamentos atendidos: {df_ventas['DEPARTAMENTO'].nunique()}")
print(f"   • Municipios atendidos: {df_ventas['MUNICIPIOS'].nunique()}")
print(f"   • Principal mercado: Boyacá ({pct_boyaca:.1f}% de ventas)")

print(f"\n💰 OPORTUNIDADES IDENTIFICADAS:")
print(f"   • {(freq_clientes['transacciones'] == 1).sum()} clientes compraron solo una vez")
print(f"   • Potencial de fidelización alto en clientes recurrentes")
print(f"   • Oportunidad de cross-selling: promedio de {productos_por_cliente['productos_unicos'].mean():.1f} productos por cliente")

estacionalidad_max = estacionalidad.idxmax()
estacionalidad_min = estacionalidad.idxmin()
print(f"   • Estacionalidad: Mayor venta en {estacionalidad_max}, menor en {estacionalidad_min}")

print(f"\n🔍 RECOMENDACIONES PARA SEGMENTACIÓN:")
print(f"   • Segmentar por valor (RFM): Identificar clientes de alto valor")
print(f"   • Segmentar por comportamiento: Clientes únicos vs recurrentes")
print(f"   • Segmentar geográficamente: Estrategias diferenciadas por región")
print(f"   • Segmentar por producto: Especialización en categorías específicas")

print(f"\n" + "=" * 80)
print("ANÁLISIS EXPLORATORIO COMPLETADO")
print("Los datos están listos para la fase de segmentación y modelado")
print("=" * 80)