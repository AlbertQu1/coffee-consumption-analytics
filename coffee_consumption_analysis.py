import pandas as pd

# === Parámetros personales ===
costo_cafetera = 12239  # MXN
precio_cafeteria = {
    'Elena': 70,
    'Chavelete': 60,
    'Coffee King': 50,
    'Starbucks': 58,
    'Quentin': 55,
    'Quentin 2': 70,
    'otro': 40
}
columnas_tazas = ['espresso', 'quad', '6oz', '8oz', '10oz', '12oz', '14oz', '16oz', '18oz']

# Gramos por taza (alineado a columnas_tazas en minúsculas)
gramos_por_taza = {
    'espresso': 18,
    'quad': 40,
    '6oz': 10,
    '8oz': 14,
    '10oz': 18,
    '12oz': 21,
    '14oz': 25,
    '16oz': 28,
    '18oz': 31
}

# URL del Google Sheet principal
url_principal = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n/pub?gid=0&single=true&output=csv"

# Leer el dataframe principal
df = pd.read_csv(url_principal)

# Procesar columna de fecha si existe
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    fechas_validas = df['fecha'].dropna()
    if not fechas_validas.empty:
        fecha_min = fechas_validas.min()
        fecha_max = fechas_validas.max()
        dias_consumo = max(1, (fecha_max - fecha_min).days + 1)
    else:
        fecha_min = fecha_max = None
        dias_consumo = None
else:
    fecha_min = fecha_max = None
    dias_consumo = None

# Asegurar columnas de tazas (si faltan, se crean en 0)
for col in columnas_tazas:
    if col not in df.columns:
        df[col] = 0

# Cast seguro
df['gramos'] = pd.to_numeric(df.get('gramos', 0), errors='coerce').fillna(0)
df['costo']  = pd.to_numeric(df.get('costo', 0),  errors='coerce').fillna(0)
df[columnas_tazas] = df[columnas_tazas].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Cálculos base ===
precio_promedio_cafeteria = sum(precio_cafeteria.values()) / len(precio_cafeteria)

gramos_comprados = float(df['gramos'].sum())          # gramos comprados (incluye regalos = 0)
cafe_comprado_kg = gramos_comprados / 1000

total_tazas_tipo = df[columnas_tazas].sum(numeric_only=True)
total_tazas = int(total_tazas_tipo.sum())

costo_total_cafe = float(df['costo'].sum())

# Consumo estimado de gramos según mezcla de tamaños
gramos_consumidos = 0.0
for tipo, cantidad in total_tazas_tipo.items():
    gramos_consumidos += float(cantidad) * float(gramos_por_taza.get(tipo, 0))

# Costo por gramo (si hay datos)
costo_por_gramo = (costo_total_cafe / gramos_comprados) if gramos_comprados > 0 else float('inf')

# Costo por taza por tipo (en MXN) y costo promedio ponderado por tu mezcla real
costo_taza_tipo = {t: (gramos_por_taza[t] * costo_por_gramo) if costo_por_gramo != float('inf') else float('inf')
                   for t in columnas_tazas}

if total_tazas > 0 and costo_por_gramo != float('inf'):
    costo_promedio_taza = sum(costo_taza_tipo[t] * total_tazas_tipo[t] for t in columnas_tazas) / total_tazas
else:
    costo_promedio_taza = float('inf')

# Métricas financieras personales (con salvaguardas)
ahorro_por_taza = precio_promedio_cafeteria - costo_promedio_taza if costo_promedio_taza != float('inf') else float('-inf')
tazas_para_recuperar = (costo_cafetera / ahorro_por_taza) if (ahorro_por_taza > 0) else float('inf')
ahorro_total = (max(0, ahorro_por_taza) * total_tazas) if total_tazas > 0 and ahorro_por_taza != float('-inf') else 0.0

# Inventario estimado (en gramos y kg)
inventario_estimado_g = max(0.0, gramos_comprados - gramos_consumidos)
inventario_estimado_kg = inventario_estimado_g / 1000

# Proyecciones usando fechas reales
if dias_consumo and total_tazas > 0:
    tazas_por_dia_real = total_tazas / dias_consumo
    ahorro_diario_real = ahorro_por_taza * tazas_por_dia_real if ahorro_por_taza > 0 else 0
    if ahorro_por_taza > 0:
        dias_para_recuperar_real = tazas_para_recuperar / tazas_por_dia_real
    else:
        dias_para_recuperar_real = float('inf')
else:
    tazas_por_dia_real = None
    ahorro_diario_real = None
    dias_para_recuperar_real = None

# === Reporte personal ===
print('\nRESUMEN PERSONAL DE CONSUMO Y AHORRO (con gramaje por tamaño)')
print(f'- Precio promedio en cafeterías externas: ${precio_promedio_cafeteria:,.2f} MXN')

if costo_promedio_taza != float('inf'):
    print(f'- Costo por gramo estimado en casa:       ${costo_por_gramo:,.4f} MXN/g')
    print(f'- Costo promedio por taza en casa:        ${costo_promedio_taza:,.2f} MXN')
else:
    print('- Costo por gramo/taza en casa:           N/D (faltan datos de gramos o tazas)')

print(f'- Café comprado:                           {cafe_comprado_kg:,.2f} kg')
print(f'- Café consumido (estimado):               {gramos_consumidos/1000:,.2f} kg')
print(f'- Inventario estimado:                     {inventario_estimado_kg:,.2f} kg')
print(f'- Tazas servidas en casa:                  {total_tazas:,}')

if (ahorro_por_taza > 0) and (total_tazas > 0) and (costo_promedio_taza != float('inf')):
    print(f'- Ahorro por taza (vs. comprar fuera):    ${ahorro_por_taza:,.2f} MXN')
    print(f'- Tazas para recuperar la cafetera:       {tazas_para_recuperar:,.0f}')
    print(f'- Ahorro total acumulado:                 ${ahorro_total:,.2f} MXN')
elif total_tazas == 0:
    print('- Ahorro por taza:                         N/D (no hay tazas registradas)')
    print('- Tazas para recuperar la cafetera:        N/D')
    print('- Ahorro total acumulado:                  $0.00 MXN')
else:
    print('- Ahorro por taza:                         No aplicable (tu costo en casa ≥ precio externo o faltan datos)')
    print('- Tazas para recuperar la cafetera:        No aplicable')
    print(f'- Ahorro total acumulado:                  ${ahorro_total:,.2f} MXN')

if fecha_min and fecha_max:
    print(f'- Primer registro: {fecha_min.date()}')
    print(f'- Último registro: {fecha_max.date()}')
    print(f'- Días totales de consumo: {dias_consumo:,}')
    if tazas_por_dia_real:
        print(f'- Tazas por día (real): {tazas_por_dia_real:,.2f}')
        print(f'- Ahorro diario (real): ${ahorro_diario_real:,.2f} MXN')
else:
    print('- No hay datos de fecha para proyecciones reales.')

# Desglose de costo por tipo
print('\nCOSTO POR TAZA POR TIPO (MXN)')
for tipo in columnas_tazas:
    c = costo_taza_tipo[tipo]
    print(f'- {tipo}: ${c:,.2f}' if c != float('inf') else f'- {tipo}: N/D')

# Promedio del costo declarado en CSV (si existe la columna categórica)
if 'Cafeteria' in df.columns:
    print('\nPROMEDIO DEL COSTO')
    for nombre, sub in df.groupby('Cafeteria', dropna=False):
        promedio = sub['costo'].mean()
        etiqueta = 'N/D' if pd.isna(nombre) else str(nombre)
        print(f'- {etiqueta}: ${promedio:,.2f} MXN')

# Detalle de tazas por tipo
print('\nDETALLE DE TAZAS POR TIPO')
for tipo, cantidad in total_tazas_tipo.items():
    print(f'- {tipo}: {int(cantidad):,}')

# === TIEMPO ESTIMADO PARA RECUPERAR EL COSTO (consumo hipotético) ===
print('\nTIEMPO ESTIMADO PARA RECUPERAR EL COSTO (consumo hipotético)')
if (ahorro_por_taza > 0) and (total_tazas > 0) and (costo_promedio_taza != float('inf')):
    for tpd in [1, 2, 3, 4]:
        dias = tazas_para_recuperar / tpd
        print(f'- {tpd} taza(s) al día: {dias:,.0f} días (~{dias/30:,.1f} meses)')
    if dias_para_recuperar_real and dias_para_recuperar_real != float('inf'):
        print(f'- Según tu ritmo real ({tazas_por_dia_real:,.2f} tazas/día): {dias_para_recuperar_real:,.0f} días (~{dias_para_recuperar_real/30:,.1f} meses)')
    else:
        print('- No se puede estimar tiempo real de recuperación (faltan fechas o ahorro negativo).')
else:
    print('- No hay recuperación si no hay tazas o si el ahorro por taza es ≤ 0 o faltan gramos.')

