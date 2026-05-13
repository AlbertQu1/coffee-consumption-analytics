# -*- coding: utf-8 -*-
"""
Dashboard personal de cafetera:
- Limpia y valida el CSV
- Calcula costos, ahorros, inventario, recuperación
- Estima ETA de agotamiento según tu ritmo real
- Compara vs precios de cafeterías externas
- (Opcional) genera gráficas de consumo
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===================== PARÁMETROS =====================
ruta_csv = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n/pub?gid=0&single=true&output=csv"
costo_cafetera = 12239  # MXN

# Precios externos (ajusta libremente)
precio_cafeteria = {
    'Elena': 70,
    'Chavelete': 60,
    'Coffee King': 50,
    'Starbucks': 58,
    'Quentin 1': 55,
    'Quentin 2': 70,
    'Otro': 40,
    'Garat': 46,
    'The Coffe': 55,
    'Cielito': 55
}

# Columnas de tamaños (mantener estas etiquetas en el CSV)
columnas_tazas = ['espresso', 'quad', '6oz', '8oz', '10oz', '12oz', '14oz', '16oz', '18oz']

# Gramaje por tamaño (alineado a columnas_tazas)
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

# Alertas                                        
ALERTA_INVENTARIO_KG = 0.1    # alerta por bajo inventario (100 gramos)
ALERTA_INVENTARIO_DIAS = 3    # alerta por días restantes

# Gráficas
GENERAR_GRAFICAS = False     # pon True si quieres ver plots


# ===================== HELPERS =====================
def fmt_mon(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'N/D'
    return f"${v:,.2f} MXN"

def fmt_num(v, dec=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'N/D'
    return f"{v:,.{dec}f}"

def coalesce_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ===================== CARGA =====================
df = pd.read_csv(ruta_csv)
df.columns = [c.strip() for c in df.columns]  # normaliza encabezados

# ===================== FECHAS =====================
# Busca una columna que empiece por "fecha"
col_fecha = None
for c in df.columns:
    if c.lower().strip().startswith('fecha'):
        col_fecha = c
        break

if col_fecha:
    # Tus datos están en formato dd/mm/yy (ej: 07/06/25 => 7-jun-2025)
    df[col_fecha] = pd.to_datetime(
        df[col_fecha].astype(str).str.strip(),
        format='%d/%m/%y',  # clave
        errors='coerce'
    )
    fechas_validas = df[col_fecha].dropna()
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

# ===================== COLUMNAS TAZAS =====================
for col in columnas_tazas:
    if col not in df.columns:
        df[col] = 0

# Cast seguro a numérico
df['gramos'] = pd.to_numeric(df.get('gramos', 0), errors='coerce').fillna(0)
df['costo']  = pd.to_numeric(df.get('costo', 0),  errors='coerce').fillna(0)
df[columnas_tazas] = df[columnas_tazas].apply(pd.to_numeric, errors='coerce').fillna(0)

# ===================== VALIDACIÓN "# de tazas" =====================
# Busca columna declarada de total de tazas (si existe)
col_total_tazas_decl = None
for c in df.columns:
    if 'taza' in c.lower():  # captura "# de tazas", "tazas", etc.
        col_total_tazas_decl = c
        break

if col_total_tazas_decl:
    # Sobrescribe la columna con la suma real por fila
    df[col_total_tazas_decl] = df[columnas_tazas].sum(axis=1)
    print(f"✅ La columna '{col_total_tazas_decl}' ha sido actualizada con la suma por tamaños en cada fila.")

# ===================== CÁLCULOS BASE =====================
precio_promedio_cafeteria = sum(precio_cafeteria.values()) / len(precio_cafeteria)

gramos_comprados = float(df['gramos'].sum())
cafe_comprado_kg = gramos_comprados / 1000.0

total_tazas_tipo = df[columnas_tazas].sum(numeric_only=True)   # Serie por tipo
total_tazas = int(total_tazas_tipo.sum())

costo_total_cafe = float(df['costo'].sum())

# Consumo estimado (gramos) según mezcla real
gramos_consumidos = 0.0
for tipo, cantidad in total_tazas_tipo.items():
    gramos_consumidos += float(cantidad) * float(gramos_por_taza.get(tipo, 0))

# Costo por gramo
costo_por_gramo = (costo_total_cafe / gramos_comprados) if gramos_comprados > 0 else np.inf

# Costo por taza por tipo y costo promedio ponderado por tu mezcla real
costo_taza_tipo = {
    t: (gramos_por_taza[t] * costo_por_gramo) if np.isfinite(costo_por_gramo) else np.inf
    for t in columnas_tazas
}

if total_tazas > 0 and np.isfinite(costo_por_gramo):
    costo_promedio_taza = sum(costo_taza_tipo[t] * total_tazas_tipo[t] for t in columnas_tazas) / total_tazas
else:
    costo_promedio_taza = np.inf

# Ahorro y recuperación vs promedio de cafeterías
ahorro_por_taza = precio_promedio_cafeteria - costo_promedio_taza if np.isfinite(costo_promedio_taza) else -np.inf
tazas_para_recuperar = (costo_cafetera / ahorro_por_taza) if (ahorro_por_taza > 0) else np.inf
ahorro_total = (max(0, ahorro_por_taza) * total_tazas) if (total_tazas > 0 and np.isfinite(ahorro_por_taza)) else 0.0

# Inventario
inventario_estimado_g = max(0.0, gramos_comprados - gramos_consumidos)
inventario_estimado_kg = inventario_estimado_g / 1000.0

# Ritmo real y ETA de agotamiento
if dias_consumo and (dias_consumo > 0) and (gramos_consumidos > 0):
    tazas_por_dia_real = total_tazas / dias_consumo
    ahorro_diario_real = ahorro_por_taza * tazas_por_dia_real if ahorro_por_taza > 0 else 0
    ritmo_g_dia = gramos_consumidos / dias_consumo
    if ritmo_g_dia > 0:
        dias_restantes = inventario_estimado_g / ritmo_g_dia
        fecha_agotamiento = fecha_max + pd.Timedelta(days=dias_restantes)
    else:
        dias_restantes = np.inf
        fecha_agotamiento = None
    if ahorro_por_taza > 0 and tazas_por_dia_real > 0:
        dias_para_recuperar_real = tazas_para_recuperar / tazas_por_dia_real
    else:
        dias_para_recuperar_real = np.inf
else:
    tazas_por_dia_real = None
    ahorro_diario_real = None
    ritmo_g_dia = None
    dias_restantes = None
    fecha_agotamiento = None
    dias_para_recuperar_real = None

# ===================== COMPARATIVA POR CAFETERÍA =====================
comparativa = []
for nombre, precio_ext in precio_cafeteria.items():
    ahorro_taza_i = (precio_ext - costo_promedio_taza) if np.isfinite(costo_promedio_taza) else -np.inf
    tazas_rec_i = (costo_cafetera / ahorro_taza_i) if (ahorro_taza_i > 0) else np.inf
    comparativa.append({
        'Cafetería': nombre,
        'Precio externo (MXN)': precio_ext,
        'Costo promedio en casa (MXN)': None if not np.isfinite(costo_promedio_taza) else round(costo_promedio_taza, 2),
        'Ahorro por taza (MXN)': None if not np.isfinite(ahorro_taza_i) else round(ahorro_taza_i, 2),
        'Tazas para recuperar': None if not np.isfinite(tazas_rec_i) else int(round(tazas_rec_i, 0))
    })
comparativa_df = pd.DataFrame(comparativa).sort_values('Precio externo (MXN)', ascending=False).reset_index(drop=True)

# ===================== REPORTE =====================
print('\n☕ RESUMEN PERSONAL DE CONSUMO Y AHORRO (con gramaje por tamaño)')
print(f'• Precio promedio en cafeterías externas: {fmt_mon(precio_promedio_cafeteria)}')
if np.isfinite(costo_promedio_taza):
    print(f'• Costo por gramo estimado en casa:       {fmt_mon(costo_por_gramo)} /g')
    print(f'• Costo promedio por taza en casa:        {fmt_mon(costo_promedio_taza)}')
else:
    print('• Costo por gramo/taza en casa:           N/D (faltan datos de gramos o tazas)')

print(f'• Café comprado:                           {fmt_num(cafe_comprado_kg)} kg')
print(f'• Café consumido (estimado):               {fmt_num(gramos_consumidos/1000)} kg')
print(f'• Inventario estimado:                     {fmt_num(inventario_estimado_kg)} kg')
print(f'• Tazas servidas en casa:                  {total_tazas:,}')

if (ahorro_por_taza > 0) and (total_tazas > 0) and np.isfinite(costo_promedio_taza):
    print(f'• Ahorro por taza (vs. promedio):          {fmt_mon(ahorro_por_taza)}')
    print(f'• Tazas para recuperar la cafetera:        {fmt_num(tazas_para_recuperar, 0)}')
    print(f'• Ahorro total acumulado:                  {fmt_mon(ahorro_total)}')
elif total_tazas == 0:
    print('• Ahorro por taza:                         N/D (no hay tazas registradas)')
    print('• Tazas para recuperar la cafetera:        N/D')
    print('• Ahorro total acumulado:                  $0.00 MXN')
else:
    print('• Ahorro por taza:                         No aplicable (costo en casa ≥ precio externo o faltan datos)')
    print('• Tazas para recuperar la cafetera:        No aplicable')
    print(f'• Ahorro total acumulado:                  {fmt_mon(ahorro_total)}')

if fecha_min and fecha_max:
    print(f'• Primer registro: {fecha_min.date()}  • Último registro: {fecha_max.date()}  • Días: {dias_consumo:,}')
    if tazas_por_dia_real is not None:
        print(f'• Tazas por día (real): {fmt_num(tazas_por_dia_real)}  • Ahorro diario (real): {fmt_mon(ahorro_diario_real)}')
else:
    print('• No hay datos de fecha para proyecciones reales.')

# Alertas
if inventario_estimado_kg <= ALERTA_INVENTARIO_KG:
    print(f'⚠️  Alerta: inventario bajo (≤ {ALERTA_INVENTARIO_KG} kg). Considera comprar café pronto.')
if (dias_restantes is not None) and np.isfinite(dias_restantes) and (dias_restantes <= ALERTA_INVENTARIO_DIAS):
    print(f'⚠️  Alerta: quedan ~{dias_restantes:.1f} días de inventario. ETA agotamiento: {fecha_agotamiento.date()}.')

# Desglose de costo por tipo
print('\n💸 COSTO POR TAZA POR TIPO (MXN)')
for tipo in columnas_tazas:
    c = costo_taza_tipo[tipo]
    print(f'• {tipo:<6}: {fmt_mon(c)}')

# Promedio del costo por proveedor/tienda si existe la columna
col_cafeteria_csv = coalesce_col(df, ['Cafeteria', 'cafeteria', 'Proveedor', 'Tienda'])
if col_cafeteria_csv:
    print('\n📊 PROMEDIO DEL COSTO')
    total_promedio = 0.0
    count_promedio = 0
    for nombre, sub in df.groupby(col_cafeteria_csv, dropna=False):
        promedio = sub['costo'].mean()
        etiqueta = 'N/D' if pd.isna(nombre) else str(nombre)
        print(f'• {etiqueta}: {fmt_mon(promedio)}')
        if pd.notna(promedio) and promedio > 0:
            total_promedio += promedio
            count_promedio += 1
    if count_promedio > 0:
        print(f'• Total promedio del costo: {fmt_mon(total_promedio)}')
    else:
        print('• Total promedio del costo: N/D')

# Detalle de tazas por tipo (acumulado)
print('\n☕ DETALLE DE TAZAS POR TIPO')
for tipo, cantidad in total_tazas_tipo.items():
    print(f'• {tipo:<6}: {int(cantidad):,}')

# Comparativa externa
print('\n🏷️  COMPARATIVA POR CAFETERÍA (vs cada precio externo)')
print(comparativa_df.to_string(index=False))

# ===================== RECUPERACIÓN (HIPOTÉTICA Y REAL) =====================
print('\n⏳ TIEMPO ESTIMADO PARA RECUPERAR EL COSTO (consumo hipotético)')
if (ahorro_por_taza > 0) and (total_tazas > 0) and np.isfinite(costo_promedio_taza):
    for tpd in [1, 2, 3, 4]:
        dias = tazas_para_recuperar / tpd
        print(f'• {tpd} taza(s) al día: {int(round(dias,0))} días (~{dias/30:,.1f} meses)')
    if (dias_para_recuperar_real is not None) and np.isfinite(dias_para_recuperar_real):
        print(f'• Según tu ritmo real ({tazas_por_dia_real:,.2f} tazas/día): {int(round(dias_para_recuperar_real,0))} días (~{dias_para_recuperar_real/30:,.1f} meses)')
    else:
        print('• No se puede estimar tiempo real de recuperación (faltan fechas o ahorro ≤ 0).')
else:
    print('• No hay recuperación si no hay tazas o si el ahorro por taza es ≤ 0 o faltan gramos.')

# ===================== GRÁFICAS OPCIONALES =====================
if GENERAR_GRAFICAS and col_fecha and not df[col_fecha].dropna().empty:
    import matplotlib.pyplot as plt

    # Serie: tazas por día
    df_g = df.copy()
    df_g[col_fecha] = pd.to_datetime(df_g[col_fecha], errors='coerce')
    tmp = df_g.dropna(subset=[col_fecha]).copy()
    tmp['tazas_totales'] = tmp[columnas_tazas].sum(axis=1)
    ts = tmp.groupby(tmp[col_fecha].dt.date)['tazas_totales'].sum()

    plt.figure()
    ts.plot()
    plt.title('Tazas por día')
    plt.xlabel('Fecha')
    plt.ylabel('Número de tazas')
    plt.tight_layout()
    plt.show()

    # Barras: mezcla de tamaños
    plt.figure()
    total_tazas_tipo.sort_values(ascending=False).plot(kind='bar')
    plt.title('Mezcla de tamaños (acumulado)')
    plt.xlabel('Tipo de taza')
    plt.ylabel('Tazas')
    plt.tight_layout()
    plt.show()
    plt.show()
