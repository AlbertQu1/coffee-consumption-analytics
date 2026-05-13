import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import timedelta

# ===================== CONFIG =====================
@dataclass(frozen=True)
class Config:
    ruta_csv: str
    costo_cafetera: float
    precio_cafeteria: Dict[str, float]
    columnas_tazas: List[str]
    gramos_por_taza: Dict[str, float]
    alerta_inventario_kg: float = 0.1
    alerta_inventario_dias: float = 2
    generar_graficas: bool = False
    ventana_movil_dias: int = 30  # ETA basada en consumo reciente

# ======= UTILIDADES =======
def fmt_mon(v):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return 'N/D'
        return f"${v:,.2f} MXN"
    except Exception:
        return 'N/D'

def fmt_num(v, dec=2):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return 'N/D'
        return f"{v:,.{dec}f}"
    except Exception:
        return 'N/D'

# ======= CARGA & NORMALIZACIÓN =======
def load_data(cfg: Config) -> pd.DataFrame:
    try:
        df = pd.read_csv(cfg.ruta_csv, skip_blank_lines=True)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()

def detect_and_parse_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
    col_fecha = next((c for c in df.columns if c.lower().strip().startswith('fecha')), None)

    fecha_min = fecha_max = None
    dias_consumo = None

    if col_fecha:
        s = df[col_fecha].astype(str).str.strip()
        parsed = pd.to_datetime(s, dayfirst=True, errors='coerce')
        if parsed.isna().all():
            parsed = pd.to_datetime(s, format='%d/%m/%y', errors='coerce')
        df[col_fecha] = parsed

        fechas_validas = parsed.dropna()
        if not fechas_validas.empty:
            fecha_min = fechas_validas.min().normalize()
            fecha_max = fechas_validas.max().normalize()
            dias_consumo = max(1, (fecha_max - fecha_min).days + 1)

    return df, col_fecha, fecha_min, fecha_max, dias_consumo

def ensure_taza_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # columnas de tazas que falten -> 0
    for col in cfg.columnas_tazas:
        if col not in df.columns:
            df[col] = 0

    # casteos numéricos en bloque
    for col in ['gramos', 'costo']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df[cfg.columnas_tazas] = df[cfg.columnas_tazas].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# ======= VALIDACIONES =======
def validate_schema(df: pd.DataFrame, cfg: Config) -> List[str]:
    issues = []
    faltantes_en_gramas = [t for t in cfg.columnas_tazas if t not in cfg.gramos_por_taza]
    if faltantes_en_gramas:
        issues.append(f"Faltan gramajes para: {faltantes_en_gramas}.")
    sobrantes_en_gramas = [t for t in cfg.gramos_por_taza if t not in cfg.columnas_tazas]
    if sobrantes_en_gramas:
        issues.append(f"Hay gramajes definidos para tamaños no presentes: {sobrantes_en_gramas}.")
    if df[cfg.columnas_tazas].lt(0).any().any():
        issues.append("Existen conteos de tazas negativos.")
    if df['gramos'].lt(0).any():
        issues.append("Existen compras con gramos negativos.")
    if df['costo'].lt(0).any():
        issues.append("Existen compras con costo negativo.")
    return issues

def sync_total_tazas(df: pd.DataFrame, cfg: Config) -> Optional[str]:
    # si existe una columna que contenga 'taza', sincronízala con la suma
    candidates = [c for c in df.columns if 'taza' in c.lower()]
    col_total = candidates[0] if candidates else None
    if col_total:
        df[col_total] = df[cfg.columnas_tazas].sum(axis=1)
    return col_total

# ======= SERIE DIARIA =======
def build_daily_series(df: pd.DataFrame, cfg: Config, col_fecha: Optional[str]) -> Optional[pd.DataFrame]:
    if not col_fecha or not df[col_fecha].notna().any():
        return None

    gramos_vec = np.array([cfg.gramos_por_taza.get(t, 0.0) for t in cfg.columnas_tazas], dtype=float)
    tazas_mat = df[cfg.columnas_tazas].to_numpy(dtype=float)

    df = df.copy()
    df['gramos_estimados'] = (tazas_mat * gramos_vec).sum(axis=1)
    df['tazas_total'] = df[cfg.columnas_tazas].sum(axis=1)

    g = (df.dropna(subset=[col_fecha])
           .groupby(df[col_fecha].dt.normalize())
           .agg(g_dia=('gramos_estimados', 'sum'),
                tazas_dia=('tazas_total', 'sum'))
         )
    g.index.name = 'fecha'
    return g

# ======= DURACIÓN DE CADA BOLSA =======
def compute_bag_lifetimes(df: pd.DataFrame, diarios: Optional[pd.DataFrame], col_fecha: Optional[str]) -> Optional[pd.DataFrame]:
    if (col_fecha is None) or (diarios is None) or diarios.empty:
        return None

    restocks = df.loc[df['gramos'] > 0, [col_fecha, 'gramos', 'costo']].dropna(subset=[col_fecha]).copy()
    if restocks.empty:
        return None

    restocks[col_fecha] = pd.to_datetime(restocks[col_fecha]).dt.normalize()
    restocks = restocks.sort_values(col_fecha).reset_index(drop=True)

    records = []
    last_day = diarios.index.max()

    for i, row in restocks.iterrows():
        start = row[col_fecha]
        grams = float(row['gramos'])
        cost = float(row['costo'])

        if i + 1 < len(restocks):
            end_limit = (restocks.loc[i + 1, col_fecha] - timedelta(days=1)).normalize()
        else:
            end_limit = last_day

        if end_limit < start:
            end_limit = start

        # rango robusto por condición booleana
        seg = diarios.loc[(diarios.index >= start) & (diarios.index <= end_limit)]
        if seg.empty:
            records.append({
                'inicio': start.date(),
                'fin_por_agotamiento': None,
                'fin_por_cambio': end_limit.date(),
                'dias_hasta_agotamiento': None,
                'dias_entre_compras': (end_limit - start).days + 1,
                'gramos_bolsa': grams,
                'gramos_consumidos': 0.0,
                'gramos_sobrantes': grams,
                'costo_bolsa': cost,
                'cambio_antes_de_agotar': True
            })
            continue

        cumsum = seg['g_dia'].cumsum()
        agot_mask = cumsum >= grams
        if agot_mask.any():
            fin_agot = cumsum.index[agot_mask.argmax()]
            grams_consumed = grams
            sobrante = 0.0
            cambio_antes = False
            dias_agot = (fin_agot - start).days + 1
        else:
            fin_agot = None
            grams_consumed = float(cumsum.iloc[-1])
            sobrante = max(0.0, grams - grams_consumed)
            cambio_antes = True
            dias_agot = None

        records.append({
            'inicio': start.date(),
            'fin_por_agotamiento': fin_agot.date() if fin_agot is not None else None,
            'fin_por_cambio': end_limit.date(),
            'dias_hasta_agotamiento': dias_agot,
            'dias_entre_compras': (end_limit - start).days + 1,
            'gramos_bolsa': grams,
            'gramos_consumidos': grams_consumed,
            'gramos_sobrantes': sobrante,
            'costo_bolsa': cost,
            'cambio_antes_de_agotar': cambio_antes
        })

    return pd.DataFrame.from_records(records)

# ======= ROI TIMELINE =======
def compute_roi_timeline(diarios: Optional[pd.DataFrame], ahorro_por_taza: float, costo_cafetera: float) -> Dict[str, Optional[object]]:
    if diarios is None or diarios.empty or not np.isfinite(ahorro_por_taza) or ahorro_por_taza <= 0:
        return {'fecha_roi': None, 'dias_hasta_roi': None, 'ahorro_total_acum': 0.0, 'ahorro_post_roi': 0.0}

    d = diarios.copy()
    d['ahorro_dia'] = d['tazas_dia'] * ahorro_por_taza
    d['ahorro_acum'] = d['ahorro_dia'].cumsum()

    reached = d['ahorro_acum'] >= costo_cafetera
    if reached.any():
        fecha_roi = reached.idxmax()
        dias_hasta_roi = (fecha_roi - d.index.min()).days + 1
        ahorro_total_acum = float(d['ahorro_acum'].iloc[-1])
        return {
            'fecha_roi': fecha_roi,
            'dias_hasta_roi': dias_hasta_roi,
            'ahorro_total_acum': ahorro_total_acum,
            'ahorro_post_roi': max(0.0, ahorro_total_acum - costo_cafetera)
        }
    else:
        ahorro_total_acum = float(d['ahorro_acum'].iloc[-1])
        return {'fecha_roi': None, 'dias_hasta_roi': None, 'ahorro_total_acum': ahorro_total_acum, 'ahorro_post_roi': 0.0}

# ======= CÁLCULOS GLOBALES =======
def compute_metrics(df: pd.DataFrame, cfg: Config, col_fecha: Optional[str], fecha_min, fecha_max, dias_consumo):
    precio_promedio_cafeteria = float(np.mean(list(cfg.precio_cafeteria.values()))) if cfg.precio_cafeteria else np.nan
    gramos_comprados = float(df['gramos'].sum())
    cafe_comprado_kg = gramos_comprados / 1000.0
    costo_total_cafe = float(df['costo'].sum())

    total_tazas_tipo = df[cfg.columnas_tazas].sum(numeric_only=True)
    total_tazas = int(total_tazas_tipo.sum())

    diarios = build_daily_series(df, cfg, col_fecha)
    gramos_consumidos = float(diarios['g_dia'].sum()) if (diarios is not None and not diarios.empty) else 0.0

    costo_por_gramo = (costo_total_cafe / gramos_comprados) if gramos_comprados > 0 else np.inf
    costo_taza_tipo = {
        t: (cfg.gramos_por_taza.get(t, 0.0) * costo_por_gramo) if np.isfinite(costo_por_gramo) else np.inf
        for t in cfg.columnas_tazas
    }
    if total_tazas > 0 and np.isfinite(costo_por_gramo):
        pesos = np.array([costo_taza_tipo[t] for t in cfg.columnas_tazas], dtype=float)
        cantidades = total_tazas_tipo.values.astype(float)
        costo_promedio_taza = float(np.dot(pesos, cantidades) / total_tazas)
    else:
        costo_promedio_taza = np.inf

    ahorro_por_taza = (precio_promedio_cafeteria - costo_promedio_taza) if np.isfinite(costo_promedio_taza) else -np.inf
    tazas_para_recuperar = (cfg.costo_cafetera / ahorro_por_taza) if (ahorro_por_taza > 0) else np.inf
    ahorro_total = (max(0, ahorro_por_taza) * total_tazas) if (total_tazas > 0 and np.isfinite(ahorro_por_taza)) else 0.0

    inventario_estimado_g = max(0.0, gramos_comprados - gramos_consumidos)
    inventario_estimado_kg = inventario_estimado_g / 1000.0

    ritmo_g_dia = None
    tazas_por_dia_real = None
    ahorro_diario_real = None
    dias_restantes = None
    fecha_agotamiento = None
    dias_para_recuperar_real = None

    if diarios is not None and not diarios.empty and (fecha_min is not None) and (fecha_max is not None):
        dias_consumo_calc = max(1, (fecha_max - fecha_min).days + 1)
        ritmo_g_dia = diarios['g_dia'].sum() / dias_consumo_calc
        tazas_por_dia_real = diarios['tazas_dia'].sum() / dias_consumo_calc

        tail = diarios.tail(cfg.ventana_movil_dias)
        if not tail.empty:
            ritmo_recent = tail['g_dia'].mean()
            tazas_recent = tail['tazas_dia'].mean()
            if np.isfinite(ritmo_recent) and ritmo_recent > 0:
                ritmo_g_dia = ritmo_recent
            if np.isfinite(tazas_recent) and tazas_recent > 0:
                tazas_por_dia_real = tazas_recent

        ahorro_diario_real = (ahorro_por_taza * tazas_por_dia_real) if (ahorro_por_taza > 0 and tazas_por_dia_real) else 0
        if ritmo_g_dia and ritmo_g_dia > 0:
            dias_restantes = inventario_estimado_g / ritmo_g_dia
            fecha_agotamiento = pd.to_datetime(fecha_max) + pd.Timedelta(days=float(dias_restantes))

        if (ahorro_por_taza > 0) and tazas_por_dia_real and (tazas_por_dia_real > 0):
            dias_para_recuperar_real = tazas_para_recuperar / tazas_por_dia_real

    bolsas_df = compute_bag_lifetimes(df, diarios, col_fecha)
    roi = compute_roi_timeline(diarios, ahorro_por_taza, cfg.costo_cafetera)

    costo_promedio_por_compra = df.loc[df['costo'] > 0, 'costo'].mean() if (df['costo'] > 0).any() else np.nan
    costo_promedio_por_kg = (costo_total_cafe / (gramos_comprados/1000.0)) if gramos_comprados > 0 else np.nan

    return {
        'precio_promedio_cafeteria': precio_promedio_cafeteria,
        'costo_por_gramo': costo_por_gramo,
        'costo_promedio_taza': costo_promedio_taza,
        'ahorro_por_taza': ahorro_por_taza,
        'tazas_para_recuperar': tazas_para_recuperar,
        'ahorro_total': ahorro_total,
        'cafe_comprado_kg': cafe_comprado_kg,
        'gramos_consumidos': gramos_consumidos,
        'inventario_estimado_kg': inventario_estimado_kg,
        'total_tazas': total_tazas,
        'total_tazas_tipo': total_tazas_tipo,
        'tazas_por_dia_real': tazas_por_dia_real,
        'ahorro_diario_real': ahorro_diario_real,
        'ritmo_g_dia': ritmo_g_dia,
        'dias_restantes': dias_restantes,
        'fecha_agotamiento': fecha_agotamiento,
        'diarios': diarios,
        'bolsas_df': bolsas_df,
        'roi': roi,
        'costo_promedio_por_compra': costo_promedio_por_compra,
        'costo_promedio_por_kg': costo_promedio_por_kg,
        'fecha_min': fecha_min,
        'fecha_max': fecha_max,
        'dias_consumo': dias_consumo
    }

# ======= REPORTE =======
def print_report(metrics, cfg: Config):
    fecha_min = metrics['fecha_min']; fecha_max = metrics['fecha_max']; dias_consumo = metrics['dias_consumo']

    print('\n☕ RESUMEN DE CONSUMO Y AHORRO')
    print(f'• Precio promedio externas: {fmt_mon(metrics["precio_promedio_cafeteria"])}')

    if np.isfinite(metrics['costo_promedio_taza']):
        print(f'• Costo por gramo:          {fmt_mon(metrics["costo_por_gramo"])} /g')
        print(f'• Costo promedio por taza:  {fmt_mon(metrics["costo_promedio_taza"])}')
    else:
        print('• Costo por gramo/taza:     N/D (faltan datos de gramos o tazas)')

    print(f'• Café comprado:            {fmt_num(metrics["cafe_comprado_kg"])} kg')
    print(f'• Café consumido estimado:  {fmt_num(metrics["gramos_consumidos"]/1000)} kg')
    print(f'• Inventario estimado:      {fmt_num(metrics["inventario_estimado_kg"])} kg')
    print(f'• Tazas servidas en casa:   {metrics["total_tazas"]:,}')

    if (metrics['ahorro_por_taza'] > 0) and (metrics["total_tazas"] > 0) and np.isfinite(metrics['costo_promedio_taza']):
        print(f'• Ahorro por taza:          {fmt_mon(metrics["ahorro_por_taza"])}')
        print(f'• Tazas para recuperar:     {fmt_num(metrics["tazas_para_recuperar"], 0)}')
        print(f'• Ahorro total acumulado:   {fmt_mon(metrics["ahorro_total"])}')
    elif metrics["total_tazas"] == 0:
        print('• Ahorro por taza:          N/D (no hay tazas registradas)')
        print('• Tazas para recuperar:     N/D')
        print('• Ahorro total acumulado:   $0.00 MXN')
    else:
        print('• Ahorro por taza:          No aplicable (costo en casa ≥ precio externo o faltan datos)')
        print('• Tazas para recuperar:     No aplicable')
        print(f'• Ahorro total acumulado:   {fmt_mon(metrics["ahorro_total"])}')

    if fecha_min and fecha_max and dias_consumo:
        print(f'• Primer registro: {fecha_min.date()}  • Último: {fecha_max.date()}  • Días: {dias_consumo:,}')
        if metrics['tazas_por_dia_real'] is not None:
            print(f'• Tazas/día (real): {fmt_num(metrics["tazas_por_dia_real"])}  • Ahorro diario (real): {fmt_mon(metrics["ahorro_diario_real"])}')
    else:
        print('• No hay datos de fecha para proyecciones reales.')

    if metrics["inventario_estimado_kg"] <= cfg.alerta_inventario_kg:
        print(f'⚠️  Alerta: inventario bajo (≤ {cfg.alerta_inventario_kg} kg). Reaprovisiona.')
    if (metrics['dias_restantes'] is not None) and np.isfinite(metrics['dias_restantes']) and (metrics['dias_restantes'] <= cfg.alerta_inventario_dias):
        eta = metrics["fecha_agotamiento"].date() if metrics["fecha_agotamiento"] is not None else "N/D"
        print(f'⚠️  Alerta: quedan ~{metrics["dias_restantes"]:.1f} días de inventario. ETA: {eta}.')

    print('\n💼 PROMEDIOS DE COSTO (agregados)')
    print(f'• Costo promedio por compra: {fmt_mon(metrics["costo_promedio_por_compra"])}')
    print(f'• Costo promedio por kg:     {fmt_mon(metrics["costo_promedio_por_kg"])}')

    print('\n☕ DETALLE DE TAZAS POR TIPO')
    for tipo, cantidad in metrics['total_tazas_tipo'].items():
        print(f'• {tipo:<6}: {int(cantidad):,}')

    if metrics['bolsas_df'] is not None and not metrics['bolsas_df'].empty:
        b = metrics['bolsas_df']
        print('\n⏱️ DURACIÓN DE CADA BOLSA (días entre compras)')
        print(b[['inicio', 'dias_entre_compras']].to_string(index=False))

        dias_compra = b['dias_entre_compras'].dropna()
        if not dias_compra.empty:
            print(f'\n📊 Promedio días entre compras: {fmt_num(dias_compra.mean(),0)}')
            print(f'Mín: {int(dias_compra.min())}  Máx: {int(dias_compra.max())}')
    else:
        print('\n⏱️ DURACIÓN DE CADA BOLSA: N/D (faltan compras con fecha y gramos)')

    print('\n⏳ ROI Y AHORRO POST-ROI')
    roi = metrics['roi']
    if np.isfinite(metrics['ahorro_por_taza']) and metrics['ahorro_por_taza'] > 0 and metrics['diarios'] is not None:
        if roi['fecha_roi'] is not None:
            print(f'• ROI alcanzado el: {roi["fecha_roi"].date()}  (en {roi["dias_hasta_roi"]} días desde el primer registro)')
            print(f'• Ahorro acumulado total: {fmt_mon(roi["ahorro_total_acum"])}')
            print(f'• Ahorro post-ROI estimado: {fmt_mon(roi["ahorro_post_roi"])}')
        else:
            faltante = cfg.costo_cafetera - roi['ahorro_total_acum']
            print('• ROI aún no alcanzado.')
            print(f'• Ahorro acumulado: {fmt_mon(roi["ahorro_total_acum"])}  • Faltante para ROI: {fmt_mon(faltante)}')
    else:
        print('• No se puede calcular ROI (ahorro por taza ≤ 0 o faltan fechas).')

    print('\n🧮 ESCENARIOS HIPOTÉTICOS DE PAGO')
    if (metrics['ahorro_por_taza'] > 0) and (metrics["total_tazas"] > 0) and np.isfinite(metrics['costo_promedio_taza']):
        for tpd in [1, 2, 3, 4]:
            dias = metrics['tazas_para_recuperar'] / tpd
            print(f'• {tpd} taza(s)/día: {int(round(dias,0))} días (~{dias/30:,.1f} meses)')
        if (metrics['tazas_por_dia_real'] is not None) and np.isfinite(metrics['tazas_por_dia_real']):
            dias_real = metrics['tazas_para_recuperar'] / metrics['tazas_por_dia_real']
            print(f'• Según tu ritmo ({metrics["tazas_por_dia_real"]:,.2f} tazas/día): {int(round(dias_real,0))} días (~{dias_real/30:,.1f} meses)')
    else:
        print('• No hay recuperación si no hay tazas, si el ahorro ≤ 0 o si faltan gramos.')

# ======= ORQUESTADOR =======
def run_dashboard(cfg: Config):
    df = load_data(cfg)
    if df.empty:
        print("El archivo CSV no se pudo cargar o está vacío. Verifica la ruta o el formato.")
        return None
    df, col_fecha, fecha_min, fecha_max, dias_consumo = detect_and_parse_dates(df)
    df = ensure_taza_columns(df, cfg)

    issues = validate_schema(df, cfg)
    if issues:
        print("🔎 Validación: se detectaron los siguientes puntos a corregir:")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")

    _ = sync_total_tazas(df, cfg)
    metrics = compute_metrics(df, cfg, col_fecha, fecha_min, fecha_max, dias_consumo)
    print_report(metrics, cfg)
    return metrics

# ===================== EJECUCIÓN =====================
if __name__ == "__main__":
    cfg = Config(
        ruta_csv="https://docs.google.com/spreadsheets/d/e/2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n/pub?gid=0&single=true&output=csv",
        costo_cafetera=12239,
        precio_cafeteria={
            'Elena': 70, 'Chavelete': 60, 'Coffee King': 50, 'Starbucks': 58,
            'Quentin 1': 55, 'Quentin 2': 70, 'Otro': 40, 'Garat': 46,
            'The Coffe': 55, 'Cielito': 55
        },
        columnas_tazas=['espresso','quad','6oz','8oz','10oz','12oz','14oz','16oz','18oz'],
        gramos_por_taza={'espresso':18,'quad':40,'6oz':10,'8oz':14,'10oz':18,'12oz':21,'14oz':25,'16oz':28,'18oz':31},
        generar_graficas=False
    )
    metrics = run_dashboard(cfg)

    from matplotlib import pyplot as plt



