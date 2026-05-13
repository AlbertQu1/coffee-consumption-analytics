import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class Config:
    id_pub: str
    gid_consumo: str
    gid_precios: str
    gid_gramajes: str
    costo_cafetera: float
    fecha_compra: str = "30/05/2025"


def predict_exhaustion_ml(diarios: pd.DataFrame, inv_actual_g: float) -> Tuple[float, float]:
    if diarios is None or len(diarios) < 3 or inv_actual_g <= 0:
        return 0.0, 0.0
    df_ml = diarios.tail(15).copy().reset_index()
    df_ml['n_dia'] = np.arange(len(df_ml))
    X, y = df_ml[['n_dia']].values, df_ml['g_dia'].values
    model = LinearRegression().fit(X, y)
    ritmo = max(0.1, model.predict([[len(df_ml)]])[0])
    return ritmo, inv_actual_g / ritmo


def fetch_csv(id_pub, gid, nombre):
    url = f"https://docs.google.com/spreadsheets/d/e/{id_pub}/pub?gid={gid}&single=true&output=csv"
    try:
        df = pd.read_csv(url, skip_blank_lines=True)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  ❌ Error de red en '{nombre}': {e}")
        return None


def run_dashboard(cfg: Config):
    df_c = fetch_csv(cfg.id_pub, cfg.gid_consumo, "Consumo")
    df_p = fetch_csv(cfg.id_pub, cfg.gid_precios, "Cafeterias")
    df_g = fetch_csv(cfg.id_pub, cfg.gid_gramajes, "Gramajes")

    if df_c is None or df_g is None:
        print("🛑 Error crítico. Revisa la conexión o los GIDs.")
        return

    # 1. MAPEO DE GRAMAJES
    nombres_limpios = df_g['tipo de bebida'].astype(
        str).str.strip().str.lower()
    gramos_limpios = pd.to_numeric(df_g['café (g)'], errors='coerce').fillna(0)
    gramos_map = dict(zip(nombres_limpios, gramos_limpios))
    columnas_tazas = list(gramos_map.keys())

    # 2. INFLACIÓN Y PRECIOS DINÁMICOS POR AÑO
    if df_p is not None and 'año' in df_p.columns and 'precio' in df_p.columns:
        df_p['año'] = pd.to_numeric(
            df_p['año'], errors='coerce').fillna(2025).astype(int)
        precios_por_anio = df_p.groupby('año')['precio'].mean().to_dict()
        precio_promedio_general = df_p['precio'].mean()
    else:
        precios_por_anio = {2025: 55.45}
        precio_promedio_general = 55.45

    # 3. PROCESAMIENTO GENERAL
    col_fecha = 'fecha'
    df_c[col_fecha] = pd.to_datetime(
        df_c.get(col_fecha), dayfirst=True, errors='coerce')
    df_c['costo'] = pd.to_numeric(
        df_c.get('costo', 0), errors='coerce').fillna(0)
    df_c['gramos'] = pd.to_numeric(
        df_c.get('gramos', 0), errors='coerce').fillna(0)

    # LIMPIEZA DE CIERRES Y NOMBRES
    df_c['cierre_crudo'] = df_c.get('cierre', pd.Series(
        dtype=str)).astype(str).str.strip().str.lower()
    df_c['cierre_crudo'] = df_c['cierre_crudo'].replace(
        ['nan', 'nat', 'none', '<na>', ''], np.nan)
    df_c['cierre_dt'] = pd.to_datetime(
        df_c['cierre_crudo'], dayfirst=True, errors='coerce')

    if 'cafeteria' in df_c.columns:
        df_c['cafeteria'] = df_c['cafeteria'].astype(
            str).str.strip().str.title()
        df_c['cafeteria'] = df_c['cafeteria'].replace(
            ['Nan', 'Nat', 'None', '<Na>'], '')

    tazas_existentes = [t for t in columnas_tazas if t in df_c.columns]
    gramos_vec = np.array([gramos_map.get(t, 0.0) for t in tazas_existentes])

    df_c[tazas_existentes] = df_c[tazas_existentes].apply(
        pd.to_numeric, errors='coerce').fillna(0)
    df_c['g_est'] = (df_c[tazas_existentes].to_numpy()
                     * gramos_vec).sum(axis=1)
    df_c['t_total'] = df_c[tazas_existentes].sum(axis=1)

    # 4. BALANCE DE MATERIA E INVENTARIO
    df_compras = df_c[df_c['gramos'] > 0].copy()
    cafe_comprado_g = df_compras['gramos'].sum()
    cafe_consumido_g = df_c['g_est'].sum()

    # Bolsas activas: Sin cierre y sin la palabra 'molido'
    bolsas_activas = df_compras[pd.isna(
        df_compras['cierre_dt']) & pd.isna(df_compras['cierre_crudo'])]
    bolsas_activas = bolsas_activas[~bolsas_activas['cafeteria'].astype(
        str).str.contains('molido', case=False, na=False)]

    inv_real = 0.0
    nombres_bolsas_activas = "Ninguna"

    if not bolsas_activas.empty:
        nombres_lista = bolsas_activas[bolsas_activas['cafeteria']
                                       != '']['cafeteria'].unique()
        nombres_bolsas_activas = " + ".join(nombres_lista) if len(
            nombres_lista) > 0 else "Sin Nombre"

        total_gramos_activos = bolsas_activas['gramos'].sum()
        fecha_ultimo_cierre = df_compras['cierre_dt'].max()

        if pd.notna(fecha_ultimo_cierre):
            consumo_activo = df_c[df_c[col_fecha] >
                                  fecha_ultimo_cierre]['g_est'].sum()
        else:
            consumo_activo = df_c['g_est'].sum()

        inv_real = max(0.0, total_gramos_activos - consumo_activo)

    # 5. ANÁLISIS POR AÑO (COMPRAS, CONSUMO Y MERMA)
    df_c_valid = df_c.dropna(subset=[col_fecha]).copy()
    df_c_valid['año_consumo'] = df_c_valid[col_fecha].dt.year
    df_compras_valid = df_compras.dropna(subset=[col_fecha]).copy()
    df_compras_valid['año_compra'] = df_compras_valid[col_fecha].dt.year

    compras_por_ano = df_compras_valid.groupby('año_compra')['gramos'].sum()
    consumo_por_ano = df_c_valid.groupby('año_consumo')['g_est'].sum()

    anios_unicos = sorted(list(set(compras_por_ano.index)
                          | set(consumo_por_ano.index)))
    anio_actual = max(anios_unicos) if anios_unicos else 2026

    # 6. ROI Y SERIES DE TIEMPO
    f_compra = pd.to_datetime(cfg.fecha_compra, dayfirst=True)
    diarios = df_c_valid.groupby(df_c_valid[col_fecha].dt.normalize()).agg(
        g_dia=('g_est', 'sum'), t_dia=('t_total', 'sum')
    ).sort_index()

    if not diarios.empty:
        rango = pd.date_range(start=f_compra, end=diarios.index.max())
        diarios = diarios.reindex(rango).fillna(0)
        diarios['año'] = diarios.index.year
        diarios['precio_mercado_hoy'] = diarios['año'].map(
            precios_por_anio).fillna(precio_promedio_general)
        diarios['ahorro_dia'] = diarios['t_dia'] * \
            diarios['precio_mercado_hoy']
        diarios['ahorro_acum'] = diarios['ahorro_dia'].cumsum()

        metricas_anuales = diarios.groupby('año').agg(
            tazas_tot=('t_dia', 'sum'),
            ahorro_tot=('ahorro_dia', 'sum')
        )
    else:
        metricas_anuales = pd.DataFrame()

    # 7. ML PREDICTION Y RANKING LIMPIO
    ritmo_ml, dias_ml = predict_exhaustion_ml(diarios, inv_real)
    eta_ml = diarios.index.max() + timedelta(days=float(dias_ml)) if (dias_ml >
                                                                      0 and not diarios.empty) else None

    df_ranking = df_compras[~df_compras['cafeteria'].astype(
        str).str.contains('molido', case=False, na=False)]
    ranking_cafes = df_ranking[df_ranking['cafeteria']
                               != '']['cafeteria'].value_counts()
    ranking_cafes = ranking_cafes[ranking_cafes > 1]

    # ==============================================================
    # 📊 REPORTE VISUAL
    # ==============================================================
    print("\n" + "="*55)
    print(" ☕ TABLERO DE DATOS CAFETERO")
    print("="*55)

    print("\n📦 INVENTARIO REAL Y SMART ML")
    print(f" • Bolsa(s) Activa(s):         {nombres_bolsas_activas}")
    print(f" • Peso restante calculado:    {inv_real:.0f} g")
    if eta_ml and inv_real > 0:
        print(f" • Ritmo actual aprendido:     {ritmo_ml:.1f} g/día")
        print(
            f" • Fecha de agotamiento (ETA): {eta_ml.date()} (en {dias_ml:.1f} días)")

    print("\n⚖️ BALANCE DE MATERIA Y MERMAS POR AÑO")
    for anio in anios_unicos:
        c_compra = compras_por_ano.get(anio, 0)
        c_consumo = consumo_por_ano.get(anio, 0)

        # Si es el año actual, descontamos el inventario activo de la merma
        if anio == anio_actual:
            merma_anio = max(0.0, c_compra - c_consumo - inv_real)
        else:
            merma_anio = max(0.0, c_compra - c_consumo)

        porcentaje_anio = (merma_anio / c_compra * 100) if c_compra > 0 else 0
        print(f" 🔹 AÑO {int(anio)}")
        print(
            f"    • Comprado: {c_compra:.0f} g | Consumo Real: {c_consumo:.0f} g")
        print(
            f"    • Merma/Pérdida: {merma_anio:.0f} g ({porcentaje_anio:.1f}%)")

    print("\n📅 RENDIMIENTO Y AHORRO POR AÑO")
    if not metricas_anuales.empty:
        for anio, row in metricas_anuales.iterrows():
            precio_usado = precios_por_anio.get(anio, precio_promedio_general)
            if row['tazas_tot'] > 0:
                print(f" 🔹 AÑO {anio} | Mercado: ${precio_usado:.2f} MXN/taza")
                print(
                    f"    • Tazas: {int(row['tazas_tot'])} | Ahorro: ${row['ahorro_tot']:,.2f} MXN")

    ahorro_total = diarios['ahorro_acum'].iloc[-1] if not diarios.empty else 0
    print("\n⏳ ESTADO DEL RETORNO DE INVERSIÓN (ROI)")
    print(f" • Inversión (Máquina):        ${cfg.costo_cafetera:,.2f} MXN")
    print(f" • Ahorro Acumulado Bruto:     ${ahorro_total:,.2f} MXN")
    if ahorro_total >= cfg.costo_cafetera:
        dia_roi = diarios[diarios['ahorro_acum']
                          >= cfg.costo_cafetera].index[0]
        print(f" ✅ ROI Alcanzado el:          {dia_roi.date()}")
    else:
        print(
            f" ⏳ Faltante para ROI:         ${(cfg.costo_cafetera - ahorro_total):,.2f} MXN")

    print("\n☕ DESGLOSE DE TAZAS Y CAFÉS PREFERIDOS")
    totales_por_tipo = df_c[tazas_existentes].sum(
    ).sort_values(ascending=False)
    for taza, cantidad in totales_por_tipo.items():
        if cantidad > 0:
            print(f" • {taza:<10}: {int(cantidad):>4} tazas")

    print(f"\n🏷️ RANKING GLOBAL DE COMPRAS (Recurrentes)")
    if not ranking_cafes.empty:
        for cafe, veces in ranking_cafes.head(5).items():
            print(f" • {cafe}: {veces} compras")
    else:
        print(" • Aún no hay cafés de especialidad con más de 1 compra.")

    print("="*55)


if __name__ == "__main__":
    ID_PUB = "2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n"
    config = Config(
        id_pub=ID_PUB,
        gid_consumo="0",
        gid_precios="49728846",
        gid_gramajes="1827085190",
        costo_cafetera=12239
    )
    run_dashboard(config)

