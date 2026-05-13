import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple
from datetime import timedelta, datetime
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
    ruta_csv: str = r"C:\Users\alber\OneDrive\Desktop\Tripleten\Cafetera\Cafe\historial_cafe.csv"


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


def guardar_historico_csv(datos: dict, ruta: str):
    """
    Logica 'Upsert': Mantiene solo un registro por día.
    """
    fecha_hoy = datetime.now().strftime('%Y-%m-%d')
    datos_ordenados = {'fecha_registro': fecha_hoy, **datos}
    df_nuevo = pd.DataFrame([datos_ordenados])

    if os.path.exists(ruta):
        try:
            df_hist = pd.read_csv(ruta)
            if 'fecha_registro' in df_hist.columns:
                df_hist = df_hist[df_hist['fecha_registro'] != fecha_hoy]

            df_final = pd.concat([df_hist, df_nuevo], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Error leyendo el histórico anterior. Detalle: {e}")
            df_final = df_nuevo
    else:
        df_final = df_nuevo

    try:
        df_final.to_csv(ruta, index=False, encoding='utf-8-sig')
        print(
            f"💾 Snapshot del día ({fecha_hoy}) guardado exitosamente en CSV.")
    except Exception as e:
        print(f"❌ Error crítico al guardar CSV: {e}")
        print("💡 TIP: Asegúrate de cerrar el archivo en Excel antes de correr el código.")


def run_dashboard(cfg: Config):
    df_c = fetch_csv(cfg.id_pub, cfg.gid_consumo, "Consumo")
    df_p = fetch_csv(cfg.id_pub, cfg.gid_precios, "Cafeterias")
    df_g = fetch_csv(cfg.id_pub, cfg.gid_gramajes, "Gramajes")

    if df_c is None or df_g is None:
        print("🛑 Error crítico al descargar datos.")
        return

    # 1. MAPEO DE GRAMAJES
    nombres_limpios = df_g['tipo de bebida'].astype(
        str).str.strip().str.lower()
    gramos_limpios = pd.to_numeric(df_g['café (g)'], errors='coerce').fillna(0)
    gramos_map = dict(zip(nombres_limpios, gramos_limpios))
    columnas_tazas = list(gramos_map.keys())

    # 2. PRECIOS DINÁMICOS
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

    # 4. BALANCE E INVENTARIO
    df_compras = df_c[df_c['gramos'] > 0].copy()
    cafe_comprado_g = df_compras['gramos'].sum()
    cafe_consumido_g = df_c['g_est'].sum()

    bolsas_activas = df_compras[pd.isna(
        df_compras['cierre_dt']) & pd.isna(df_compras['cierre_crudo'])]
    bolsas_activas = bolsas_activas[~bolsas_activas['cafeteria'].astype(
        str).str.contains('molido', case=False, na=False)]

    inv_real = 0.0
    cafe_activo = "Ninguno"

    if not bolsas_activas.empty:
        nombres_lista = bolsas_activas[bolsas_activas['cafeteria']
                                       != '']['cafeteria'].unique()
        cafe_activo = " + ".join(nombres_lista) if len(
            nombres_lista) > 0 else "Sin Nombre"

        total_gramos_activos = bolsas_activas['gramos'].sum()
        fecha_ultimo_cierre = df_compras['cierre_dt'].max()

        if pd.notna(fecha_ultimo_cierre):
            consumo_activo = df_c[df_c[col_fecha] >
                                  fecha_ultimo_cierre]['g_est'].sum()
        else:
            consumo_activo = df_c['g_est'].sum()

        inv_real = max(0.0, total_gramos_activos - consumo_activo)

    diferencia_global = cafe_comprado_g - cafe_consumido_g
    merma_historica = max(0.0, diferencia_global - inv_real)

    # 5. ANÁLISIS FINANCIERO Y RENDIMIENTO
    df_c_valid = df_c.dropna(subset=[col_fecha]).copy()
    total_tazas = df_c['t_total'].sum()
    costo_total_insumos = df_compras['costo'].sum()
    costo_por_gramo = costo_total_insumos / \
        cafe_comprado_g if cafe_comprado_g > 0 else 0
    costo_promedio_taza_casa = (
        cafe_consumido_g * costo_por_gramo) / total_tazas if total_tazas > 0 else 0

    f_compra = pd.to_datetime(cfg.fecha_compra, dayfirst=True)
    diarios = df_c_valid.groupby(df_c_valid[col_fecha].dt.normalize()).agg(
        g_dia=('g_est', 'sum'), t_dia=('t_total', 'sum')
    ).sort_index()

    ahorro_total = 0.0
    if not diarios.empty:
        rango = pd.date_range(start=f_compra, end=diarios.index.max())
        diarios = diarios.reindex(rango).fillna(0)
        diarios['año'] = diarios.index.year
        diarios['precio_mercado_hoy'] = diarios['año'].map(
            precios_por_anio).fillna(precio_promedio_general)
        diarios['ahorro_dia'] = diarios['t_dia'] * \
            diarios['precio_mercado_hoy']
        diarios['ahorro_acum'] = diarios['ahorro_dia'].cumsum()

        ahorro_total = diarios['ahorro_acum'].iloc[-1]

        metricas_anuales = diarios.groupby('año').agg(
            tazas_tot=('t_dia', 'sum'),
            ahorro_tot=('ahorro_dia', 'sum')
        )
    else:
        metricas_anuales = pd.DataFrame()

    ganancia_post_roi = max(0.0, ahorro_total - cfg.costo_cafetera)

    # 6. ML PREDICTION Y RANKING
    ritmo_ml, dias_ml = predict_exhaustion_ml(diarios, inv_real)
    eta_ml = diarios.index.max() + timedelta(days=float(dias_ml)) if (dias_ml >
                                                                      0 and not diarios.empty) else None

    df_ranking = df_compras[~df_compras['cafeteria'].astype(
        str).str.contains('molido', case=False, na=False)]
    ranking_cafes = df_ranking[df_ranking['cafeteria']
                               != '']['cafeteria'].value_counts()
    ranking_cafes = ranking_cafes[ranking_cafes > 1]

    # ==============================================================
    # 📝 EXPORTACIÓN AL CSV (UPSERT)
    # ==============================================================
    totales_por_tipo = df_c[tazas_existentes].sum(
    ).sort_values(ascending=False)

    datos_exportacion = {
        'hora_ejecucion': datetime.now().strftime('%H:%M:%S'),
        'cafe_activo': cafe_activo,
        'inventario_restante_g': round(inv_real, 2),
        'ritmo_consumo_g_dia': round(ritmo_ml, 2),
        'eta_agotamiento': eta_ml.strftime('%Y-%m-%d') if eta_ml else "Sin datos",
        'ahorro_bruto_acumulado_mxn': round(ahorro_total, 2),
        'ganancia_post_roi_mxn': round(ganancia_post_roi, 2),
        'costo_promedio_taza_casa_mxn': round(costo_promedio_taza_casa, 2),
        'precio_promedio_calle_mxn': round(precio_promedio_general, 2),
        'total_tazas_servidas': int(total_tazas),
        'merma_historica_g': round(merma_historica, 2)
    }

    # Desglose de tazas por año para el CSV
    if not metricas_anuales.empty:
        for anio, row in metricas_anuales.iterrows():
            if row['tazas_tot'] > 0:
                datos_exportacion[f'tazas_totales_{anio}'] = int(
                    row['tazas_tot'])

    # Tipos de tazas globales
    for taza, cantidad in totales_por_tipo.items():
        if cantidad > 0:
            datos_exportacion[f'tazas_tipo_{taza}'] = int(cantidad)

    # ==============================================================
    # 📊 REPORTE VISUAL (DASHBOARD EN CONSOLA)
    # ==============================================================
    print("\n" + "="*55)
    print(" ☕ TABLERO DE DATOS CAFETERO")
    print("="*55)

    print("\n📦 INVENTARIO REAL Y SMART ML")
    print(f" • Bolsa(s) Activa(s):         {cafe_activo}")
    print(f" • Peso restante calculado:    {inv_real:.0f} g")
    if eta_ml and inv_real > 0:
        print(f" • Ritmo actual aprendido:     {ritmo_ml:.1f} g/día")
        print(
            f" • Fecha de agotamiento (ETA): {eta_ml.date()} (en {dias_ml:.1f} días)")
    else:
        print(" ⚠️  No hay datos suficientes para proyectar agotamiento.")

    print("\n⚖️ BALANCE DE MATERIA (MERMAS Y PÉRDIDAS)")
    print(f" • Total comprado (Histórico): {cafe_comprado_g:.0f} g")
    print(f" • Total servido (Real):       {cafe_consumido_g:.0f} g")
    print(f" • Diferencia global:          {diferencia_global:.0f} g")
    print(f" • Merma histórica de bolsas:  {merma_historica:.0f} g")

    print("\n📅 ANÁLISIS DESAGREGADO POR AÑO")
    print(f" 🏆 TOTAL HISTÓRICO:           {int(total_tazas)} tazas servidas")
    if not metricas_anuales.empty:
        for anio, row in metricas_anuales.iterrows():
            precio_usado = precios_por_anio.get(anio, precio_promedio_general)
            if row['tazas_tot'] > 0:
                print(f" 🔹 AÑO {anio} | Mercado: ${precio_usado:.2f} MXN/taza")
                print(
                    f"    • Tazas consumidas:        {int(row['tazas_tot'])} tazas")
                print(
                    f"    • Ahorro generado:         ${row['ahorro_tot']:,.2f} MXN")

    print("\n⏳ ESTADO DEL RETORNO DE INVERSIÓN (ROI)")
    print(f" • Inversión (Máquina):        ${cfg.costo_cafetera:,.2f} MXN") 
    print(f" • Ahorro Acumulado Bruto:     ${ahorro_total:,.2f} MXN")
    if ahorro_total >= cfg.costo_cafetera:
        dia_roi = diarios[diarios['ahorro_acum']
                          >= cfg.costo_cafetera].index[0]
        print(f" ✅ ROI Alcanzado el:          {dia_roi.date()}")
        print(f" 💵 Ganancia Post-ROI: ${ganancia_post_roi:,.2f} MXN")
    else:
        print(
            f" ⏳ Faltante para ROI:         ${(cfg.costo_cafetera - ahorro_total):,.2f} MXN")

    print("\n☕ DESGLOSE DE TAZAS Y CAFÉS PREFERIDOS")
    for taza, cantidad in totales_por_tipo.items():
        if cantidad > 0:
            print(f" • {taza:<10}: {int(cantidad):>4} tazas")

    print(f"\n🏷️ RANKING GLOBAL DE COMPRAS (Recurrentes)")
    if not ranking_cafes.empty:
        for cafe, veces in ranking_cafes.head(5).items():
            print(f" • {cafe}: {veces} compras")
    else:
        print(" • Aún no hay cafés con más de 1 compra.")

    print("="*55)

    # 💾 Ejecutar guardado (Upsert Diario)
    guardar_historico_csv(datos_exportacion, cfg.ruta_csv)


if __name__ == "__main__":
    ID_PUB = "2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n"
    config = Config(
        id_pub=ID_PUB,
        # Importante: Mantén tus GID de consumo, precios y gramajes tal cual los necesitas.
        gid_consumo="0",
        gid_precios="49728846",
        gid_gramajes="1827085190",
        costo_cafetera=12239
    )
    run_dashboard(config)
