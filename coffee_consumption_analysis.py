import os
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class Config:
    id_pub: str
    gid_consumo: str
    gid_precios: str
    gid_gramajes: str
    costo_cafetera: float
    fecha_compra: str = "30/05/2025"
    ruta_csv: str = (
        r"C:\Users\alber\OneDrive\Desktop\Tripleten\Cafetera\Cafe"
        r"\historial_cafe.csv"
    )
    guardar_snapshot: bool = True


def normalize_key(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    by_normalized = {normalize_key(col): col for col in df.columns}
    for candidate in candidates:
        found = by_normalized.get(normalize_key(candidate))
        if found is not None:
            return found
    return None


def money(value: float) -> str:
    return f"${value:,.2f} MXN"


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "invierno"
    if month in (3, 4, 5):
        return "primavera"
    if month in (6, 7, 8):
        return "verano"
    return "otono"


def weekday_name(day: int) -> str:
    names = {
        0: "lunes",
        1: "martes",
        2: "miercoles",
        3: "jueves",
        4: "viernes",
        5: "sabado",
        6: "domingo",
    }
    return names.get(int(day), "sin dia")


def fetch_csv(id_pub: str, gid: str, nombre: str) -> Optional[pd.DataFrame]:
    url = (
        f"https://docs.google.com/spreadsheets/d/e/{id_pub}/pub?"
        f"gid={gid}&single=true&output=csv"
    )
    try:
        df = pd.read_csv(url, skip_blank_lines=True)
        df.columns = [normalize_key(c) for c in df.columns]
        return df
    except Exception as exc:
        print(f"❌ Error de red en '{nombre}': {exc}")
        return None


def guardar_historico_csv(datos: dict, ruta: str) -> None:
    fecha_hoy = datetime.now().strftime("%Y-%m-%d")
    datos_ordenados = {"fecha_registro": fecha_hoy, **datos}
    df_nuevo = pd.DataFrame([datos_ordenados])

    if os.path.exists(ruta):
        try:
            df_hist = pd.read_csv(ruta)
            if "fecha_registro" in df_hist.columns:
                df_hist = df_hist[df_hist["fecha_registro"] != fecha_hoy]
            df_final = pd.concat([df_hist, df_nuevo], ignore_index=True)
        except Exception as exc:
            print(f"⚠️ No se pudo leer el histórico anterior: {exc}")
            df_final = df_nuevo
    else:
        df_final = df_nuevo

    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_final.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"💾 Snapshot diario ({fecha_hoy}) guardado en CSV.")
    except Exception as exc:
        print(f"❌ Error crítico al guardar CSV: {exc}")
        print("💡 Tip: cierra el archivo en Excel antes de correr el código.")


def build_price_context(df_p: Optional[pd.DataFrame]) -> tuple[dict[int, float], float]:
    precio_promedio = 55.45
    precios_por_anio = {2025: precio_promedio}

    if df_p is None:
        return precios_por_anio, precio_promedio

    anio_col = find_column(df_p, ["anio", "ano", "año", "year"])
    precio_col = find_column(df_p, ["precio", "precio taza", "costo"])
    if anio_col is None or precio_col is None:
        return precios_por_anio, precio_promedio

    df_p = df_p.copy()
    df_p[anio_col] = pd.to_numeric(df_p[anio_col], errors="coerce").fillna(2025)
    df_p[anio_col] = df_p[anio_col].astype(int)
    df_p[precio_col] = pd.to_numeric(df_p[precio_col], errors="coerce")
    precios_por_anio = df_p.groupby(anio_col)[precio_col].mean().dropna().to_dict()
    precio_promedio = float(df_p[precio_col].mean())
    return precios_por_anio, precio_promedio


def build_gram_map(df_g: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    tipo_col = find_column(df_g, ["tipo de bebida", "bebida", "tipo"])
    cafe_g_col = find_column(df_g, ["cafe (g)", "cafe g", "gramos cafe"])
    if tipo_col is None or cafe_g_col is None:
        raise ValueError("La hoja de gramajes necesita tipo de bebida y cafe (g).")

    tipos = df_g[tipo_col].astype(str).map(normalize_key)
    gramos = pd.to_numeric(df_g[cafe_g_col], errors="coerce").fillna(0)
    gramos_map = dict(zip(tipos, gramos))
    return gramos_map, list(gramos_map.keys())


def prepare_rows(
    df_c: pd.DataFrame,
    gramos_map: dict[str, float],
    columnas_gramajes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    col_fecha = find_column(df_c, ["fecha", "date"])
    if col_fecha is None:
        raise ValueError("La hoja de consumo necesita una columna fecha.")

    costo_col = find_column(df_c, ["costo", "precio pagado"])
    gramos_col = find_column(df_c, ["gramos", "gramaje", "peso"])
    cierre_col = find_column(df_c, ["cierre", "fecha cierre"])
    cafeteria_col = find_column(df_c, ["cafeteria", "cafe", "origen"])
    tazas_total_col = find_column(df_c, ["# de tazas", "numero de tazas", "tazas"])

    df = df_c.copy()
    df["fecha_apertura"] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")
    df["cierre_crudo"] = (
        df[cierre_col].astype(str).str.strip().str.lower()
        if cierre_col
        else pd.Series(dtype=str)
    )
    df["cierre_crudo"] = df["cierre_crudo"].replace(
        ["nan", "nat", "none", "<na>", ""], np.nan
    )
    df["fecha_cierre"] = pd.to_datetime(df["cierre_crudo"], dayfirst=True, errors="coerce")
    df["cafe"] = (
        df[cafeteria_col].astype(str).str.strip().str.title()
        if cafeteria_col
        else "Sin Nombre"
    )
    df["cafe"] = df["cafe"].replace(["Nan", "Nat", "None", "<Na>", ""], "Sin Nombre")
    df["gramos"] = (
        pd.to_numeric(df[gramos_col], errors="coerce").fillna(0)
        if gramos_col
        else 0
    )
    df["costo"] = (
        pd.to_numeric(df[costo_col], errors="coerce").fillna(0)
        if costo_col
        else 0
    )

    tazas_cols = [col for col in columnas_gramajes if col in df.columns]
    if not tazas_cols:
        raise ValueError("No encontré columnas de tazas que coincidan con gramajes.")

    gramos_vec = np.array([gramos_map.get(col, 0.0) for col in tazas_cols])
    df[tazas_cols] = df[tazas_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["g_consumidos_est"] = (df[tazas_cols].to_numpy() * gramos_vec).sum(axis=1)
    df["tazas_calculadas"] = df[tazas_cols].sum(axis=1)
    if tazas_total_col:
        df["tazas_total"] = pd.to_numeric(df[tazas_total_col], errors="coerce").fillna(
            df["tazas_calculadas"]
        )
    else:
        df["tazas_total"] = df["tazas_calculadas"]

    df = df[df["gramos"] > 0].copy()
    today = pd.Timestamp(datetime.now().date())
    df["es_molido_anual"] = df["cafe"].str.contains("molido", case=False, na=False)
    df["esta_cerrada"] = df["fecha_cierre"].notna()
    df["fecha_fin_analitica"] = df["fecha_cierre"].fillna(today)
    df["dias_ciclo"] = (df["fecha_fin_analitica"] - df["fecha_apertura"]).dt.days + 1
    df["ciclo_valido"] = df["fecha_apertura"].notna() & (df["dias_ciclo"] > 0)
    df["mes_apertura"] = df["fecha_apertura"].dt.month
    df["anio_apertura"] = df["fecha_apertura"].dt.year
    df["temporada_apertura"] = df["mes_apertura"].apply(
        lambda value: season_name(value) if pd.notna(value) else "sin temporada"
    )
    df["g_por_taza"] = np.where(
        df["tazas_total"] > 0, df["g_consumidos_est"] / df["tazas_total"], 0
    )
    df["costo_por_taza"] = np.where(df["tazas_total"] > 0, df["costo"] / df["tazas_total"], 0)
    df["costo_por_250g"] = np.where(df["gramos"] > 0, df["costo"] / df["gramos"] * 250, 0)
    df["tazas_por_250g"] = np.where(df["gramos"] > 0, df["tazas_total"] / df["gramos"] * 250, 0)
    df["es_regalo"] = df["costo"] == 0
    df["dia_cierre"] = df["fecha_cierre"].dt.weekday
    df["dia_cierre_nombre"] = df["dia_cierre"].apply(
        lambda value: weekday_name(value) if pd.notna(value) else "sin cierre"
    )
    df["tipo_dia_cierre"] = np.where(df["dia_cierre"].isin([5, 6]), "fin de semana", "entre semana")

    return df.reset_index(drop=True), tazas_cols


def distribute_cycles_by_day(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = cycles[cycles["ciclo_valido"]].copy()

    for _, row in valid.iterrows():
        start = row["fecha_apertura"].normalize()
        end = row["fecha_fin_analitica"].normalize()
        days = int(row["dias_ciclo"])
        if days <= 0:
            continue

        for date in pd.date_range(start, end):
            rows.append(
                {
                    "fecha": date,
                    "cafe": row["cafe"],
                    "tazas": float(row["tazas_total"]) / days,
                    "gramos": float(row["g_consumidos_est"]) / days,
                    "estado": "cerrada" if row["esta_cerrada"] else "activa",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["fecha", "cafe", "tazas", "gramos", "estado"])
    return pd.DataFrame(rows)


MODEL_FEATURES = ["gramos", "costo", "costo_por_250g", "mes_apertura", "es_regalo"]


def build_features(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(rows[MODEL_FEATURES + ["cafe"]], columns=["cafe"], dummy_na=False)


def train_regression_model(
    cycles: pd.DataFrame, target: str
) -> tuple[Optional[RandomForestRegressor], list[str], Optional[float]]:
    closed = cycles[
        cycles["esta_cerrada"] & cycles["ciclo_valido"] & cycles[target].notna()
    ].copy()
    if len(closed) < 5:
        return None, [], None

    x = build_features(closed)
    y = closed[target].astype(float)
    errors = []

    if len(closed) >= 7:
        for idx in range(len(x)):
            model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=1)
            train_x = x.drop(x.index[idx])
            train_y = y.drop(y.index[idx])
            model.fit(train_x, train_y)
            pred = float(model.predict(x.iloc[[idx]])[0])
            errors.append(abs(pred - y.iloc[idx]))

    model = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=1)
    model.fit(x, y)
    mae = float(np.mean(errors)) if errors else None
    return model, list(x.columns), mae


def predict_rows(
    model: Optional[RandomForestRegressor],
    feature_columns: list[str],
    rows: pd.DataFrame,
    output_col: str,
) -> pd.DataFrame:
    result = rows.copy()
    if result.empty or model is None:
        result[output_col] = np.nan
        return result

    x = build_features(result).reindex(columns=feature_columns, fill_value=0)
    result[output_col] = model.predict(x)
    return result


def run_dashboard(cfg: Config) -> None:
    df_c = fetch_csv(cfg.id_pub, cfg.gid_consumo, "Consumo")
    df_p = fetch_csv(cfg.id_pub, cfg.gid_precios, "Cafeterias")
    df_g = fetch_csv(cfg.id_pub, cfg.gid_gramajes, "Gramajes")

    if df_c is None or df_g is None:
        print("🛑 Error crítico al descargar datos base.")
        return

    gramos_map, columnas_gramajes = build_gram_map(df_g)
    precios_por_anio, precio_promedio_general = build_price_context(df_p)
    all_rows, tazas_cols = prepare_rows(df_c, gramos_map, columnas_gramajes)
    cycles = all_rows[~all_rows["es_molido_anual"]].copy()

    if cycles.empty:
        print("🛑 No hay bolsas de café con gramos registrados para analizar.")
        return

    closed = cycles[cycles["esta_cerrada"] & cycles["ciclo_valido"]].copy()
    active_all = cycles[~cycles["esta_cerrada"]].copy()

    total_tazas_all = float(all_rows["tazas_total"].sum())
    total_consumido_all_g = float(all_rows["g_consumidos_est"].sum())
    cups_by_type = all_rows[tazas_cols].sum()

    total_comprado_g = float(cycles["gramos"].sum())
    total_consumido_g = float(cycles["g_consumidos_est"].sum())
    inv_activo_estimado = max(
        0.0, float(active_all["gramos"].sum() - active_all["g_consumidos_est"].sum())
    )
    merma_historica = max(0.0, total_comprado_g - total_consumido_g - inv_activo_estimado)

    paid = cycles[cycles["costo"] > 0].copy()
    gasto_real = float(paid["costo"].sum())
    bolsas_pagadas = int(len(paid))
    costo_promedio_bolsa_pagada = gasto_real / bolsas_pagadas if bolsas_pagadas else 0
    costo_promedio_250g_pagado = (
        gasto_real / paid["gramos"].sum() * 250 if not paid.empty and paid["gramos"].sum() else 0
    )
    costo_promedio_taza_real = (
        gasto_real / cycles["tazas_total"].sum() if cycles["tazas_total"].sum() else 0
    )

    daily = distribute_cycles_by_day(cycles)
    if not daily.empty:
        daily["anio_mes"] = daily["fecha"].dt.to_period("M")
        daily["mes"] = daily["fecha"].dt.month
        daily["temporada"] = daily["mes"].apply(season_name)
        monthly = daily.groupby("anio_mes").agg(
            tazas=("tazas", "sum"),
            gramos=("gramos", "sum"),
            dias=("fecha", "nunique"),
        )
        monthly["tazas_dia"] = monthly["tazas"] / monthly["dias"]
        seasonal = daily.groupby("temporada").agg(
            tazas=("tazas", "sum"),
            gramos=("gramos", "sum"),
            dias=("fecha", "nunique"),
        )
        seasonal["tazas_dia"] = seasonal["tazas"] / seasonal["dias"]
    else:
        monthly = pd.DataFrame()
        seasonal = pd.DataFrame()

    duration_model, duration_features, duration_mae = train_regression_model(cycles, "dias_ciclo")
    cups_model, cups_features, cups_mae = train_regression_model(cycles, "tazas_total")

    active_predictions = predict_rows(
        duration_model, duration_features, active_all[active_all["ciclo_valido"]], "dias_predichos_ml"
    )
    if not active_predictions.empty:
        active_predictions["inventario_restante_g"] = np.maximum(
            0, active_predictions["gramos"] - active_predictions["g_consumidos_est"]
        )
        active_predictions["ritmo_observado_g_dia"] = np.where(
            active_predictions["dias_ciclo"] > 0,
            active_predictions["g_consumidos_est"] / active_predictions["dias_ciclo"],
            0,
        )
        active_predictions["dias_totales_por_ritmo"] = np.where(
            active_predictions["ritmo_observado_g_dia"] > 0,
            active_predictions["dias_ciclo"]
            + active_predictions["inventario_restante_g"]
            / active_predictions["ritmo_observado_g_dia"],
            active_predictions["dias_predichos_ml"],
        )
        active_predictions["dias_predichos_ml"] = np.maximum(
            active_predictions["dias_predichos_ml"],
            active_predictions["dias_totales_por_ritmo"],
        )
        active_predictions["dias_restantes_ml"] = (
            active_predictions["dias_predichos_ml"] - active_predictions["dias_ciclo"]
        )
        active_predictions["fecha_cierre_estimada_ml"] = active_predictions["fecha_apertura"] + active_predictions[
            "dias_predichos_ml"
        ].round().astype(int).apply(lambda days: timedelta(days=max(days - 1, 0)))

    last_five = closed.sort_values("fecha_cierre", ascending=False).head(5).sort_values("fecha_cierre")
    last_five = predict_rows(cups_model, cups_features, last_five, "tazas_proyectadas_ml")
    last_five = predict_rows(duration_model, duration_features, last_five, "dias_proyectados_ml")

    if not monthly.empty:
        top_month_label = str(monthly.sort_values("tazas_dia", ascending=False).index[0])
        top_month = monthly.loc[monthly.sort_values("tazas_dia", ascending=False).index[0]]
    else:
        top_month_label = "Sin datos"
        top_month = None

    if not seasonal.empty:
        top_season_label = seasonal.sort_values("tazas_dia", ascending=False).index[0]
        low_season_label = seasonal.sort_values("tazas_dia", ascending=True).index[0]
        top_season = seasonal.loc[top_season_label]
        low_season = seasonal.loc[low_season_label]
    else:
        top_season_label = "Sin datos"
        low_season_label = "Sin datos"
        top_season = None
        low_season = None

    close_weekdays = closed[closed["fecha_cierre"].notna()]["dia_cierre_nombre"].value_counts()
    close_day_types = closed[closed["fecha_cierre"].notna()]["tipo_dia_cierre"].value_counts()
    most_common_close_day = close_weekdays.index[0] if not close_weekdays.empty else "Sin datos"

    ranking = cycles[cycles["cafe"] != "Sin Nombre"]["cafe"].value_counts().head(5)

    ahorro_total = 0.0
    if not daily.empty:
        daily["anio"] = daily["fecha"].dt.year
        daily["precio_mercado"] = daily["anio"].map(precios_por_anio).fillna(
            precio_promedio_general
        )
        ahorro_total = float((daily["tazas"] * daily["precio_mercado"]).sum())
    ganancia_post_roi = max(0.0, ahorro_total - cfg.costo_cafetera)

    datos_exportacion = {
        "hora_ejecucion": datetime.now().strftime("%H:%M:%S"),
        "bolsas_analizadas": int(len(cycles)),
        "bolsas_cerradas": int(len(closed)),
        "bolsas_activas": int(len(active_all)),
        "total_tazas_consumidas_incl_molido": int(total_tazas_all),
        "total_consumido_g_incl_molido": round(total_consumido_all_g, 2),
        "inventario_activo_estimado_g": round(inv_activo_estimado, 2),
        "gasto_real_cafe_mxn": round(gasto_real, 2),
        "costo_promedio_250g_pagado_mxn": round(costo_promedio_250g_pagado, 2),
        "costo_promedio_taza_real_mxn": round(costo_promedio_taza_real, 2),
        "mes_mayor_consumo": top_month_label,
        "temporada_mayor_consumo": top_season_label,
        "temporada_menor_consumo": low_season_label,
        "dia_mas_comun_cierre_bolsa": most_common_close_day,
        "ml_error_medio_dias": round(duration_mae, 2) if duration_mae is not None else "Sin validacion",
        "ml_error_medio_tazas": round(cups_mae, 2) if cups_mae is not None else "Sin validacion",
    }

    print("\n" + "═" * 72)
    print("☕ COFFEE CONSUMPTION ANALYTICS · WIP EVOLUCION")
    print("═" * 72)

    print("\n🧠 LECTURA EJECUTIVA")
    print(f" • Bolsas analizadas:        {len(cycles)} ({len(closed)} cerradas, {len(active_all)} activas)")
    print(f" • Tazas consumidas:         {int(total_tazas_all)}")
    print(f" • Total consumido:          {total_consumido_all_g:,.0f} g")
    print(f" • Inventario activo est.:   {inv_activo_estimado:,.0f} g")
    print(f" • Merma historica est.:     {merma_historica:,.0f} g")
    print(f" • Gasto real en cafe:       {money(gasto_real)}")
    print(f" • Promedio bolsa pagada:    {money(costo_promedio_bolsa_pagada)}")
    print(f" • Promedio pagado/250g:     {money(costo_promedio_250g_pagado)}")
    print(f" • Costo real por taza:      {money(costo_promedio_taza_real)}")

    print("\n🤖 MODELO ML · POR QUE RANDOM FOREST")
    print(" • Lo uso porque tus datos mezclan numeros, meses y nombres de cafe; captura relaciones no lineales sin pedir mucha preparacion.")
    print(" • Error medio estimado = prueba leave-one-out: entreno con todas menos una bolsa y comparo contra la bolsa que deje fuera.")
    if duration_model is None:
        print(" • Faltan ciclos cerrados suficientes para entrenar duracion con confianza.")
    else:
        print(f" • Error duracion:           ±{duration_mae:.1f} dias por bolsa")
    if cups_model is None:
        print(" • Faltan ciclos cerrados suficientes para proyectar tazas con confianza.")
    else:
        print(f" • Error tazas:              ±{cups_mae:.1f} tazas por bolsa")
    if not active_predictions.empty:
        print(" • ETA para bolsas activas con fecha:")
        for _, row in active_predictions.sort_values("dias_restantes_ml").iterrows():
            print(
                f"   - {row['cafe']}: {row['fecha_cierre_estimada_ml'].date()} "
                f"({row['dias_restantes_ml']:.1f} dias restantes, {row['inventario_restante_g']:.0f} g)"
            )

    print("\n🏆 RENDIMIENTO · ULTIMAS 5 BOLSAS CERRADAS")
    if last_five.empty:
        print(" • No hay ciclos cerrados suficientes.")
    else:
        for _, row in last_five.iterrows():
            cierre = row["fecha_cierre"].date() if pd.notna(row["fecha_cierre"]) else "sin cierre"
            tazas_ml = row["tazas_proyectadas_ml"]
            dias_ml = row["dias_proyectados_ml"]
            print(
                f" • {cierre} | {row['cafe']}: "
                f"{int(row['tazas_total'])} tazas reales vs {tazas_ml:.1f} ML | "
                f"{int(row['dias_ciclo'])} dias reales vs {dias_ml:.1f} ML"
            )

    print("\n🗓️ CONSUMO MENSUAL Y ESTACIONAL")
    if top_month is None:
        print(" • No hay suficientes ciclos validos para distribuir consumo por mes.")
    else:
        print(
            f" • Mes mas intenso:          {top_month_label} "
            f"({top_month['tazas_dia']:.2f} tazas/dia, {top_month['gramos']:.0f} g)"
        )
        if top_season is not None and low_season is not None:
            print(
                f" • Temporada mas intensa:    {top_season_label} "
                f"({top_season['tazas_dia']:.2f} tazas/dia)"
            )
            print(
                f" • Temporada mas baja:       {low_season_label} "
                f"({low_season['tazas_dia']:.2f} tazas/dia)"
            )
        print(" • Ultimos 6 meses:")
        for period, row in monthly.tail(6).iterrows():
            print(
                f"   - {period}: {row['tazas']:.0f} tazas | "
                f"{row['gramos']:.0f} g | {row['tazas_dia']:.2f} tazas/dia"
            )

    print("\n📅 CIERRE DE BOLSAS")
    if close_weekdays.empty:
        print(" • No hay cierres suficientes para analizar dias.")
    else:
        print(f" • Dia donde mas terminas bolsas: {most_common_close_day}")
        entre = int(close_day_types.get("entre semana", 0))
        finde = int(close_day_types.get("fin de semana", 0))
        print(f" • Cierres entre semana:     {entre}")
        print(f" • Cierres fin de semana:    {finde}")

    print("\n💰 ROI Y VALOR")
    print(f" • Ahorro bruto estimado:    {money(ahorro_total)}")
    if ahorro_total >= cfg.costo_cafetera:
        print(f" ✅ Ganancia post-ROI:        {money(ganancia_post_roi)}")
    else:
        print(f" ⏳ Faltante para ROI:        {money(cfg.costo_cafetera - ahorro_total)}")

    print("\n☕ DESGLOSE DE TAZAS · ORDEN SHEETS")
    for taza, cantidad in cups_by_type.items():
        if cantidad > 0:
            print(f" • {taza:<12}: {int(cantidad):>4} tazas")

    print("\n🏷️ TOP 5 CAFES MAS COMPRADOS")
    if ranking.empty:
        print(" • Aun no hay compras registradas para ranking.")
    else:
        for cafe, veces in ranking.items():
            print(f" • {cafe}: {veces} compras")

    print("═" * 72)

    if cfg.guardar_snapshot:
        guardar_historico_csv(datos_exportacion, cfg.ruta_csv)
    else:
        print("🧪 Modo WIP: snapshot no guardado.")


if __name__ == "__main__":
    ID_PUB = "2PACX-1vTdOWbzsFlVlSTBguq9_cjLXzvDO-uUqnIY4zaex27J4biHRk2t5u7aCHShyaCVmKhtJ12XLDQ8Nu2n"
    config = Config(
        id_pub=ID_PUB,
        gid_consumo="0",
        gid_precios="49728846",
        gid_gramajes="1827085190",
        costo_cafetera=12239,
    )
    run_dashboard(config)
