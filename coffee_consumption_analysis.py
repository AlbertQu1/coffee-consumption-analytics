import os
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional
import json
import numpy_financial as npf
from dotenv import load_dotenv
import gspread
import requests


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV,
    Lasso, LassoCV,
    Ridge, RidgeCV,
)

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

def connect_google_sheets():
    credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    sheet_name = os.getenv("GOOGLE_SHEET_NAME")

    try:
        gc = gspread.service_account()
        return gc
    except Exception as exc:
        print(f"❌ Error al conectar con Google Sheets: {exc}")
        return None

@dataclass(frozen=True)
class Config:
    id_pub: str
    gid_consumo: str
    gid_precios: str
    gid_gramajes: str
    gid_merma: str
    costo_cafetera: float
    fecha_compra: str = "30/05/2025"
    ruta_csv: str = (
        r"C:\Users\alber\OneDrive\Desktop\Tripleten\Cafetera\Cafe"
        r"\historial_cafe.csv"
    )
    guardar_snapshot: bool = True


def normalize_key(value: object) -> str:
    text = str(value).strip()
    text = "".join(ch for ch in text if ch.isprintable())
    text = text.lower()
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


def fetch_cetes_rate() -> float:
    token= os.getenv("BANXICO_TOKEN")
    if not token:
        print("⚠️ No se encontró el token de Banxico en las variables de entorno. — usando tasa por defecto 10%")
        return 0.10
    try:
        url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43936/datos/oportuno"
        headers = {"Bmx-Token": token}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        dato = response.json()["bmx"]["series"][0]["datos"][0]["dato"]
        return round(float(dato) / 100,6)
    except Exception as exc:
        print(f"⚠️ Error al obtener la tasa de Cetes: {exc} — usando tasa por defecto 10%")
        return 0.10
        
def calcular_van_tir(costo_inicial: float, ahorro_mensual: float, meses: int, tasa_anual: float) -> dict:
    tasa_mensual = tasa_anual / 12
    flujos= [-costo_inicial] + [ahorro_mensual] * meses
    van= npf.npv(tasa_mensual, flujos)
    tir_mensual= npf.irr(flujos)
    tir_anual= (1+ tir_mensual)**12 - 1 if tir_mensual is not None else None
    return {
        "van": round(van,2),
        "tir_anual": round(tir_anual * 100,2) if tir_anual else None,
        "tasa_descuento": round(tasa_anual *100, 2)
    }

def build_gram_map(df_g: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    tipo_col = find_column(df_g, ["tipo de bebida", "bebida", "tipo"])
    cafe_g_col = find_column(df_g, ["cafe (g)", "cafe g", "gramos cafe"])
    if tipo_col is None or cafe_g_col is None:
        raise ValueError("La hoja de gramajes necesita tipo de bebida y cafe (g).")

    tipos = df_g[tipo_col].astype(str).map(normalize_key)
    gramos = pd.to_numeric(df_g[cafe_g_col], errors="coerce").fillna(0)
    gramos_map = dict(zip(tipos, gramos))
    return gramos_map, list(gramos_map.keys())

def build_merma_map(df_m: pd.DataFrame) -> dict[tuple[str, str], float]:
    fecha_col = find_column(df_m, ["fecha"])
    cafe_col = find_column(df_m, ["cafe"])
    merma_col = find_column(df_m, ["perdida"])
    if fecha_col is None or cafe_col is None:
        raise ValueError("La fecha de merma es incorrecta o el cafe ingresado es inválido.")
    
    cafe= df_m[cafe_col].astype(str).map(normalize_key)
    fecha= pd.to_datetime(df_m[fecha_col], dayfirst=True, errors="coerce")
    merma= pd.to_numeric(df_m[merma_col], errors="coerce").fillna(0)
    merma_map= {(cafe.iloc[i], fecha.iloc[i]): merma.iloc[i] for i in range(len(cafe))}
    return merma_map


def prepare_rows(
    df_c: pd.DataFrame,
    gramos_map: dict[str, float],
    columnas_gramajes: list[str],
    merma_map: dict =None,
) -> tuple[pd.DataFrame, list[str]]:
    col_fecha = find_column(df_c, ["fecha", "date"])
    if col_fecha is None:
        raise ValueError("La hoja de consumo necesita una columna fecha.")

    costo_col = find_column(df_c, ["costo", "precio pagado"])
    gramos_col = find_column(df_c, ["gramos", "gramaje", "peso"])
    cierre_col = find_column(df_c, ["cierre", "fecha cierre"])
    cafeteria_col = find_column(df_c, [
        "cafeteria", "cafe", "origen", "nombre", "marca",
        "tostador", "proveedor", "descripcion", "tipo cafe",
        "nombre cafe", "nombre del cafe", "bolsa",
    ])
    if cafeteria_col is None:
        cols = ", ".join(sorted(df_c.columns.tolist()))
        print(f"⚠️  Columna de cafetería no encontrada. Columnas disponibles: {cols}")
        print("   Agrega el nombre exacto a la lista de candidatos en el código.")
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

    if merma_map is None:
        merma_map = {}
    df["merma_manual"] = [
        merma_map.get((normalize_key(c), f), 0.0)
        for c, f in zip(df["cafe"], df["fecha_apertura"])
    ]
    df["gramos_efectivos"] = (df["gramos"] - df["merma_manual"]).clip(lower=0)

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
    df["tazas_por_250g"] = np.where(df["gramos_efectivos"] > 0, df["tazas_total"] / df["gramos_efectivos"] * 250, 0)
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


MODEL_VERSION = "auto_v1"
FEATURES_TAZAS = ["gramos", "mes_apertura"]
FEATURES_DURACION = ["gramos", "mes_apertura"]

# P4: variables solo disponibles DESPUÉS de cerrar la bolsa — nunca deben ser features
VARIABLES_POST_CIERRE: frozenset[str] = frozenset({
    "dias_ciclo", "tazas_total", "g_consumidos_est", "tazas_por_dia", "gramos_por_dia"
})
COEF_THRESHOLD = 0.01  # umbral para considerar un coeficiente "activo"


def build_features(rows: pd.DataFrame, features: list[str], use_cafe_modelo: bool = True) -> pd.DataFrame:
    if use_cafe_modelo:
        return pd.get_dummies(rows[features + ["cafe_modelo"]], columns=["cafe_modelo"], dummy_na=False)
    return rows[features].copy()


def _loo_mae(model_class, params: dict, x: pd.DataFrame, y: pd.Series, scale=False) -> Optional[float]:
    if len(x) < 5:
        return None
    errors = []
    for idx in range(len(x)):
        m = model_class(**params)
        m.fit(x.drop(x.index[idx]), y.drop(y.index[idx]))
        errors.append(abs(float(m.predict(x.iloc[[idx]])[0]) - y.iloc[idx]))
    return float(np.mean(errors))


def _confianza_nivel(n: int) -> str:
    if n < 20:
        return "Baja (<20 bolsas)"
    if n <= 50:
        return "Media (20-50 bolsas)"
    return "Alta (>50 bolsas)"


def train_best_model(
    cycles: pd.DataFrame, target: str, features: list[str]
) -> tuple[Optional[object], list[str], Optional[float], str, dict[str, float], dict[str, float], dict]:
    _empty_meta: dict = {"cafe_modelo_ayudo": None, "mae_sin_cafe": None, "mae_con_cafe": None}
    closed = cycles[
        cycles["esta_cerrada"] & cycles["ciclo_valido"] & cycles[target].notna()
    ].copy()
    if len(closed) < 3:
        return None, [], None, "ninguno", {}, {}, _empty_meta

    x_con = build_features(closed, features, use_cafe_modelo=True)
    x_sin = build_features(closed, features, use_cafe_modelo=False)
    y = closed[target].astype(float)
    cv_folds = min(len(closed), 5)

    # Tune hyperparameters on the full feature set (con cafe_modelo)
    lasso_cv = LassoCV(cv=cv_folds, random_state=42, max_iter=10000)
    lasso_cv.fit(x_con, y)
    ridge_cv = RidgeCV(cv=cv_folds)
    ridge_cv.fit(x_con, y)
    en_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
        cv=cv_folds, random_state=42, max_iter=10000,
    )
    en_cv.fit(x_con, y)

    candidates = {
        "Lasso": (Lasso, {"alpha": lasso_cv.alpha_, "max_iter": 10000}),
        "Ridge": (Ridge, {"alpha": ridge_cv.alpha_}),
        "ElasticNet": (ElasticNet, {"alpha": en_cv.alpha_, "l1_ratio": en_cv.l1_ratio_, "max_iter": 10000}),
        "KNN": (KNeighborsRegressor, {"n_neighbors": 3})
    }

    # Benchmark con cafe_modelo
    benchmark: dict[str, float] = {}
    for name, (cls, params) in candidates.items():
        mae_val = _loo_mae(cls, params, x_con, y)
        if mae_val is not None:
            benchmark[name] = mae_val

    winner_name = min(benchmark, key=lambda k: benchmark[k]) if benchmark else "ElasticNet"
    cls_winner, params_winner = candidates[winner_name]
    mae_con = benchmark.get(winner_name)

    # P1: probar si cafe_modelo realmente ayuda comparando MAE del ganador sin él
    mae_sin = _loo_mae(cls_winner, params_winner, x_sin, y)

    if mae_sin is not None and mae_con is not None and mae_sin <= mae_con:
        # cafe_modelo no aporta → recalcular benchmark completo sin él y elegir nuevo ganador
        benchmark = {}
        for name, (cls, params) in candidates.items():
            mae_val = _loo_mae(cls, params, x_sin, y)
            if mae_val is not None:
                benchmark[name] = mae_val
        winner_name = min(benchmark, key=lambda k: benchmark[k]) if benchmark else winner_name
        cls_winner, params_winner = candidates[winner_name]
        x = x_sin
        cafe_modelo_ayudo = False
    else:
        x = x_con
        cafe_modelo_ayudo = True if mae_con is not None else None

    model = cls_winner(**params_winner)
    model.fit(x, y)

    coefs = {f: float(c) for f, c in zip(x.columns, model.coef_)}
    coefs_sorted = dict(sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    meta = {
        "cafe_modelo_ayudo": cafe_modelo_ayudo,
        "mae_sin_cafe": mae_sin,
        "mae_con_cafe": mae_con,
    }

    return model, list(x.columns), benchmark.get(winner_name), winner_name, coefs_sorted, benchmark, meta


def predict_rows(
    model: Optional[object],
    feature_columns: list[str],
    rows: pd.DataFrame,
    output_col: str,
    base_features: list[str],
) -> pd.DataFrame:
    result = rows.copy()
    if result.empty or model is None:
        result[output_col] = np.nan
        return result

    x = build_features(result, base_features).reindex(columns=feature_columns, fill_value=0)
    result[output_col] = model.predict(x)
    return result


def run_dashboard(cfg: Config) -> None:
    df_c = fetch_csv(cfg.id_pub, cfg.gid_consumo, "Consumo")
    df_p = fetch_csv(cfg.id_pub, cfg.gid_precios, "Cafeterias")
    df_g = fetch_csv(cfg.id_pub, cfg.gid_gramajes, "Gramajes")
    df_m = fetch_csv(cfg.id_pub, cfg.gid_merma, "Merma")

    if df_c is None or df_g is None:
        print("🛑 Error crítico al descargar datos base.")
        return

    gramos_map, columnas_gramajes = build_gram_map(df_g)
    precios_por_anio, precio_promedio_general = build_price_context(df_p)
    merma_map = build_merma_map(df_m)
    all_rows, tazas_cols = prepare_rows(df_c, gramos_map, columnas_gramajes, merma_map)
    cycles = all_rows[~all_rows["es_molido_anual"]].copy()

    if cycles.empty:
        print("🛑 No hay bolsas de café con gramos registrados para analizar.")
        return

    def reportar_fechas_invalidas(df: pd.DataFrame) -> None:
        mascara = df["fecha_cierre"] < df["fecha_apertura"]
        errores = df[mascara]
    
        for _, fila in errores.iterrows():
            print(
                f"⚠️  {fila['cafe']}: "
                f"cierre ({fila['fecha_cierre'].strftime('%d/%m/%Y')}) "
                f"es anterior a la apertura ({fila['fecha_apertura'].strftime('%d/%m/%Y')})"
            )

    reportar_fechas_invalidas(cycles)

    conteo_cafe = cycles["cafe"].value_counts()
    cycles["cafe_modelo"] = np.where(
        cycles["cafe"].map(conteo_cafe) >= 3,
        cycles["cafe"],
        "OTROS",
    )

    closed = cycles[cycles["esta_cerrada"] & cycles["ciclo_valido"]].copy()
    active_all = cycles[~cycles["esta_cerrada"]].copy()
    en_maquina = active_all[active_all["tazas_total"] > 0].copy()
    en_espera = active_all[active_all["tazas_total"] == 0].copy()

    closed["tazas_por_dia"] = np.where(
        closed["dias_ciclo"] > 0, closed["tazas_total"] / closed["dias_ciclo"], np.nan
    )
    closed["gramos_por_dia"] = np.where(
        closed["dias_ciclo"] > 0, closed["g_consumidos_est"] / closed["dias_ciclo"], np.nan
    )
    active_all["tazas_por_dia"] = np.where(
        active_all["dias_ciclo"] > 0, active_all["tazas_total"] / active_all["dias_ciclo"], np.nan
    )
    active_all["gramos_por_dia"] = np.where(
        active_all["dias_ciclo"] > 0, active_all["g_consumidos_est"] / active_all["dias_ciclo"], np.nan
    )

    # P2: MAE LOO para duración estimada por ritmo histórico (tazas/dia promedio)
    ritmo_mae_dur: Optional[float] = None
    if len(closed) >= 5:
        c_r = closed[closed["dias_ciclo"] > 0].copy().reset_index(drop=True)
        ritmo_errors = []
        for i in range(len(c_r)):
            subset = c_r.drop(i)
            ritmo = float(subset["tazas_por_dia"].mean()) if not subset.empty else 0.0
            if ritmo > 0:
                dias_pred = float(c_r.loc[i, "tazas_total"]) / ritmo
                dias_real = float(c_r.loc[i, "dias_ciclo"])
                ritmo_errors.append(abs(dias_pred - dias_real))
        if ritmo_errors:
            ritmo_mae_dur = float(np.mean(ritmo_errors))

    total_tazas_all = float(all_rows["tazas_total"].sum())
    total_consumido_all_g = float(all_rows["g_consumidos_est"].sum())
    cups_by_type = all_rows[tazas_cols].sum()

    inv_activo_estimado = max(
        0.0, float(active_all["gramos"].sum() - active_all["g_consumidos_est"].sum())
    )
    merma_historica = float(
        (closed["gramos"] - closed["g_consumidos_est"]).clip(lower=0).sum()
    ) if not closed.empty else 0.0

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
    costo_por_taza_anio: dict[int, float] = {}
    if not paid.empty:
        for anio_g, grupo in paid.groupby("anio_apertura"):
            tazas_anio = grupo["tazas_total"].sum()
            if tazas_anio > 0:
                costo_por_taza_anio[int(anio_g)] = float(grupo["costo"].sum() / tazas_anio)

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

    (duration_model, duration_features, duration_mae,
     duration_model_name, duration_coefs, duration_bench, duration_meta) = train_best_model(
        cycles, "dias_ciclo", FEATURES_DURACION
    )
    (cups_model, cups_features, cups_mae,
     cups_model_name, cups_coefs, cups_bench, cups_meta) = train_best_model(
        cycles, "tazas_total", FEATURES_TAZAS
    )

    active_predictions = predict_rows(
        duration_model, duration_features,
        active_all[active_all["ciclo_valido"]], "dias_predichos_ml",
        FEATURES_DURACION,
    )
    if not active_predictions.empty and duration_model is None and not closed.empty:
        media_ciclo = float(closed["dias_ciclo"].mean())
        active_predictions["dias_predichos_ml"] = media_ciclo
    if not active_predictions.empty:
        active_predictions["dias_ml_puro"] = active_predictions["dias_predichos_ml"]
        active_predictions["inventario_restante_g"] = np.maximum(
            0, active_predictions["gramos"] - active_predictions["g_consumidos_est"]
        )
        active_predictions["ritmo_observado_g_dia"] = np.where(
            active_predictions["dias_ciclo"] > 0,
            active_predictions["g_consumidos_est"] / active_predictions["dias_ciclo"],
            0,
        )
        active_predictions["dias_por_ritmo"] = np.where(
            active_predictions["ritmo_observado_g_dia"] > 0,
            active_predictions["dias_ciclo"]
            + active_predictions["inventario_restante_g"]
            / active_predictions["ritmo_observado_g_dia"],
            active_predictions["dias_ml_puro"],
        )
        active_predictions["dias_predichos_ml"] = np.maximum(
            active_predictions["dias_ml_puro"],
            active_predictions["dias_por_ritmo"],
        )
        active_predictions["dias_restantes_ml"] = (
            active_predictions["dias_predichos_ml"] - active_predictions["dias_ciclo"]
        )
        active_predictions["fecha_cierre_estimada_ml"] = active_predictions["fecha_apertura"] + active_predictions[
            "dias_predichos_ml"
        ].round().astype(int).apply(lambda days: timedelta(days=max(days - 1, 0)))

    last_five = closed.sort_values("fecha_cierre", ascending=False).head(5).sort_values("fecha_cierre")
    last_five = predict_rows(cups_model, cups_features, last_five, "tazas_proyectadas_ml", FEATURES_TAZAS)
    last_five = predict_rows(duration_model, duration_features, last_five, "dias_proyectados_ml", FEATURES_DURACION)

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

    vc_ranking = cycles[cycles["cafe"] != "Sin Nombre"]["cafe"].value_counts()
    ranking = vc_ranking[vc_ranking >= 2].head(5)

    ahorro_total = 0.0
    ahorro_por_anio: dict[int, float] = {}
    if not daily.empty:
        daily["anio"] = daily["fecha"].dt.year
        daily["precio_mercado"] = daily["anio"].map(precios_por_anio).fillna(
            precio_promedio_general
        )
        ahorro_total = float((daily["tazas"] * daily["precio_mercado"]).sum())
        ahorro_por_anio = (
            daily.groupby("anio")
            .apply(lambda d: float((d["tazas"] * d["precio_mercado"]).sum()), include_groups=False)
            .round(2)
            .to_dict()
        )

    gasto_real_por_anio: dict[int, float] = {}
    if not paid.empty:
        gasto_real_por_anio = {
            int(k): round(float(v), 2)
            for k, v in paid.groupby("anio_apertura")["costo"].sum().items()
        }

    all_anios = sorted(set(list(ahorro_por_anio.keys()) + list(gasto_real_por_anio.keys())))
    ahorro_neto_por_anio: dict[int, float] = {
        a: round(ahorro_por_anio.get(a, 0.0) - gasto_real_por_anio.get(a, 0.0), 2)
        for a in all_anios
    }
    ahorro_neto_total = ahorro_total - gasto_real
    ganancia_post_roi = max(0.0, ahorro_neto_total - cfg.costo_cafetera)

    fecha_compra_dt = datetime.strptime(cfg.fecha_compra, "%d/%m/%Y")
    dias_desde_compra = (datetime.now() - fecha_compra_dt).days
    meses_desde_compra = round(dias_desde_compra / 30.44, 1)
    ahorro_por_mes = ahorro_neto_total / meses_desde_compra if meses_desde_compra > 0 else 0.0
    tasa_cetes =fetch_cetes_rate()
    van_tir = calcular_van_tir(cfg.costo_cafetera, ahorro_por_mes, int(meses_desde_compra), tasa_cetes)

    datos_exportacion = {
        "hora_ejecucion": datetime.now().strftime("%H:%M:%S"),
        "fecha_compra_cafetera": cfg.fecha_compra,
        "dias_desde_compra": dias_desde_compra,
        "meses_desde_compra": meses_desde_compra,
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
        "ahorro_bruto_mxn": round(ahorro_total, 2),
        "ahorro_por_mes_mxn": round(ahorro_por_mes, 2),
        "ganancia_post_roi_mxn": round(ganancia_post_roi, 2),
        "version_modelo": MODEL_VERSION,
        "modelo_duracion": duration_model_name,
        "modelo_tazas": cups_model_name,
        "error_duracion": round(duration_mae, 2) if duration_mae is not None else None,
        "error_tazas": round(cups_mae, 2) if cups_mae is not None else None,
        "confianza_modelo": ("Baja" if len(closed) < 20 else "Media" if len(closed) <= 50 else "Alta"),
        "cafe_modelo_ayudo_duracion": duration_meta.get("cafe_modelo_ayudo"),
        "cafe_modelo_ayudo_tazas": cups_meta.get("cafe_modelo_ayudo"),
        "ritmo_mae_dias": round(ritmo_mae_dur, 2) if ritmo_mae_dur is not None else None,
    }

    print("\n" + "═" * 72)
    print("☕ COFFEE CONSUMPTION ANALYTICS · WIP EVOLUCION")
    print("═" * 72)

    print("\n🧠 LECTURA EJECUTIVA")
    print(f" • Bolsas analizadas:        {len(cycles)} ({len(closed)} cerradas, {len(en_maquina)} en maquina, {len(en_espera)} en espera)")
    print(f" • Tazas consumidas:         {int(total_tazas_all)}")
    print(f" • Total consumido:          {total_consumido_all_g:,.0f} g")
    print(f" • Inventario activo est.:   {inv_activo_estimado:,.0f} g")
    print(f" • Merma historica est.:     {merma_historica:,.0f} g")
    print(f" • Gasto real en cafe:       {money(gasto_real)}")
    print(f" • Promedio bolsa pagada:    {money(costo_promedio_bolsa_pagada)}")
    print(f" • Promedio pagado/250g:     {money(costo_promedio_250g_pagado)}")
    print(f" • Costo real por taza:      {money(costo_promedio_taza_real)} (historico)")
    for anio_t, costo_t in sorted(costo_por_taza_anio.items()):
        print(f"   - {anio_t}: {money(costo_t)}")

    print("\n🤖 MODELO ML")
    n_cerradas_ml = int(len(closed))
    confianza = _confianza_nivel(n_cerradas_ml)
    print(f"  Estabilidad: {n_cerradas_ml} bolsas cerradas  |  Confianza: {confianza}")

    def _bench_str(bench: dict[str, float]) -> str:
        if not bench:
            return "sin validacion LOO (n<5)"
        return "  ".join(f"{k}: ±{v:.1f}" for k, v in bench.items())

    def _cafe_modelo_str(meta: dict) -> str:
        ayudo = meta.get("cafe_modelo_ayudo")
        mc = meta.get("mae_con_cafe")
        ms = meta.get("mae_sin_cafe")
        if ayudo is None:
            return "cafe_modelo: sin datos LOO (n<5)"
        con_s = f"±{mc:.1f}" if mc is not None else "n/a"
        sin_s = f"±{ms:.1f}" if ms is not None else "n/a"
        if ayudo:
            return f"cafe_modelo: incluido  (con {con_s}  vs sin {sin_s} — ayuda)"
        return f"cafe_modelo: excluido  (sin {sin_s}  vs con {con_s} — sin es mejor)"

    if duration_model is None:
        if n_cerradas_ml > 0:
            print(f" • Duracion: usando promedio historico ({n_cerradas_ml} bolsa(s), min. 3 para ML)")
        else:
            print(" • Duracion: sin bolsas cerradas para referencia.")
    else:
        print(f" • Duracion [{duration_model_name}]  MAE: ±{duration_mae:.1f} dias")
        print(f"   Benchmark: {_bench_str(duration_bench)}")
        # P2: comparar vs duración estimada por ritmo histórico
        if ritmo_mae_dur is not None and duration_mae is not None:
            mejor = "ML mejor" if duration_mae <= ritmo_mae_dur else "ritmo mejor"
            print(f"   vs. ritmo historico: ±{ritmo_mae_dur:.1f} dias  ({mejor})")
        elif ritmo_mae_dur is not None:
            print(f"   vs. ritmo historico: ±{ritmo_mae_dur:.1f} dias")
        # P1: cafe_modelo
        print(f"   {_cafe_modelo_str(duration_meta)}")
        # P3: auditoría de variables
        activas_dur = [(f, c) for f, c in duration_coefs.items() if abs(c) >= COEF_THRESHOLD]
        inactivas_dur = [f for f, c in duration_coefs.items() if abs(c) < COEF_THRESHOLD]
        if activas_dur:
            print("   Activas (|coef|>=0.01): " + "  ".join(f"{f}({c:+.2f})" for f, c in activas_dur[:5]))
        if inactivas_dur:
            print("   Inactivas (coef~0):     " + "  ".join(inactivas_dur[:8]))

    if cups_model is None:
        if n_cerradas_ml > 0:
            print(f" • Tazas:    usando promedio historico ({n_cerradas_ml} bolsa(s), min. 3 para ML)")
        else:
            print(" • Tazas: sin bolsas cerradas para referencia.")
    else:
        print(f" • Tazas    [{cups_model_name}]  MAE: ±{cups_mae:.1f} tazas")
        print(f"   Benchmark: {_bench_str(cups_bench)}")
        # P1: cafe_modelo
        print(f"   {_cafe_modelo_str(cups_meta)}")
        # P3: auditoría de variables
        activas_taz = [(f, c) for f, c in cups_coefs.items() if abs(c) >= COEF_THRESHOLD]
        inactivas_taz = [f for f, c in cups_coefs.items() if abs(c) < COEF_THRESHOLD]
        if activas_taz:
            print("   Activas (|coef|>=0.01): " + "  ".join(f"{f}({c:+.2f})" for f, c in activas_taz[:5]))
        if inactivas_taz:
            print("   Inactivas (coef~0):     " + "  ".join(inactivas_taz[:8]))

    # P4: auditoría de leakage
    vars_dur_base = set(FEATURES_DURACION)
    vars_taz_base = set(FEATURES_TAZAS)
    leakage_dur_detected = vars_dur_base & VARIABLES_POST_CIERRE
    leakage_taz_detected = vars_taz_base & VARIABLES_POST_CIERRE
    ok_vars = sorted((vars_dur_base | vars_taz_base) - VARIABLES_POST_CIERRE) + ["cafe_modelo"]
    print("  Auditoria de leakage:")
    print(f"   OK (pre-apertura):       {', '.join(ok_vars)}")
    print(f"   Excluidas (post-cierre): {', '.join(sorted(VARIABLES_POST_CIERRE))}")
    if leakage_dur_detected or leakage_taz_detected:
        if leakage_dur_detected:
            print(f"   ⚠️  Leakage en duracion: {', '.join(sorted(leakage_dur_detected))}")
        if leakage_taz_detected:
            print(f"   ⚠️  Leakage en tazas:    {', '.join(sorted(leakage_taz_detected))}")
    else:
        print("   ✅ Sin leakage detectado en features actuales")

    pred_maquina = active_predictions[active_predictions.index.isin(en_maquina.index)] if not active_predictions.empty else pd.DataFrame()
    if not pred_maquina.empty:
        print(" • ETA bolsa en maquina:")
        for _, row in pred_maquina.sort_values("dias_restantes_ml").iterrows():
            dias_rest = row["dias_restantes_ml"]
            eta = row["fecha_cierre_estimada_ml"].date() if pd.notna(row["fecha_cierre_estimada_ml"]) else "sin ETA"
            inv_g = row["inventario_restante_g"]
            estado = "⚠️ por terminar hoy" if dias_rest <= 0 else f"{dias_rest:.0f} dias restantes"
            dias_ml_p = row.get("dias_ml_puro", float("nan"))
            dias_ritmo = row.get("dias_por_ritmo", float("nan"))
            ml_str = f"ML puro: {dias_ml_p:.0f}d" if pd.notna(dias_ml_p) else ""
            ritmo_str = f"ritmo: {dias_ritmo:.0f}d" if pd.notna(dias_ritmo) else ""
            extras = "  ".join(s for s in [ml_str, ritmo_str] if s)
            tpd = row.get("tazas_por_dia", float("nan"))
            tpd_str = f"  {tpd:.2f} tazas/dia" if pd.notna(tpd) and tpd > 0 else ""
            print(f"   - {row['cafe']}: ETA {eta} ({estado}, {inv_g:.0f} g)")
            if extras:
                print(f"     {extras}{tpd_str}")
            if duration_mae is not None and dias_rest > 0:
                lo = max(0, dias_rest - duration_mae)
                hi = dias_rest + duration_mae
                print(f"     Rango dias restantes: {lo:.0f}–{hi:.0f} dias")
    if not en_espera.empty:
        print(f" • En espera ({len(en_espera)} bolsa(s) sin abrir):")
        for _, row in en_espera.iterrows():
            print(f"   - {row['cafe']} {int(row['gramos'])}g  |  {money(row['costo']) if row['costo'] > 0 else 'regalo'}")

    print("\n🏆 RENDIMIENTO · ULTIMAS 5 BOLSAS CERRADAS")
    if last_five.empty:
        print(" • No hay ciclos cerrados suficientes.")
    else:
        for _, row in last_five.iterrows():
            cierre = row["fecha_cierre"].date() if pd.notna(row["fecha_cierre"]) else "sin cierre"
            tazas_ml = row["tazas_proyectadas_ml"]
            dias_ml = row["dias_proyectados_ml"]
            merma_bolsa = max(0.0, float(row["gramos"] - row["g_consumidos_est"]))
            merma_pct = (merma_bolsa / row["gramos"] * 100) if row["gramos"] > 0 else 0
            merma_str = f"merma ~{merma_bolsa:.0f} g ({merma_pct:.1f}%)" if merma_bolsa > 0.5 else "sin merma"
            tpd = row.get("tazas_por_dia", float("nan"))
            tpd_str = f"  {tpd:.2f} tz/dia" if pd.notna(tpd) and tpd > 0 else ""
            tazas_rango = ""
            dias_rango = ""
            if pd.notna(tazas_ml) and cups_mae is not None:
                tazas_rango = f" [{tazas_ml - cups_mae:.0f}–{tazas_ml + cups_mae:.0f}]"
            if pd.notna(dias_ml) and duration_mae is not None:
                dias_rango = f" [{dias_ml - duration_mae:.0f}–{dias_ml + duration_mae:.0f}]"
            print(
                f" • {cierre} | {row['cafe']}: "
                f"{int(row['tazas_total'])} tazas (ML: {tazas_ml:.1f}{tazas_rango}) | "
                f"{int(row['dias_ciclo'])} dias (ML: {dias_ml:.1f}{dias_rango}) | "
                f"{merma_str}{tpd_str}"
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
    print(f" • Cafetera comprada:        {cfg.fecha_compra} ({dias_desde_compra} dias, {meses_desde_compra} meses)")
    print(f" • Equiv. cafeteria (bruto): {money(ahorro_total)}")
    for anio_roi, ahorro_a in sorted(ahorro_por_anio.items()):
        gasto_a = gasto_real_por_anio.get(anio_roi, 0.0)
        print(f"   - {anio_roi}: {money(ahorro_a)} bruto  |  gasto cafe {money(gasto_a)}")
    print(f" • Gasto real en cafe:       {money(gasto_real)}")
    print(f" • Ahorro neto:              {money(ahorro_neto_total)}")
    for anio_roi, neto_a in sorted(ahorro_neto_por_anio.items()):
        print(f"   - {anio_roi}: {money(neto_a)}")
    print(f" • Ahorro neto/mes:          {money(ahorro_por_mes)}")
    if ahorro_neto_total >= cfg.costo_cafetera:
        print(f" ✅ Ganancia post-ROI:        {money(ganancia_post_roi)}")
    else:
        faltante = cfg.costo_cafetera - ahorro_neto_total
        meses_para_roi = faltante / ahorro_por_mes if ahorro_por_mes > 0 else float("inf")
        print(f" ⏳ Faltante para ROI:        {money(faltante)} (~{meses_para_roi:.1f} meses)")
    print(f"\n📊 VAN / TIR  (CETES {van_tir['tasa_descuento']}%)")
    print(f" • VAN:  {money(van_tir['van'])}")
    if van_tir['tir_anual']:
        print(f" • TIR:  {van_tir['tir_anual']}% anual")  

    print("\n☕ DESGLOSE DE TAZAS · ORDEN SHEETS")
    for taza, cantidad in cups_by_type.items():
        if cantidad > 0:
            print(f" • {taza:<12}: {int(cantidad):>4} tazas")

    print("\n🏷️ TOP 5 CAFES MAS COMPRADOS (min. 2 compras)")
    if vc_ranking.empty:
        print(" • No se detectó columna de cafetería en los datos.")
        print("   Verifica que exista una columna 'cafeteria', 'cafe' u 'origen' en la hoja.")
    elif ranking.empty:
        print(" • Aún no hay cafés repetidos (todas las compras son únicas).")
        print("   Compras registradas: " + " | ".join(f"{c} ({n}x)" for c, n in vc_ranking.head(5).items()))
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
        gid_merma= "1861632685",
        costo_cafetera=12239,
    )
    run_dashboard(config)