from config import Config
from sqlalchemy import create_engine 
import psycopg2
import pandas as pd
import requests


def get_connection(cfg: Config):
    url = f"postgresql+psycopg2://{cfg.pguser}:{cfg.pgpassword}@{cfg.pghost}:{cfg.pgport}/{cfg.pgdatabase}"
    return create_engine(url)


def fetch_bags(conn) -> pd.DataFrame:
    sql = """
        SELECT 
            cb.id, cb.coffee_name, cb.roaster, cb.purchase_date,
            cb.weight_grams, cb.price, cb.is_gift,
            cb.opened_date, cb.closed_date, cb.status,
            pl.name as location_name
        FROM coffee_bags cb
        JOIN purchase_locations pl on pl.id = cb.purchase_location_id
        """
    return pd.read_sql_query(sql, conn)


def fetch_preparation(conn) -> pd.DataFrame:
    sql = """
        SELECT
            cp.id, date_trunc('minute', cp.prepared_timestamp) as prepared_timestamp, cp.bag_id,
            cb.coffee_name as coffee_name, cs.name as cup_size, cp.cups_prepared, cp.with_milk, cp.cold_coffee,
            cp.alcohol_type_ids, cp.other_alcohol, cp.flavor_ids, cp.other_flavor, cs.coffee_grams as cup_grams
        FROM coffee_preparations cp
        JOIN cup_sizes cs on cs.id = cp.cup_size_id
        JOIN coffee_bags cb on cb.id = cp.bag_id
        """
    return pd.read_sql_query(sql, conn)


def fetch_waste(conn) -> pd.DataFrame:
    sql = """
        SELECT cw.id, cw.bag_id, cb.coffee_name as coffee_name, cw.waste_date, cw.grams_lost, cw.reason
        FROM waste cw
        JOIN coffee_bags as cb on cb.id = cw.bag_id
        """
    return pd.read_sql_query(sql, conn)


def fetch_maintenance(conn) -> pd.DataFrame:
    sql = """
        SELECT cm.id, cm.maintenance_type as type, cm.performed_date as maintenance_date
        FROM maintenance cm
        """
    return pd.read_sql_query(sql, conn)


def fetch_benchmark(conn) -> pd.DataFrame:
    sql = """
        SELECT id, cafeteria_name, city, price, created_at::date as benchmark_date
        FROM cafeteria_benchmarks
        """
    return pd.read_sql_query(sql, conn)


def fetch_cetes_rate(token: str) -> float:
    try:
        url = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43936/datos/oportuno"
        headers = {"Bmx-Token": token}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        value = response.json()["bmx"]["series"][0]["datos"][0]["dato"]
        return round(float(value) / 100, 6)
    except Exception as exc:
        print(f"⚠️ Error al obtener la tasa de Cetes: {exc}...")
        return 0.10


def fetch_weather_data(
    latitude: float, longitude: float, start_date: str, end_date: str
) -> pd.DataFrame:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,precipitation_sum"
        f"&timezone=America%2FMexico_City"
    )
    try:
        resp= requests.get(url,timeout=15)
        resp.raise_for_status()
        data = resp.json()["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(data["time"]),
                "temp_mean": data["temperature_2m_mean"],
                "precipitation": data["precipitation_sum"]
            }
        )
        return df
    except Exception as exc:
       print(f"⚠️ Error al obtener datos meteorológicos: {exc}")
    return pd.DataFrame()
