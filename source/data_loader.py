from config import Config
from sqlalchemy import create_engine, text
import psycopg2
import pandas as pd
import requests
import icalendar


def get_connection(cfg: Config):
    url = f"postgresql+psycopg2://{cfg.pguser}:{cfg.pgpassword}@{cfg.pghost}:{cfg.pgport}/{cfg.pgdatabase}"
    return create_engine(url)


def get_vacation_connection(cfg: Config):
    # Vacaciones app lives in its own database on the same Postgres cluster.
    url = f"postgresql+psycopg2://{cfg.pguser}:{cfg.pgpassword}@{cfg.pghost}:{cfg.pgport}/{cfg.vacation_database}"
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

def insert_bag_prediction(conn, bag_id, predicted_total_cups, predicted_cycle_days, cups_model, duration_model):
    with conn.begin() as c:
        c.execute(
            text("""
                INSERT INTO bag_predictions
                    (bag_id, predicted_total_cups, predicted_cycle_days, cups_model, duration_model)
                VALUES (:bag_id, :predicted_total_cups, :predicted_cycle_days, :cups_model, :duration_model)
            """),
            {
                "bag_id": bag_id,
                "predicted_total_cups": predicted_total_cups,
                "predicted_cycle_days": predicted_cycle_days,
                "cups_model": cups_model,
                "duration_model": duration_model,
            },
        )
        
def fetch_bag_predictions(conn) -> pd.DataFrame:
    sql = "SELECT id, bag_id, prediction_date, predicted_total_cups, predicted_cycle_days, cups_model, duration_model FROM bag_predictions"
    return pd.read_sql_query(sql, conn)

def fetch_vacation_trips(cfg: Config) -> pd.DataFrame:
    """Vacation data now comes from the Vacaciones app's own database (vacation_trips
    table) instead of an iCal feed. Returns the same shape fetch_ical_events used to
    (summary/start_date/end_date) so flag_atypical_months stays untouched."""
    conn = get_vacation_connection(cfg)
    df = pd.read_sql_query(
        "SELECT destination AS summary, departure_date AS start_date, return_date AS end_date FROM vacation_trips",
        conn,
    )
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def fetch_ical_events(url: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame(columns=["summary", "start_date", "end_date"])
    response = requests.get(url)
    cal = icalendar.Calendar.from_ical(response.text)
    events = []
    for component in cal.walk("VEVENT"):
        start = pd.Timestamp(component.get("DTSTART").dt)
        end = pd.Timestamp(component.get("DTEND").dt)
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_localize(None)
        events.append({
            "summary": str(component.get("SUMMARY")),
            "start_date": start.normalize(),
            "end_date": end.normalize(),
        })
    return pd.DataFrame(events)