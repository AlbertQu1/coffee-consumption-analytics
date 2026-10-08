import pandas as pd
import numpy as np
from datetime import datetime
import numpy_financial as npf


def season_name(month: int) -> str:
    if pd.isna(month):
        return "sin temporada"
    if month in (12, 1, 2):
        return "invierno"
    if month in (3, 4, 5):
        return "primavera"
    if month in (6, 7, 8):
        return "verano"
    return "otoño"


def climate_season(month: int) -> str:
    if pd.isna(month):
        return "sin temporada"
    if month in (5, 6, 7, 8, 9, 10):
        return "lluvias"
    return "seca"


def weekday_name(day: int) -> str:
    if pd.isna(day):
        return "sin dia"
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


def add_season_columns(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["opened_date"] = pd.to_datetime(df_bags["opened_date"])
    df_bags["opening_month"] = df_bags["opened_date"].dt.month
    df_bags["opening_season"] = df_bags["opening_month"].apply(season_name)
    df_bags["opening_climate_season"] = df_bags["opening_month"].apply(climate_season)
    return df_bags


def add_cycle_days(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["opened_date"] = pd.to_datetime(df_bags["opened_date"])
    df_bags["closed_date"] = pd.to_datetime(df_bags["closed_date"])
    today = pd.Timestamp(datetime.now().date())
    df_bags["end_date"] = df_bags["closed_date"].fillna(today)
    df_bags["cycle_days"] = (df_bags["end_date"] - df_bags["opened_date"]).dt.days + 1
    df_bags = df_bags.drop(columns=["end_date"])
    return df_bags


def add_cost_metrics(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["cost_per_cup_$"] = np.where(
        df_bags["total_cups"] > 0, df_bags["price"] / df_bags["total_cups"], np.nan
    ).round(2)
    df_bags["grams_per_cup"] = np.where(
        df_bags["total_cups"] > 0,
        df_bags["consumed_grams"] / df_bags["total_cups"],
        np.nan,
    ).round(2)
    return df_bags


def add_closed_day_info(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["closed_date"] = pd.to_datetime(df_bags["closed_date"])
    df_bags["closed_weekday"] = df_bags["closed_date"].dt.weekday

    conditions = [
        df_bags["opened_date"].isna(),
        df_bags["closed_date"].isna(),
    ]
    choices = ["pendiente", "bolsa abierta"]
    df_bags["closed_weekday_name"] = np.select(
        conditions, choices, default=df_bags["closed_weekday"].apply(weekday_name)
    )
    df_bags["is_weekend_closed"] = np.where(
        df_bags["closed_date"].isna(), np.nan, df_bags["closed_weekday"].isin([5, 6])
    )
    return df_bags


def add_normalized_metrics(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["cost_per_250g"] = np.where(
        df_bags["weight_grams"] > 0,
        df_bags["price"] / df_bags["weight_grams"] * 250,
        np.nan,
    ).round(2)
    df_bags["cups_per_205g"] = np.where(
        df_bags["weight_grams"] > 0,
        df_bags["total_cups"] / df_bags["weight_grams"] * 250,
        np.nan,
    ).round(2)
    return df_bags


def cups_since_maintenance(
    df_prep: pd.DataFrame, df_maintenance: pd.DataFrame, maintenance_type: str
) -> int:
    df_maintenance = df_maintenance.copy()
    df_maintenance["maintenance_date"] = pd.to_datetime(
        df_maintenance["maintenance_date"]
    ).dt.tz_localize("UTC")

    filtered = df_maintenance[df_maintenance["type"] == maintenance_type]
    if filtered.empty:
        return None
    last_date = filtered["maintenance_date"].max()
    recent_preps = df_prep[df_prep["prepared_timestamp"] >= last_date]
    return int(recent_preps["cups_prepared"].sum())


def unaccounted_grams(df_bags: pd.DataFrame, df_waste: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_waste = df_waste.copy()
    waste_summary = df_waste.groupby("bag_id")["grams_lost"].sum().reset_index()
    waste_summary = waste_summary.rename(columns={"grams_lost": "total_waste"})

    df_bags = df_bags.merge(waste_summary, left_on="id", right_on="bag_id", how="left")
    df_bags["total_waste"] = df_bags["total_waste"].fillna(0)

    df_bags["unaccounted_grams"] = np.where(
        df_bags["status"] == "CLOSED",
        df_bags["weight_grams"] - df_bags["consumed_grams"] - df_bags["total_waste"],
        np.nan,
    )
    df_bags = df_bags.drop(columns=["bag_id", "total_waste"])
    return df_bags


def calculate_npv_irr(
    initial_cost: float, monthly_savings: float, months: int, annual_rate: float
) -> dict:
    months = min(months, 12)
    monthly_rate = annual_rate / 12
    cash_flows = [-initial_cost] + [monthly_savings] * months
    npv = npf.npv(monthly_rate, cash_flows)
    monthly_irr = npf.irr(cash_flows)
    annual_irr = (1 + monthly_irr) ** 12 - 1 if monthly_irr is not None else None
    return {
        "npv": round(npv, 2),
        "annual_irr": round(annual_irr * 100, 2) if annual_irr else None,
        "discount_rate": round(annual_rate * 100, 2),
    }


def add_weather_metrics(
    df_bags: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    df_bags = df_bags.copy()

    def weather_during_cycle(row):
        if pd.isna(row["opened_date"]):
            return pd.Series({"avg_precipitation": np.nan, "avg_temp": np.nan})
        end_date = (
            row["closed_date"]
            if pd.notna(row["closed_date"])
            else pd.Timestamp(datetime.now().date())
        )
        mask = (df_weather["date"] >= row["opened_date"]) & (
            df_weather["date"] <= end_date
        )
        subset = df_weather[mask]
        if subset.empty:
            return pd.Series({"avg_precipitation": np.nan, "avg_temp": np.nan})
        return pd.Series(
            {
                "avg_precipitation": subset["precipitation"].mean(),
                "avg_temp": subset["temp_mean"].mean(),
            }
        )

    weather_cols = df_bags.apply(weather_during_cycle, axis=1)
    df_bags = pd.concat([df_bags, weather_cols], axis=1)
    return df_bags

def vacation_days_in_range(start_date, end_date, df_vacations: pd.DataFrame) -> int:
    """Cuenta cuantos dias de [start_date, end_date] (inclusive) caen dentro de algun
    viaje de df_vacations (columnas start_date/end_date). Usado para ajustar tasas de
    consumo (cups_per_day, etc.) restando dias en los que no habia nadie en casa."""
    if pd.isna(start_date) or pd.isna(end_date) or df_vacations.empty:
        return 0
    overlap_start = df_vacations["start_date"].clip(lower=start_date)
    overlap_end = df_vacations["end_date"].clip(upper=end_date)
    overlap_days = (overlap_end - overlap_start).dt.days + 1
    return int(overlap_days.clip(lower=0).sum())


def add_vacation_days_metric(df_bags: pd.DataFrame, df_vacations: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()

    def vacation_days_during_cycle(row):
        if pd.isna(row["opened_date"]):
            return 0
        end_date = (
            row["closed_date"]
            if pd.notna(row["closed_date"])
            else pd.Timestamp(datetime.now().date())
        )
        return vacation_days_in_range(row["opened_date"], end_date, df_vacations)

    df_bags["vacation_days_in_cycle"] = df_bags.apply(vacation_days_during_cycle, axis=1)
    return df_bags

def add_visit_days_metric(df_bags: pd.DataFrame, df_visits: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()

    def visit_days_during_cycle(row):
        if pd.isna(row["opened_date"]) or df_visits.empty:
            return 0
        end_date = (
            row["closed_date"]
            if pd.notna(row["closed_date"])
            else pd.Timestamp(datetime.now().date())
        )
        overlap_start = df_visits["start_date"].clip(lower=row["opened_date"])
        overlap_end = df_visits["end_date"].clip(upper=end_date)
        overlap_days = (overlap_end - overlap_start).dt.days + 1
        return overlap_days.clip(lower=0).sum()

    df_bags["visit_days_in_cycle"] = df_bags.apply(visit_days_during_cycle, axis=1)
    return df_bags

def add_bag_eta(df_bags: pd.DataFrame, ml_days_col: str) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["remaining_grams"] = np.maximum(
        0, df_bags["weight_grams"] - df_bags["consumed_grams"]
    )
    df_bags["observed_rate_g_per_day"] = np.where(
        df_bags["cycle_days"] > 0,
        df_bags["consumed_grams"] / df_bags["cycle_days"],
        0,
    )
    df_bags["days_by_rate"] = np.where(
        df_bags["observed_rate_g_per_day"] > 0,
        df_bags["cycle_days"]
        + df_bags["remaining_grams"] / df_bags["observed_rate_g_per_day"],
        df_bags[ml_days_col],
    )
    df_bags["predicted_total_days"] = np.maximum(
        df_bags[ml_days_col], df_bags["days_by_rate"]
    )
    df_bags["remaining_days"] = df_bags["predicted_total_days"] - df_bags["cycle_days"]
    df_bags["estimated_close_date"] = df_bags["opened_date"] + pd.to_timedelta(
        (df_bags["predicted_total_days"].round() - 1).clip(lower=0), unit="D"
    )
    return df_bags


def rank_efficiency(df_bags: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    paid = df_bags[df_bags["price"] > 0].copy()
    if paid.empty:
        return pd.DataFrame()
    paid["coffee_label"] = paid["roaster"] + " - " + paid["coffee_name"]
    paid["waste_pct"] = np.where(
        paid["weight_grams"] > 0,
        np.maximum(0, paid["weight_grams"] - paid["consumed_grams"]) / paid["weight_grams"] * 100,
        0,
    )
    ranking = (
        paid.groupby("coffee_label")
        .agg(
            cost_per_cup=("cost_per_cup_$", "mean"),
            total_cups=("total_cups", "sum"),
            purchases=("price", "count"),
            waste_pct=("waste_pct", "mean"),
        )
        .sort_values("cost_per_cup")
        .head(top_n)
    )
    return ranking


def top_coffees(
    df_bags: pd.DataFrame, min_purchase: int = 2, top_n: int = 5
) -> pd.Series:
    combined = df_bags["roaster"] + " - " + df_bags["coffee_name"]
    counts = combined.value_counts()
    return counts[counts >= min_purchase].head(top_n)


def purchases_by_location(df_bags: pd.DataFrame) -> pd.Series:
    real_locations = df_bags[~df_bags["location_name"].isin(["Anual"])]
    return (
        real_locations.groupby("location_name")["coffee_name"]
        .count()
        .sort_values(ascending=False)
    )


def monthly_consumption(df_prep: pd.DataFrame, df_vacations: pd.DataFrame | None = None) -> pd.DataFrame:
    df_prep = df_prep.copy()
    df_prep["month_period"] = df_prep["prepared_timestamp"].dt.to_period("M")
    monthly = df_prep.groupby("month_period").agg(
        total_cups=("cups_prepared", "sum"),
    )
    month_starts = monthly.index.to_timestamp()
    days_in_month = month_starts.days_in_month
    if df_vacations is not None and not df_vacations.empty:
        month_ends = month_starts + pd.offsets.MonthEnd(0)
        vacation_days = [
            vacation_days_in_range(s, e, df_vacations)
            for s, e in zip(month_starts, month_ends)
        ]
    else:
        vacation_days = 0
    # dias en casa: total del mes menos los que estuve de viaje (minimo 1 para no dividir entre 0
    # si un mes entero se fue de vacaciones).
    monthly["active_days"] = np.maximum(days_in_month - np.array(vacation_days), 1)
    monthly["cups_per_day"] = monthly["total_cups"] / monthly["active_days"]
    return monthly


def forecast_next_month(monthly: pd.DataFrame, reference_date=None) -> dict:
    if reference_date is None:
        reference_date = pd.Timestamp(datetime.now().date())
    next_month = (reference_date.replace(day=1) + pd.DateOffset(months=1)).to_period(
        "M"
    )
    reference_period = next_month - 12
    if reference_period not in monthly.index:
        return {"reference_period": str(reference_period), "cups_per_day": None}
    return {
        "reference_period": str(reference_period),
        "cups_per_day": monthly.loc[reference_period, "cups_per_day"],
    }


def estimate_active_inventory(df_bags: pd.DataFrame) -> float:
    active = df_bags[df_bags["status"].isin(["ACTIVE", "PENDING"])]
    remaining = np.maximum(0, active["weight_grams"] - active["consumed_grams"])
    return float(remaining.sum())


def spending_summary(df_bags: pd.DataFrame) -> dict:
    paid = df_bags[(df_bags["price"] > 0) & (df_bags["status"] != "PENDING")]
    total_spent = float(paid["price"].sum())
    paid_bags_count = int(len(paid))
    avg_price_per_bag = total_spent / paid_bags_count if paid_bags_count else 0
    avg_cost_per250g = (
        total_spent / paid["weight_grams"].sum() * 250
        if not paid.empty and paid["weight_grams"].sum()
        else 0
    )
    avg_cost_per_cup = (
        total_spent / df_bags["total_cups"].sum() if df_bags["total_cups"].sum() else 0
    )
    return {
        "total_spent": total_spent,
        "paid_bags_count": paid_bags_count,
        "avg_price_per_bag": round(avg_price_per_bag, 2),
        "avg_cost_per_250g": round(avg_cost_per250g, 2),
        "avg_cost_per_cup": round(avg_cost_per_cup, 2),
    }


def cost_per_cup_by_year(df_bags: pd.DataFrame) -> dict:
    df_bags = df_bags.copy()
    df_bags["opened_date"] = pd.to_datetime(df_bags["opened_date"])
    paid = df_bags[(df_bags["price"] > 0) & (df_bags["status"] != "PENDING")]
    result = {}
    for year, group in paid.groupby(paid["opened_date"].dt.year):
        total_cups_year = group["total_cups"].sum()
        if total_cups_year > 0:
            result[int(year)] = float(group["price"].sum() / total_cups_year)
    return result


def bag_status_counts(df_bags: pd.DataFrame) -> pd.Series:
    return df_bags["status"].value_counts()


def cup_size_breakdown(df_prep: pd.DataFrame) -> pd.Series:
    return (
        df_prep.groupby("cup_size")["cups_prepared"].sum().sort_values(ascending=False)
    )


def historical_waste_estimate(df_bags: pd.DataFrame) -> float:
    closed = df_bags[df_bags["status"] == "CLOSED"]
    waste = np.maximum(0, closed["weight_grams"] - closed["consumed_grams"])
    return float(waste.sum())


def weekday_consumption(df_prep: pd.DataFrame) -> pd.Series:
    df_prep = df_prep.copy()
    df_prep["weekday"] = df_prep["prepared_timestamp"].dt.weekday.apply(weekday_name)
    return (
        df_prep.groupby("weekday")["cups_prepared"].sum().sort_values(ascending=False)
    )


def closing_day_summary(df_bags: pd.DataFrame) -> dict:
    closed = df_bags[df_bags["closed_date"].notna()]
    weekday_counts = closed["closed_weekday_name"].value_counts()
    most_common_day = weekday_counts.index[0] if not weekday_counts.empty else None
    weekend_count = int(closed["is_weekend_closed"].sum())
    weekday_count = len(closed) - weekend_count
    return {
        "most_common_close_day": most_common_day,
        "weekday_closures": weekday_count,
        "weekend_closures": weekend_count,
    }


def seasonal_consumption(df_prep: pd.DataFrame, season_func) -> pd.DataFrame:
    df_prep = df_prep.copy()
    df_prep["season"] = df_prep["prepared_timestamp"].dt.month.apply(season_func)
    seasonal = df_prep.groupby("season").agg(
        total_cups=("cups_prepared", "sum"),
        active_days=("prepared_timestamp", lambda s: s.dt.date.nunique()),
    )
    seasonal["cups_per_day"] = seasonal["total_cups"] / seasonal["active_days"]
    return seasonal


def maintenance_alerts(df_prep: pd.DataFrame, df_maintenance: pd.DataFrame) -> dict:
    cups_since_clean = cups_since_maintenance(df_prep, df_maintenance, "CLEAN")
    cups_since_descale = cups_since_maintenance(df_prep, df_maintenance, "DESCALE")

    remaining_to_clean = None if cups_since_clean is None else max(0, 200 - cups_since_clean)
    remaining_to_descale = None if cups_since_descale is None else max(0, 600 - cups_since_descale)

    return {
        "needs_clean": cups_since_clean is not None and cups_since_clean >= 200,
        "needs_descale": cups_since_descale is not None and cups_since_descale >= 600,
        "remaining_to_clean": remaining_to_clean,
        "remaining_to_descale": remaining_to_descale,
    }


def savings_vs_cafeteria(
    df_prep: pd.DataFrame, df_bags: pd.DataFrame, df_benchmark: pd.DataFrame, cfg
) -> dict:
    df_prep = df_prep.copy()
    df_prep["date"] = df_prep["prepared_timestamp"].dt.date
    daily = df_prep.groupby("date").agg(cups=("cups_prepared", "sum")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year

    df_benchmark = df_benchmark.copy()
    df_benchmark["year"] = pd.to_datetime(df_benchmark["benchmark_date"]).dt.year
    price_by_year = df_benchmark.groupby("year")["price"].mean().to_dict()
    overall_avg_price = df_benchmark["price"].mean()

    daily["market_price"] = daily["year"].map(price_by_year).fillna(overall_avg_price)
    daily["equivalent_cost"] = daily["cups"] * daily["market_price"]

    equivalent_total = float(daily["equivalent_cost"].sum())
    equivalent_by_year = (
        daily.groupby("year")["equivalent_cost"].sum().round(2).to_dict()
    )

    spending = spending_summary(df_bags)
    df_bags = df_bags.copy()
    df_bags["opened_date"] = pd.to_datetime(df_bags["opened_date"])
    paid = df_bags[(df_bags["price"] > 0) & (df_bags["status"] != "PENDING")]
    spent_by_year = (
        paid.groupby(paid["opened_date"].dt.year)["price"].sum().round(2).to_dict()
    )

    all_years = sorted(
        set(list(equivalent_by_year.keys()) + list(spent_by_year.keys()))
    )
    net_by_year = {
        int(y): round(equivalent_by_year.get(y, 0.0) - spent_by_year.get(y, 0.0), 2)
        for y in all_years
    }

    net_total = equivalent_total - spending["total_spent"]
    profit_after_roi = max(0.0, net_total - cfg.machine_cost)

    purchase_date = pd.to_datetime(cfg.purchase_date, dayfirst=True)
    days_since_purchase = (pd.Timestamp(datetime.now().date()) - purchase_date).days
    months_since_purchase = round(days_since_purchase / 30.44, 1)
    monthly_savings = (
        net_total / months_since_purchase if months_since_purchase > 0 else 0.0
    )

    return {
        "equivalent_cafeteria_cost": round(equivalent_total, 2),
        "equivalent_by_year": {int(k): v for k, v in equivalent_by_year.items()},
        "real_spending": spending["total_spent"],
        "spent_by_year": {int(k): v for k, v in spent_by_year.items()},
        "net_savings_by_year": net_by_year,
        "net_savings_total": round(net_total, 2),
        "days_since_purchase": days_since_purchase,
        "months_since_purchase": months_since_purchase,
        "monthly_savings": round(monthly_savings, 2),
        "profit_after_roi": round(profit_after_roi, 2),
    }


def last_five_bags_performance(
    df_bags: pd.DataFrame, cups_mae: float, duration_mae: float
) -> pd.DataFrame:
    closed = df_bags[(df_bags["status"] == "CLOSED") & (df_bags["roaster"] != "Anual")].copy()
    last_five = (
        closed.sort_values("closed_date", ascending=False)
        .head(5)
        .sort_values("closed_date")
    )

    last_five["waste_grams"] = np.maximum(
        0, last_five["weight_grams"] - last_five["consumed_grams"]
    )
    last_five["waste_pct"] = np.where(
        last_five["weight_grams"] > 0,
        last_five["waste_grams"] / last_five["weight_grams"] * 100,
        0,
    )
    last_five["cups_per_day"] = np.where(
        last_five["cycle_days"] > 0,
        last_five["total_cups"] / last_five["cycle_days"],
        np.nan,
    )
    if "predicted_total_cups" in last_five.columns and cups_mae is not None:
        last_five["cups_range_low"] = last_five["predicted_total_cups"] - cups_mae
        last_five["cups_range_high"] = last_five["predicted_total_cups"] + cups_mae
    if "predicted_cycle_days" in last_five.columns and duration_mae is not None:
        last_five["days_range_low"] = last_five["predicted_cycle_days"] - duration_mae
        last_five["days_range_high"] = last_five["predicted_cycle_days"] + duration_mae

        cols_to_round = ["waste_grams", "waste_pct", "cups_per_day"]
    if "cups_range_low" in last_five.columns:
        cols_to_round += ["cups_range_low", "cups_range_high"]
    if "days_range_low" in last_five.columns:
        cols_to_round += ["days_range_low", "days_range_high"]
    last_five[cols_to_round] = last_five[cols_to_round].round(2)
    last_five["predicted_total_cups"] = last_five["predicted_total_cups"].round(2)
    last_five["predicted_cycle_days"] = last_five["predicted_cycle_days"].round(2)

    return last_five


def forecast_active_bag_completion(
    active_bag: pd.DataFrame, forecast: dict
) -> pd.DataFrame:
    active_bag = active_bag.copy()
    cups_per_day_forecast = forecast.get("cups_per_day")
    if cups_per_day_forecast and cups_per_day_forecast > 0:
        remaining_cups = np.maximum(
            0, active_bag["predicted_total_cups"] - active_bag["total_cups"]
        )
        active_bag["forecast_remaining_days"] = remaining_cups / cups_per_day_forecast
        active_bag["forecast_completion_date"] = pd.Timestamp(
            datetime.now().date()
        ) + pd.to_timedelta(active_bag["forecast_remaining_days"], unit="D")
    else:
        active_bag["forecast_remaining_days"] = np.nan
        active_bag["forecast_completion_date"] = pd.NaT
    return active_bag


def find_invalid_date_bags(df_bags: pd.DataFrame) -> pd.DataFrame:
    return df_bags[
        df_bags["closed_date"].notna()
        & df_bags["opened_date"].notna()
        & (df_bags["closed_date"] < df_bags["opened_date"])
    ]

def weather_consumption_correlation(df_bags: pd.DataFrame) -> dict:
    valid = df_bags[
        df_bags["avg_precipitation"].notna() & (df_bags["cycle_days"] > 0)
    ].copy()
    valid["cups_per_day"] = valid["total_cups"] / valid["cycle_days"]
    return {
        "precipitation_correlation": round(valid["avg_precipitation"].corr(valid["cups_per_day"]), 3),
        "temperature_correlation": round(valid["avg_temp"].corr(valid["cups_per_day"]), 3),
        "sample_size": len(valid),
    }
    
def consumption_trend(monthly: pd.DataFrame) -> dict:
    if len(monthly) < 2:
        return {"trend_direction": None, "slope": None}
    y = monthly["cups_per_day"].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    return {"slope": round(float(slope), 4), "trend_direction": direction}

if __name__ == "__main__":
    from config import load_config
    from data_loader import get_connection, fetch_bags, fetch_preparation, fetch_weather_data
    from cleaning import clean_bags, clean_preparations, bags_with_preparations

    cfg = load_config()
    conn = get_connection(cfg)

    df_bags = clean_bags(fetch_bags(conn))
    df_prep = fetch_preparation(conn)
    prep_summary = clean_preparations(df_prep)
    df_clean = bags_with_preparations(df_bags, prep_summary)
    df_clean = add_season_columns(df_clean)
    df_clean = add_cycle_days(df_clean)

    fecha_inicio = df_clean["opened_date"].min().strftime("%Y-%m-%d")
    fecha_fin = df_clean["closed_date"].max().strftime("%Y-%m-%d")
    df_clima = fetch_weather_data(cfg.latitude, cfg.longitude, fecha_inicio, fecha_fin)
    df_clean = add_weather_metrics(df_clean, df_clima)

    print("--- correlacion clima-consumo ---")
    print(weather_consumption_correlation(df_clean))

    monthly = monthly_consumption(df_prep)
    print("--- tendencia de consumo ---")
    print(consumption_trend(monthly))
    
def prediction_accuracy(df_bags: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
    closed = df_bags[df_bags["status"] == "CLOSED"][["id", "coffee_name", "closed_date", "total_cups", "cycle_days"]]
    last_five_closed = closed.sort_values("closed_date", ascending=False).head(5)

    latest_predictions = (
        df_predictions.sort_values("id")
        .groupby("bag_id", as_index=False)
        .last()
    )

    merged = latest_predictions.merge(last_five_closed, left_on="bag_id", right_on="id", how="inner")
    merged["cups_error"] = merged["predicted_total_cups"] - merged["total_cups"]
    merged["days_error"] = merged["predicted_cycle_days"] - merged["cycle_days"]
    return merged[[
        "coffee_name", "prediction_date", "predicted_total_cups", "total_cups", "cups_error",
        "predicted_cycle_days", "cycle_days", "days_error", "cups_model", "duration_model",
    ]]
    
def flag_atypical_months(monthly: pd.DataFrame, vacations: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    month_starts = monthly.index.to_timestamp()
    month_ends = month_starts + pd.offsets.MonthEnd(0)

    def overlaps(events, m_start, m_end):
        if events.empty:
            return False
        return ((events["start_date"] <= m_end) & (events["end_date"] >= m_start)).any()

    monthly["on_vacation"] = [overlaps(vacations, s, e) for s, e in zip(month_starts, month_ends)]
    monthly["had_visitors"] = [overlaps(visits, s, e) for s, e in zip(month_starts, month_ends)]
    return monthly

def add_recent_cup_size_metric(
    df_bags: pd.DataFrame, df_prep: pd.DataFrame, window_days: int = 30
) -> pd.DataFrame:
    df_bags = df_bags.copy()
    prep_timestamps = df_prep["prepared_timestamp"].dt.tz_localize(None)
    global_avg = (df_prep["cup_grams"] * df_prep["cups_prepared"]).sum() / df_prep["cups_prepared"].sum()

    def recent_cup_size(row):
        if pd.isna(row["opened_date"]):
            return global_avg
        window_start = row["opened_date"] - pd.Timedelta(days=window_days)
        mask = (prep_timestamps >= window_start) & (prep_timestamps < row["opened_date"])
        subset = df_prep[mask]
        if subset.empty or subset["cups_prepared"].sum() == 0:
            return global_avg
        return (subset["cup_grams"] * subset["cups_prepared"]).sum() / subset["cups_prepared"].sum()

    df_bags["recent_avg_cup_grams"] = df_bags.apply(recent_cup_size, axis=1)
    return df_bags

def add_cycle_avg_cup_grams_metric(df_bags: pd.DataFrame, df_prep: pd.DataFrame) -> pd.DataFrame:
    """Promedio real de gramos/taza DENTRO del ciclo de cada bolsa (no en una
    ventana previa a abrirla) -- captura que una bolsa consumida casi puro
    Espresso dura mas que una con mezcla de tazas grandes, aunque el peso
    total sea el mismo."""
    df_bags = df_bags.copy()
    global_avg = (df_prep["cup_grams"] * df_prep["cups_prepared"]).sum() / df_prep["cups_prepared"].sum()

    def cycle_cup_size(row):
        subset = df_prep[df_prep["bag_id"] == row["id"]]
        if subset.empty or subset["cups_prepared"].sum() == 0:
            return global_avg
        return (subset["cup_grams"] * subset["cups_prepared"]).sum() / subset["cups_prepared"].sum()

    df_bags["cycle_avg_cup_grams"] = df_bags.apply(cycle_cup_size, axis=1)
    return df_bags