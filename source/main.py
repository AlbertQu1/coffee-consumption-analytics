import sys
import pandas as pd
from config import load_config
from data_loader import (
    get_connection, fetch_bags, fetch_preparation, fetch_maintenance, fetch_waste,
    fetch_benchmark, fetch_weather_data, fetch_cetes_rate, insert_bag_prediction,
    fetch_bag_predictions, fetch_ical_events
)
from cleaning import clean_bags, clean_preparations, bags_with_preparations
from calculations import (
    add_season_columns, add_cycle_days, add_closed_day_info, add_cost_metrics,
    add_weather_metrics, find_invalid_date_bags, spending_summary,
    estimate_active_inventory, add_bag_eta, last_five_bags_performance,
    monthly_consumption, seasonal_consumption, season_name, climate_season,
    savings_vs_cafeteria, calculate_npv_irr, cup_size_breakdown, top_coffees,
    rank_efficiency, forecast_next_month, forecast_active_bag_completion,
    maintenance_alerts, purchases_by_location, weather_consumption_correlation,
    consumption_trend, prediction_accuracy, flag_atypical_months,
)
from modeling import train_best_model, predict_rows, historical_rate_baseline_mae
from report import (
    print_invalid_dates_warning, print_executive_summary, print_maintenance,
    print_bag_status, print_monthly_seasonal, print_weather_and_trend,
    print_roi_van_tir, print_ml_results, print_last_five_performance,
    print_top_coffees, print_efficiency_ranking, print_cup_breakdown,
    print_purchase_locations, print_prediction_accuracy,
)


def connect_or_exit():
    cfg = load_config()
    try:
        conn = get_connection(cfg)
        conn.connect().close()
        return cfg, conn
    except Exception as exc:
        print(f"⚠️ No se pudo conectar a la base de datos: {exc}")
        sys.exit(1)


def main():
    cfg, conn = connect_or_exit()

    df_bags = clean_bags(fetch_bags(conn))
    df_prep = fetch_preparation(conn)
    prep_summary = clean_preparations(df_prep)
    df_clean = bags_with_preparations(df_bags, prep_summary)
    df_maintenance = fetch_maintenance(conn)
    df_waste = fetch_waste(conn)
    df_benchmark = fetch_benchmark(conn)

    df_clean = add_season_columns(df_clean)
    df_clean = add_cycle_days(df_clean)
    df_clean = add_closed_day_info(df_clean)
    df_clean = add_cost_metrics(df_clean)

    fecha_inicio = df_clean["opened_date"].min().strftime("%Y-%m-%d")
    fecha_fin = min(df_clean["closed_date"].max(), pd.Timestamp.today()).strftime("%Y-%m-%d")
    df_clima = fetch_weather_data(cfg.latitude, cfg.longitude, fecha_inicio, fecha_fin)
    df_clean = add_weather_metrics(df_clean, df_clima)

    invalid_bags = find_invalid_date_bags(df_clean)
    if not invalid_bags.empty:
        print_invalid_dates_warning(invalid_bags)

    print("☕ Aquí está tu historia de café hasta hoy.\n")

    # 1. Panorama general
    spending = spending_summary(df_clean)
    active_inventory = estimate_active_inventory(df_clean)
    total_cups = df_clean["total_cups"].sum()
    total_grams = df_clean["consumed_grams"].sum()
    bags_for_count = df_clean[df_clean["roaster"] != "Anual"]
    print_executive_summary(bags_for_count, spending, active_inventory, total_cups, total_grams)

    # 7. Mantenimiento
    print_maintenance(maintenance_alerts(df_prep, df_maintenance))

    # 2. Tu bolsa de ahora
    features = ["weight_grams", "opening_month", "avg_precipitation", "avg_temp"]
    cups_result = train_best_model(df_clean, "total_cups", features)
    duration_result = train_best_model(df_clean, "cycle_days", features)
    cups_model, cups_features = cups_result[0], cups_result[1]
    duration_model, duration_features = duration_result[0], duration_result[1]

    df_to_predict = df_clean[df_clean["opened_date"].notna()].copy()
    df_predicted = predict_rows(cups_model, cups_features, df_to_predict, "predicted_total_cups", features)
    df_predicted = predict_rows(duration_model, duration_features, df_predicted, "predicted_cycle_days", features)

    active_bag = df_predicted[df_predicted["status"] == "ACTIVE"]
    active_bag_eta = add_bag_eta(active_bag, "predicted_cycle_days")
    if not active_bag_eta.empty:
        row = active_bag_eta.iloc[0]
        insert_bag_prediction(
            conn,
            int(row["id"]),
            float(row["predicted_total_cups"]),
            float(row["predicted_cycle_days"]),
            cups_result[3],
            duration_result[3],
        )

    monthly = monthly_consumption(df_prep)
    df_vacations = fetch_ical_events(cfg.vacation_calendar_url)
    df_visits = fetch_ical_events(cfg.visits_calendar_url)
    monthly = flag_atypical_months(monthly, df_vacations, df_visits)
    forecast = forecast_next_month(monthly)
    active_forecast = forecast_active_bag_completion(active_bag_eta, forecast)
    print_bag_status(active_bag_eta, active_forecast)

    # 4. Patrones de consumo
    seasonal = seasonal_consumption(df_prep, season_name)
    seasonal_climate = seasonal_consumption(df_prep, climate_season)
    print_monthly_seasonal(monthly, seasonal, seasonal_climate)
    print_cup_breakdown(cup_size_breakdown(df_prep))
    print_weather_and_trend(weather_consumption_correlation(df_clean), consumption_trend(monthly))

    # 6. El dinero
    savings = savings_vs_cafeteria(df_prep, df_clean, df_benchmark, cfg)
    tasa = fetch_cetes_rate(cfg.banxico_token)
    van_tir = calculate_npv_irr(cfg.machine_cost, savings["monthly_savings"], int(savings["months_since_purchase"]), tasa)
    print_roi_van_tir(savings, van_tir)

    # 3. Confiabilidad del modelo
    ritmo_mae = historical_rate_baseline_mae(df_clean)
    n_closed = len(df_clean[(df_clean["status"] == "CLOSED") & (df_clean["roaster"] != "Anual")])
    print_ml_results(cups_result, duration_result, ritmo_mae, n_closed)
    last_five = last_five_bags_performance(df_predicted, cups_result[2], duration_result[2])
    print_last_five_performance(last_five)
    df_predictions = fetch_bag_predictions(conn)
    accuracy = prediction_accuracy(df_clean, df_predictions)
    print_prediction_accuracy(accuracy)

    # 5. Recomendaciones de compra
    print_top_coffees(top_coffees(df_clean))
    print_efficiency_ranking(rank_efficiency(df_clean))
    print_purchase_locations(purchases_by_location(df_clean))


if __name__ == "__main__":
    main()
    