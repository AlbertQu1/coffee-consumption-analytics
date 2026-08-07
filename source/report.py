import pandas as pd
from calculations import bag_status_counts
from modeling import _confidence_level


def money(value: float) -> str:
    return f"${value:,.2f} MXN"


def print_invalid_dates_warning(invalid_bags):
    for _, row in invalid_bags.iterrows():
        print(
            f"⚠️  {row['coffee_name']}: cierre ({row['closed_date'].date()}) es anterior a la apertura ({row['opened_date'].date()})"
        )


def print_executive_summary(
    df_bags, spending, active_inventory, total_cups, total_grams
):
    status_counts = bag_status_counts(df_bags)
    closed = status_counts.get("CLOSED", 0)
    active = status_counts.get("ACTIVE", 0)
    pending = status_counts.get("PENDING", 0)
    total = len(df_bags)

    print("\n🧠 LECTURA EJECUTIVA")
    print(
        f" • Bolsas analizadas:        {total} ({closed} cerradas, {active} en maquina, {pending} en espera)"
    )
    print(f" • Tazas consumidas:         {int(total_cups)}")
    print(f" • Total consumido:          {total_grams:,.0f} g")
    print(f" • Inventario activo est.:   {active_inventory:,.0f} g")
    print(f" • Gasto real en cafe:       {money(spending['total_spent'])}")
    print(
        f" • Costo real por taza:      {money(spending['avg_cost_per_cup'])} (historico)"
    )


def print_maintenance(alerts):
    if alerts:
        print(
            f"\nAntes de seguir, un aviso importante: tu cafetera necesita {' y '.join(alerts).lower()} pronto."
        )
    else:
        print(
            "\nAntes de seguir, un chequeo rápido: tu cafetera está al día, no necesita mantenimiento todavía."
        )


def print_bag_status(active_bag_with_eta, active_forecast):
    print("\n☕ TU BOLSA DE AHORA")
    if active_bag_with_eta.empty:
        print(" • Sin café disponible — comprar pronto")
        return
    row = active_bag_with_eta.iloc[0]
    own_days = f"{row['remaining_days']:.0f} dias"
    own_date = (
        str(row["estimated_close_date"].date())
        if pd.notna(row["estimated_close_date"])
        else "sin ETA"
    )

    if not active_forecast.empty and pd.notna(
        active_forecast.iloc[0]["forecast_completion_date"]
    ):
        seasonal_row = active_forecast.iloc[0]
        seasonal_days = f"{seasonal_row['forecast_remaining_days']:.0f} dias"
        seasonal_date = str(seasonal_row["forecast_completion_date"].date())
    else:
        seasonal_days = "sin datos"
        seasonal_date = "sin datos"

    print(f" • {row['coffee_name']} — {row['remaining_grams']:.0f}g restantes")
    print(f" {'':<22}{'Segun tu ritmo':<20}{'Segun temporada':<20}")
    print(f" {'Dias restantes':<22}{own_days:<20}{seasonal_days:<20}")
    print(f" {'Fecha estimada':<22}{own_date:<20}{seasonal_date:<20}")


def print_monthly_seasonal(monthly, seasonal_traditional, seasonal_climate):
    print("\n🗓️ CONSUMO MENSUAL Y ESTACIONAL")
    if monthly.empty:
        print(" • No hay suficientes datos para distribuir consumo por mes.")
        return
    top_month = monthly.sort_values("cups_per_day", ascending=False).index[0]
    print(f" • Mes mas intenso:          {top_month} ({monthly.loc[top_month, 'cups_per_day']:.2f} tz/dia)")
    if not seasonal_traditional.empty:
        top_season = seasonal_traditional.sort_values("cups_per_day", ascending=False).index[0]
        low_season = seasonal_traditional.sort_values("cups_per_day", ascending=True).index[0]
        print(f" • Temporada mas intensa:    {top_season} ({seasonal_traditional.loc[top_season, 'cups_per_day']:.2f} tz/dia)")
        print(f" • Temporada mas baja:       {low_season} ({seasonal_traditional.loc[low_season, 'cups_per_day']:.2f} tz/dia)")
    if not seasonal_climate.empty:
        top_climate = seasonal_climate.sort_values("cups_per_day", ascending=False).index[0]
        print(f" • Lluvias vs. seca:         {top_climate} consume mas ({seasonal_climate.loc[top_climate, 'cups_per_day']:.2f} tz/dia)")
    print(" • Ultimos 6 meses:")
    for period, row in monthly.tail(6).iterrows():
        print(f"   - {period}: {row['total_cups']:.0f} tazas")


def print_weather_and_trend(correlation, trend):
    direction_text = {
        "increasing": "va en aumento",
        "decreasing": "va a la baja",
        "stable": "se mantiene estable",
    }.get(trend["trend_direction"], "sin tendencia clara")
    print(
        f"\n🌦️ Tu consumo {direction_text} con el tiempo, sin una relación fuerte con el clima "
        f"(correlacion lluvia: {correlation['precipitation_correlation']}, temperatura: {correlation['temperature_correlation']})."
    )


def print_roi_van_tir(savings, van_tir):
    print("\n💰 ROI Y VALOR")
    print(
        f" • Cafetera comprada:        {savings['days_since_purchase']} dias, {savings['months_since_purchase']} meses"
    )
    print(f" • Equiv. cafeteria (bruto): {money(savings['equivalent_cafeteria_cost'])}")
    print(f" • Gasto real en cafe:       {money(savings['real_spending'])}")
    print(f" • Ahorro neto:              {money(savings['net_savings_total'])}")
    print(f" • Ahorro neto/mes:          {money(savings['monthly_savings'])}")
    if savings["profit_after_roi"] > 0:
        print(f" ✅ Ganancia post-ROI:        {money(savings['profit_after_roi'])}")
        print(
            "   (ganancia real acumulada durante todo el tiempo de uso, sin tope de 12 meses)"
        )
    print(f"\n📊 VAN / TIR  (CETES {van_tir['discount_rate']}%)")
    print(
        "   Nota: evaluado sobre 12 meses desde la fecha de compra del equipo (no año calendario ni tiempo total real)"
    )
    print(
        "   — así cualquier inversión futura (equipo adicional, reventa) se puede comparar con el mismo criterio."
    )
    print(f" • VAN:  {money(van_tir['npv'])}")
    if van_tir["annual_irr"]:
        print(f" • TIR:  {van_tir['annual_irr']}% anual")


def print_ml_results(cups_result, duration_result, ritmo_mae, n_closed):
    (
        cups_model,
        cups_features,
        cups_mae,
        cups_winner,
        cups_coefs,
        cups_bench,
        cups_meta,
    ) = cups_result
    (
        duration_model,
        duration_features,
        duration_mae,
        duration_winner,
        duration_coefs,
        duration_bench,
        duration_meta,
    ) = duration_result

    confidence = _confidence_level(n_closed)
    print("\n🤖 MODELO ML")
    print(f"  Estabilidad: {n_closed} bolsas cerradas  |  Confianza: {confidence}")

    if duration_model is None:
        print(
            f" • Duracion: usando promedio historico ({n_closed} bolsa(s), min. 3 para ML)"
        )
    else:
        print(f" • Duracion [{duration_winner}]  MAE: ±{duration_mae:.1f} dias")
        bench_str = "  ".join(f"{k}: ±{v:.1f}" for k, v in duration_bench.items())
        print(f"   Benchmark: {bench_str}")
        if ritmo_mae is not None:
            mejor = "ML mejor" if duration_mae <= ritmo_mae else "ritmo mejor"
            print(f"   vs. ritmo historico: ±{ritmo_mae:.1f} dias  ({mejor})")
        helped = duration_meta.get("coffee_name_helped")
        print(
            f"   coffee_name: {'incluido (ayuda)' if helped else 'excluido (no ayuda)'}"
        )
        active_coefs = [(f, c) for f, c in duration_coefs.items() if abs(c) >= 0.01]
        if active_coefs:
            print(
                "   Activas (|coef|>=0.01): "
                + "  ".join(f"{f}({c:+.2f})" for f, c in active_coefs[:5])
            )

    if cups_model is None:
        print(
            f" • Tazas: usando promedio historico ({n_closed} bolsa(s), min. 3 para ML)"
        )
    else:
        print(f" • Tazas [{cups_winner}]  MAE: ±{cups_mae:.1f} tazas")
        bench_str = "  ".join(f"{k}: ±{v:.1f}" for k, v in cups_bench.items())
        print(f"   Benchmark: {bench_str}")
        helped = cups_meta.get("coffee_name_helped")
        print(
            f"   coffee_name: {'incluido (ayuda)' if helped else 'excluido (no ayuda)'}"
        )


def print_last_five_performance(last_five):
    print("\n🏆 RENDIMIENTO · ULTIMAS 5 BOLSAS CERRADAS")
    if last_five.empty:
        print(" • No hay ciclos cerrados suficientes.")
        return
    print(
        f" {'Fecha':<12}{'Cafe':<22}{'Tazas (ML)':<16}{'Dias (ML)':<16}{'Merma':<10}{'Tz/dia':<8}"
    )
    for _, row in last_five.iterrows():
        cierre = (
            str(row["closed_date"].date())
            if pd.notna(row["closed_date"])
            else "sin cierre"
        )
        tazas_str = f"{int(row['total_cups'])} ({row['predicted_total_cups']:.1f})"
        dias_str = f"{int(row['cycle_days'])} ({row['predicted_cycle_days']:.1f})"
        merma_str = f"{row['waste_pct']:.1f}%"
        print(
            f" {cierre:<12}{row['coffee_name']:<22}{tazas_str:<16}{dias_str:<16}{merma_str:<10}{row['cups_per_day']:<8.2f}"
        )


def print_top_coffees(top):
    print("\n🏷️ TOP CAFES MAS COMPRADOS (min. 2 compras)")
    if top.empty:
        print(" • Aun no hay cafes repetidos.")
        return
    for coffee, count in top.items():
        print(f" • {coffee}: {count} compras")


def print_efficiency_ranking(ranking):
    print("\n📊 RANKING EFICIENCIA (menor costo/taza = mejor)")
    if ranking.empty:
        print(" • Sin bolsas pagadas registradas.")
        return
    for coffee, row in ranking.iterrows():
        cost_str = f"{money(row['cost_per_cup'])}/taza"
        print(f" • {coffee:<30}{cost_str:<18}|  {int(row['total_cups']):>3} tazas  |  {int(row['purchases'])} compra(s)")


def print_cup_breakdown(breakdown):
    print("\n☕ DESGLOSE DE TAZAS")
    for size, cups in breakdown.items():
        print(f" • {size:<12}: {int(cups):>4} tazas")


def print_purchase_locations(locations):
    print("\n🗺️ ORIGEN DE COMPRAS")
    for city, count in locations.items():
        print(f" • {city:<20} {count} bolsa(s)")
