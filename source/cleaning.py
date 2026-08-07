import pandas as pd


def clean_preparations(df_prep: pd.DataFrame) -> pd.DataFrame:
    df_prep = df_prep.copy()
    df_prep["consumed_grams"] = df_prep["cups_prepared"] * df_prep["cup_grams"]
    summary = (
        df_prep.groupby("bag_id")
        .agg(
            total_cups=("cups_prepared", "sum"),
            consumed_grams=("consumed_grams", "sum"),
        )
        .reset_index()
    )
    return summary


def clean_bags(df_bags: pd.DataFrame) -> pd.DataFrame:
    df_bags = df_bags.copy()
    df_bags["roaster"] = df_bags["roaster"].replace(r"^\s*$", pd.NA, regex=True)
    df_bags["roaster"] = df_bags["roaster"].fillna(df_bags["coffee_name"])
    return df_bags


def bags_with_preparations(
    df_bags: pd.DataFrame, prep_summary: pd.DataFrame
) -> pd.DataFrame:
    df_bags = df_bags.copy()
    merged = df_bags.merge(prep_summary, left_on="id", right_on="bag_id", how="left")
    merged["total_cups"] = merged["total_cups"].fillna(0)
    merged["consumed_grams"] = merged["consumed_grams"].fillna(0)
    merged = merged.drop(columns=["bag_id"])
    return merged


if __name__ == "__main__":
    from config import load_config
    from data_loader import (
    get_connection,
    fetch_bags,
    fetch_preparation,
    fetch_waste,
    fetch_maintenance,
    fetch_benchmark,
)

    cfg = load_config()
    conn = get_connection(cfg)
    df_bags = fetch_bags(conn)
    df_prep = fetch_preparation(conn)

    df_bags = clean_bags(df_bags)
    prep_summary = clean_preparations(df_prep)
    df_clean = bags_with_preparations(df_bags, prep_summary)
    df_waste = fetch_waste(conn)
    df_maintenance = fetch_maintenance(conn)
    df_benchmark = fetch_benchmark(conn)

    print(df_waste)
    print(df_maintenance)
    print(df_benchmark)
