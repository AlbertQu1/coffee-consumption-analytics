import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor


def build_features(
    rows: pd.DataFrame, features: list[str], use_coffee_name: bool = True
) -> pd.DataFrame:
    if use_coffee_name:
        rows = rows.copy()
        coffee_counts = rows["coffee_name"].value_counts()
        rows["coffee_model"] = np.where(
            rows["coffee_name"].map(coffee_counts) >= 3, rows["coffee_name"], "OTROS"
        )
        return pd.get_dummies(
            rows[features + ["coffee_model"]], columns=["coffee_model"], dummy_na=False
        )
    return rows[features].copy()


def _loo_mae(
    model_class, params: dict, x: pd.DataFrame, y: pd.Series, scale=False
) -> Optional[float]:
    if len(x) < 5:
        return None
    errors = []
    for idx in range(len(x)):
        x_train = x.drop(x.index[idx])
        x_test = x.iloc[[idx]]
        y_train = y.drop(y.index[idx])
        if scale:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
        m = model_class(**params)
        m.fit(x_train, y_train)
        errors.append(abs(float(m.predict(x_test)[0]) - y.iloc[idx]))
    return float(np.mean(errors))


def _confidence_level(n: int) -> str:
    if n < 20:
        return "Baja <20 bolsas"
    elif n <= 50:
        return "media (20-50 bolsas)"
    return "Alta"


def train_best_model(cycles: pd.DataFrame, target: str, features: list[str]):
    empty_meta = {
        "coffee_name_helped": None,
        "mae_without_coffee": None,
        "mae_with_coffee": None,
    }
    closed = cycles[cycles["status"] == "CLOSED"].copy()
    closed = closed[closed["roaster"] != "Anual"]
    closed = closed[closed[target].notna()]
    if len(closed) < 3:
        return None, [], None, "ninguno", {}, {}, empty_meta

    x_with_name = build_features(closed, features, use_coffee_name=True)
    x_without_name = build_features(closed, features, use_coffee_name=False)
    y = closed[target].astype(float)
    cv_folds = min(len(closed), 5)

    lasso_cv = LassoCV(cv=cv_folds, random_state=42, max_iter=10000)
    lasso_cv.fit(x_with_name, y)
    ridge_cv = RidgeCV(cv=cv_folds)
    ridge_cv.fit(x_with_name, y)
    en_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0], cv=cv_folds, random_state=42, max_iter=10000
    )
    en_cv.fit(x_with_name, y)

    candidates = {
        "Lasso": (Lasso, {"alpha": lasso_cv.alpha_, "max_iter": 10000}, False),
        "Ridge": (Ridge, {"alpha": ridge_cv.alpha_}, False),
        "ElasticNet": (
            ElasticNet,
            {"alpha": en_cv.alpha_, "l1_ratio": en_cv.l1_ratio_, "max_iter": 10000},
            False,
        ),
        "KNN": (KNeighborsRegressor, {"n_neighbors": 3}, True),
    }

    benchmark = {}
    for name, (cls, params, scale) in candidates.items():
        mae_val = _loo_mae(cls, params, x_with_name, y, scale=scale)
        if mae_val is not None:
            benchmark[name] = mae_val

    winner_name = (
        min(benchmark, key=lambda k: benchmark[k]) if benchmark else "ElasticNet"
    )
    cls_winner, params_winner, scale_winner = candidates[winner_name]
    mae_with_name = benchmark.get(winner_name)

    mae_without_name = _loo_mae(
        cls_winner, params_winner, x_without_name, y, scale=scale_winner
    )

    if (
        mae_without_name is not None
        and mae_with_name is not None
        and mae_without_name <= mae_with_name
    ):
        benchmark = {}
        for name, (cls, params, scale) in candidates.items():
            mae_val = _loo_mae(cls, params, x_without_name, y, scale=scale)
            if mae_val is not None:
                benchmark[name] = mae_val
        winner_name = (
            min(benchmark, key=lambda k: benchmark[k]) if benchmark else winner_name
        )
        cls_winner, params_winner, scale_winner = candidates[winner_name]
        x = x_without_name
        coffee_name_helped = False
    else:
        x = x_with_name
        coffee_name_helped = True if mae_with_name is not None else None

    model = cls_winner(**params_winner)
    model.fit(x, y)

    if hasattr(model, "coef_"):
        coefs = {f: float(c) for f, c in zip(x.columns, model.coef_)}
    else:
        coefs = {}
    coefs_sorted = dict(sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    meta = {
        "coffee_name_helped": coffee_name_helped,
        "mae_without_coffee": mae_without_name,
        "mae_with_coffee": mae_with_name,
    }

    return (
        model,
        list(x.columns),
        benchmark.get(winner_name),
        winner_name,
        coefs_sorted,
        benchmark,
        meta,
    )


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

    x = build_features(result, base_features).reindex(
        columns=feature_columns, fill_value=0
    )
    result[output_col] = model.predict(x)
    return result


def historical_rate_baseline_mae(df_bags: pd.DataFrame) -> Optional[float]:
    closed = df_bags[
        (df_bags["status"] == "CLOSED") & (df_bags["cycle_days"] > 0) & (df_bags["roaster"] != "Anual")
    ].copy()
    if len(closed) < 5:
        return None
    closed["cups_per_day"] = closed["total_cups"] / closed["cycle_days"]
    closed = closed.reset_index(drop=True)
    errors = []
    for i in range(len(closed)):
        subset = closed.drop(i)
        rate = subset["cups_per_day"].mean() if not subset.empty else 0.0
        if rate > 0:
            predict_days = closed.loc[i, "total_cups"] / rate
            real_days = closed.loc[i, "cycle_days"]
            errors.append(abs(predict_days - real_days))
        if not errors:
            return None
        return float(np.mean(errors))
