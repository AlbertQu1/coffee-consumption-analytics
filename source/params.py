MODEL_VERSION = "auto_v2"
CUP_FEATURES = [
    "weight_grams", "opening_month", "avg_temp", "avg_precipitation",
    "vacation_days_in_cycle", "visit_days_in_cycle", "recent_avg_cup_grams",
    "cycle_avg_cup_grams"
    ]
DURATION_FEATURES = [
    "weight_grams", "opening_month", "avg_temp", "avg_precipitation",
    "vacation_days_in_cycle", "visit_days_in_cycle", "recent_avg_cup_grams",
    "cycle_avg_cup_grams"
    ]

POST_CLOSURE_VARIABLES : frozenset[str] = frozenset({
    "cycle_days", "total_cups", "consumed_grams", "cups_per_day", "grams_per_day"
})

COEF_THRESHOLD = 0.01