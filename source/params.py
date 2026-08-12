MODEL_VERSION = "auto_v1"
CUP_FEATURES = ["weight_grams", "opening_month", "vacation_days_in_cycle"]
DURATION_FEATURES = ["weight_grams", "opening_month", "vacation_days_in_cycle"]

POST_CLOSURE_VARIABLES : frozenset[str] = frozenset({
    "cycle_days", "total_cups", "consumed_grams", "cups_per_day", "grams_per_day"
})

COEF_THRESHOLD = 0.01