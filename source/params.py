MODEL_VERSION = "auto_v1"
CUP_FEATURES = ["weight_grams", "opening_month"]
DURATION_FEATURES = ["weight_grams", "opening_month"]

POST_CLOSURE_VARIABLES : frozenset[str] = frozenset({
    "cycle_days", "total_cups", "consumed_grams", "cups_per_day", "grams_per_day"
})

COEF_THRESHOLD = 0.01