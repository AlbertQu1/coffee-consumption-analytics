# Coffee Consumption Analytics — Project Summary

**Version:** 2.0 · August 2026

---

## Overview

End-to-end personal analytics project built entirely in Python to track, analyze, and forecast coffee consumption from a home espresso machine. The project spans data engineering, financial modeling, machine learning, and external API integration — reading directly from the same PostgreSQL database that powers a companion full-stack logging app ([Coffee Logger](https://github.com/AlbertQu1/Loggers)).

This is a full refactor of an earlier Google-Sheets-driven monolithic script into a clean, modular pipeline: `config → data_loader → cleaning → params → calculations → modeling → report → main`, mirroring standard data-engineering separation of concerns (ingestion, transformation, business logic, ML, presentation, orchestration).

---

## Business Problem

How do you measure the actual ROI of a home espresso machine vs. buying coffee at a café? And beyond financials, can consumption data be used to predict when a coffee bag will run out, identify seasonal/weekly patterns, flag maintenance needs automatically, and rank purchases by real cost-efficiency?

---

## Technical Stack

- Python (Pandas, NumPy, Scikit-learn, SQLAlchemy)
- PostgreSQL as the single source of truth — same database a live Next.js logging app writes to in real time
- External APIs: Banxico SIE (CETES discount rate), Open-Meteo (historical weather)
- numpy_financial for NPV/IRR calculations
- Scikit-learn: Lasso, Ridge, ElasticNet, KNN Regressor — automatic model selection per target
- Planned: Power BI / Looker Studio / Tableau Public dashboards, orchestrated on a schedule via n8n

---

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Typed, frozen config loaded from environment variables; secrets never printed |
| `data_loader.py` | Pure data fetching — Postgres tables + external APIs, no business logic |
| `cleaning.py` | Per-source normalization (data-quality fixes, merges across tables at the correct grain) |
| `params.py` | ML constants: feature lists, data-leakage guardrails, coefficient threshold |
| `calculations.py` | All derived business metrics — cost/cup, cycle length, seasonal patterns, ROI, waste, forecasting |
| `modeling.py` | Feature engineering, leave-one-out cross-validation, automatic model selection, prediction |
| `report.py` | Human-readable formatting only — no calculations |
| `main.py` | Orchestrates the full pipeline end to end, with graceful DB-connection error handling |

---

## Key Features

**Financial Analysis**
- ROI and payback period tracked against the machine's real purchase date
- Net savings computed against real, per-year café benchmark pricing (not a flat average)
- NPV and IRR calculated at the live CETES discount rate via Banxico's API, capped to a standardized 12-month evaluation window (so future equipment purchases stay comparable)
- Efficiency ranking: coffees ranked by real cost per cup, gift bags correctly excluded from cost math, unopened bags excluded from realized-cost averages

**Consumption Analytics**
- True per-cup timestamps (not estimated/interpolated) power monthly, seasonal, and weekday consumption breakdowns
- Two parallel seasonal lenses: traditional 4-season and CDMX-specific rain/dry season — chosen because prior analysis suggested rainy season drives higher consumption
- Statistical correlation (not just visual proximity) between weather and consumption rate
- Linear trend detection on monthly consumption rate
- Forecasting: next month's projected rate derived from the same calendar month one year prior, applied to the active bag's remaining predicted cups to estimate a completion date

**Machine Learning**
- Leave-one-out cross-validation for unbiased evaluation on a small dataset (bag-level grain, not per-cup)
- Automatic model selection across Lasso, Ridge, ElasticNet, and KNN — winner chosen by MAE, re-evaluated with and without a coffee-identity feature
- Rare coffee varieties (bought fewer than 3 times) are grouped into a generic category before one-hot encoding, preventing the model from "memorizing" one-off purchases instead of learning a real pattern
- Explicit data-leakage guardrail: any variable only knowable after a bag closes is excluded from training by design
- Benchmark comparison: ML model vs. a naive historical-rate baseline, to justify (or reject) the added complexity
- Confidence levels calibrated to sample size

**Inventory & Predictions**
- Active-bag completion estimate combining two independent methods (the bag's own observed consumption rate, and the trained ML model), taking the more conservative of the two
- A separate, seasonally-adjusted forecast for the same active bag, shown side by side for comparison
- Data-quality safeguard: automatic detection of bags with an impossible date order (closed before opened)

**Maintenance Tracking**
- Automated clean/descale alerts based on real cup-count thresholds since the last logged maintenance event

---

## Design Decisions Worth Highlighting

- **Legacy vs. new data granularity is handled explicitly, not papered over.** Historical (pre-app) data only has per-bag aggregate counts; the new app captures true per-cup timestamps. Any analysis that needs per-cup detail (time-of-day, day-of-week patterns) is scoped to new data only, by design — legacy data is never backfilled with fabricated precision it doesn't have.
- **A "grain" discipline runs through the whole pipeline**: `data_loader.py` never joins across tables that answer different questions (a bag purchase vs. a cup consumed vs. a maintenance event); combining them only happens in `cleaning.py`/`calculations.py`, deliberately.
- **Two real data-quality bugs were caught by validating against actual data, not assumptions** — a `NaN`-vs-empty-string bug in a text field that `.fillna()` silently ignored, and a one-hot-encoding "memorization" bug from a synthetic test dataset with too many unique categories, mirroring a guardrail the original script had already engineered around.

---

## Roadmap (Active Development)

- Finish writing pipeline outputs to Postgres/Google Sheets for Power BI, Looker Studio, and Tableau Public consumption
- One-time migration of ~34 months of pre-app historical data (reconciled against known data-entry inconsistencies, documented and corrected)
- Cutover to a dedicated production database, separate from the development/test database
- Prediction logging: every model run against the active bag is stored, so realized accuracy can be measured once that bag actually closes
- Vacation/guest-visit tracking to explain consumption dips and spikes that aren't seasonal
- Home Assistant sensor integration to replace city-level weather with actual home climate data

---

*This document reflects the project state as of August 2026 and will be updated as the pipeline moves from development/test data to production.*
