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

- **Legacy vs. new data granularity is handled explicitly, not papered over.** Historical (pre-app) data only has per-bag aggregate cup-size counts; the new app captures true per-cup timestamps. Any analysis that needs per-cup detail (time-of-day, day-of-week patterns) is scoped to new data only, by design — legacy data is never backfilled with fabricated precision it doesn't have. Where legacy data does need reconstructed timestamps (e.g. distributing a bag's cup-size counts across its open/close window for monthly trend analysis), that limitation is documented, not hidden.
- **A "grain" discipline runs through the whole pipeline**: `data_loader.py` never joins across tables that answer different questions (a bag purchase vs. a cup consumed vs. a maintenance event); combining them only happens in `cleaning.py`/`calculations.py`, deliberately.
- **A recurring "special case" bag (a perpetual ground-coffee resource, not a normal purchase-to-empty cycle) is explicitly excluded from every pattern-based calculation** — ML training, historical-rate baselines, and recent-performance reporting all filter it out by the same rule, rather than letting it silently distort results in some places and not others.
- **Real migrated data surfaced bugs that synthetic test data never would have** — validating against the actual multi-year history (not just hand-built seed rows) caught issues like a monthly consumption-rate metric that quietly assumed every day of the month had at least one log entry, and a top-coffees ranking that conflated two different roasters selling a similarly-named coffee. Fixing these against real data, rather than relying solely on synthetic tests, is treated as a required step before any production cutover.

---

## Roadmap (Active Development)

**Done:**
- One-time migration of the full historical record (bags, preparations reconstructed from cup-size counts, waste, cafeteria benchmarks) from the legacy Google Sheet into a disposable, freely-resettable test database — verified end to end (cup totals match the source exactly)
- Prediction logging: every pipeline run stores a snapshot of its prediction for the active bag, so realized accuracy can be measured once that bag actually closes

**Still pending:**
- Finish writing pipeline outputs to Postgres/Google Sheets for Power BI, Looker Studio, and Tableau Public consumption
- Cutover to a dedicated production database, separate from the development/test database
- Build the actual prediction-vs-reality comparison report (the data is being logged; the comparison view isn't wired into the report output yet)
- Vacation/guest-visit tracking to explain consumption dips and spikes that aren't seasonal
- Home Assistant sensor integration to replace city-level weather with actual home climate data
- Remote access via Tailscale, with logging screens restricted to the home network (VPN configured on the home server so far, not yet on mobile)

---

*This document reflects the project state as of August 2026 and will be updated as the pipeline moves from development/test data to production.*
