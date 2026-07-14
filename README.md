# Coffee Consumption Analytics — Project Summary

**Version:** 1.1 · July 14, 2026

---

## Overview

End-to-end personal analytics project built entirely in Python to track, analyze, and forecast coffee consumption from a home espresso machine. The project spans data engineering, financial modeling, machine learning, external API integration, and automated maintenance tracking — using real consumption data logged over 13+ months.

---

## Business Problem

How do you measure the actual ROI of a $12,239 MXN home espresso machine vs. buying coffee at a café? And beyond financials, can historical consumption data be used to predict inventory depletion, identify seasonal patterns, automate maintenance alerts, and track purchasing behavior across travel?

---

## Technical Stack

- Python (Pandas, NumPy, Scikit-learn)
- Google Sheets as live data source (gspread + public CSV API)
- gspread write access for automated data entry (maintenance cup counter)
- External APIs: Banxico SIE (CETES rate), Open-Meteo (historical weather)
- numpy_financial for NPV/IRR calculations
- Scikit-learn: Lasso, Ridge, ElasticNet, KNN Regressor

---

## Data Pipeline

Raw consumption data is logged manually in Google Sheets (bag name, grams, open/close dates, preparation type and count, purchase city). Python fetches this via API on each run, performs all transformations, runs ML models, and appends a daily snapshot to a local CSV for historical tracking.

The pipeline handles: multi-sheet joins (consumption, pricing, gram standards, waste, maintenance, travel), date validation, waste integration, gram standardization across 9 preparation types, automated gspread writes, and graceful fallbacks when external APIs are unavailable.

---

## Key Features

**Financial Analysis**
- ROI tracking: machine paid itself off in ~5.6 months
- Net savings to date: $29,091 MXN ($2,171/month)
- Post-ROI gain: $16,852 MXN
- Year-over-year breakdown: 2025 vs 2026 savings and spend
- NPV and IRR calculated at live CETES rate via Banxico API

**Consumption Analytics**
- Daily distribution of consumption across all active and closed bags
- Monthly and seasonal trend analysis with peak/low identification
- Weather correlation: temperature and precipitation joined to consumption periods via Open-Meteo API
- Seasonal validation: June/July consumption confirmed as structural (rainy season in CDMX), not habit growth — validated by comparing Jun 2025 (~2.17 cups/day) vs Jun 2026 (2.20 cups/day)

**Machine Learning**
- Leave-One-Out cross-validation for unbiased evaluation on small datasets
- Model selection across Lasso, Ridge, ElasticNet, and KNN — winner chosen by MAE
- Automatic feature importance audit: active vs inactive coefficients
- Leakage prevention: post-closure variables (total cups, cycle days) explicitly excluded from training
- Conditional StandardScaler applied to KNN (distance-based) but not linear models
- Benchmark comparison: ML models vs simple historical rhythm baseline
- Confidence levels calibrated to dataset size (low/medium/high)

**Inventory & Predictions**
- ETA prediction for active bag combining ML forecast and observed consumption rhythm
- Confidence interval displayed alongside point estimate
- Waste tracking: integration of manual loss data from a separate Sheets tab

**Maintenance Tracking**
- Automated clean/descale alerts based on cup count thresholds (clean ≥200, descale ≥600)
- User logs date and maintenance type in Google Sheets; script auto-fills cup count via gspread write API
- Counter resets automatically on each new maintenance event
- Bidirectional Sheets integration: reads via public CSV API, writes via gspread service account

---

## Key Insights from Data

- 73% of all 696 cups prepared are straight espresso — consistent behavioral pattern
- Summer is the highest-consumption season (2.07 cups/day), winter lowest (1.21)
- Simple rhythm-based prediction (±8.2 days) marginally outperforms ML (±8.6 days) — indicating consumption is more habitual than seasonal at bag level
- Bags tend to close on Wednesdays (77% weekday closures)

---

## What Makes This Project Relevant for BI Roles

- Real-world pipeline: data collection → transformation → modeling → reporting
- Business framing: every technical decision connects to a financial or operational question
- Defensible ML: LOO validation, leakage audit, model comparison, confidence calibration
- API integration with graceful degradation (Banxico, Open-Meteo)
- Bidirectional Google Sheets integration: read via public CSV API, write via gspread service account (maintenance tracking)
- Iterative development with version control and feature roadmap management
- Designed with future scalability: modular architecture, dashboard-ready CSV output, multi-sheet data model

---

## Roadmap (Active Development)

- ML cup prediction — add predicted cup count alongside day prediction in active bag ETA output
- Output restructuring — reorganize into 5 executive sections (Inventory, Finance, Consumption, Last 5 bags, ML)
- Fix historical CSV export — unify schema across all rows (prerequisite for Tableau/Looker)
- Modular architecture — split into coffee_config, coffee_data, coffee_models, coffee_analytics, coffee_main
- Forecasting — project next month's consumption based on same-month historical average
- Tableau and Looker Studio dashboards connected to Google Sheets / Supabase
- Climate features in ML model — staged addition at 50 bags (temperature) and 60 bags (precipitation)
- Simpson's Paradox analysis on preparation mix
- CAGR when 3+ years of data available

---

*This document reflects the project state as of July 2026 and will be updated with each significant feature release.*