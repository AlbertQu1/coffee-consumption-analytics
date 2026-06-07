# ☕ Coffee Consumption Analytics

A personal consumption intelligence system built on real data tracked since 2025. Combines live data ingestion from Google Sheets, statistical analysis, and a Machine Learning model that predicts when a coffee bag will run out.

---

## The Challenge

How much does it actually cost to brew coffee at home versus buying it at a café? Which brand offers the best value per gram? When will I run out of coffee?

These questions led to building a personal data pipeline: manual logging in Google Sheets → automatic ingestion → statistical analysis → ML predictions → daily executive report.

---

## Process

**1. Data ingestion and pipeline**

Data is recorded across three Google Sheets tabs (consumption log, historical prices, gram weights by drink type) and loaded automatically on each run via public URL. Column names are normalized using `unicodedata` to handle spelling variations, and a daily CSV snapshot is persisted locally for historical tracking.

**2. Metric engineering**

For each closed bag: actual cups brewed, grams consumed, duration in days, estimated waste, and cost per cup adjusted to the purchase year's market price. An ROI metric is built by comparing real spend against the equivalent cost of buying the same drinks at a coffee shop.

**3. Machine Learning model**

A `RandomForestRegressor` (scikit-learn) was trained on closed bag data to predict two variables per active bag: estimated days until depletion and number of cups remaining. The model retrains automatically with each new closed bag, improving its predictions over time.

**4. Seasonality and behavioral analysis**

Daily consumption averages were calculated by month and season to identify behavioral patterns throughout the year.

---

## Results

| Metric | Value |
|--------|-------|
| Bags analyzed | 31 (24 closed, 4 active) |
| Cups consumed | 611 |
| Total coffee consumed | 11,858 g |
| Total spent on coffee | $3,183 MXN |
| **Real cost per cup** | **$5.56 MXN** |
| **Net savings vs. café** | **$13,194 MXN** |
| Model error (duration) | ±9.8 days per bag |
| Model error (cups) | ±3.1 cups per bag |

**Seasonal consumption:**
- Peak season: **summer** (1.66 cups/day)
- Lowest season: **winter** (1.21 cups/day)
- Historical peak month: June 2025 (2.00 cups/day)

**Most consumed drink:** espresso (450 out of 611 cups = 73.7%)

**Model accuracy on last 5 closed bags:**

| Bag | Actual cups | ML cups | Actual days | ML days |
|-----|-------------|---------|-------------|---------|
| Tierra Garat | 12 | 12.3 | 5 | 6.6 |
| Chav Medio | 25 | 24.6 | 14 | 13.6 |
| N'Duva | 14 | 13.3 | 10 | 11.9 |
| Colombia | 23 | 22.5 | 20 | 19.5 |
| Costalero | 25 | 25.5 | 30 | 25.8 |

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243)
![Google Sheets](https://img.shields.io/badge/Data-Google%20Sheets-34A853)

- **Language:** Python
- **Libraries:** Pandas, NumPy, scikit-learn (RandomForestRegressor), dataclasses
- **Data source:** Google Sheets (live ingestion via public CSV)
- **Persistence:** Local CSV with automatic daily snapshot
- **Techniques:** Metric engineering, seasonality analysis, Random Forest regression, historical tracking

---

## Dataset

Personal data collected since May 2025. Each consumption record includes: date, drink type, brand, bag weight, price paid, and cups brewed. Café prices are updated annually for ROI calculation.

---

## Repository Structure

```
├── coffee_consumption_analysis.py   # Main pipeline (ingestion → analysis → ML → report)
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Roadmap

- Embedded visualizations (monthly consumption chart, model accuracy curve)
- Interactive dashboard with Streamlit
- Automatic alerts when inventory drops below a critical threshold
