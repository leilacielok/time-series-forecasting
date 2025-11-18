# 🌀 Time Series Forecasting of Renewable Energy Consumption

This project provides a **modular and reproducible pipeline** for forecasting monthly renewable energy consumption across multiple sectors in the United States, using models such as **Seasonal Naïve**, **Random Walk**, **ARIMA**, **SARIMA**, **VAR** and **VARX**.

It is designed as a clean, production-ready Python package structure.  
The goal of this academic project is to benchmark simple univariate models against multivariate models and evaluate their forecasting accuracy across different renewable energy sources and economic sectors. Seasonal Naïve, Random Walk, ARIMA and SARIMA serve as benchmarks — for each sector, we select the best-performing model as a baseline for comparison. We then assess the performance of more advanced multivariate models like VAR and VARX against these baselines. VAR (Vector Autoregression) captures interdependencies among multiple sector time series, while VARX (Vector Autoregression with exogenous regressors) incorporates external variables (e.g., oil and natural gas prices) that might influence renewable energy consumption.

---

## 📁 Project Structure

```
time_series_forecasting/
├─ data/
│ ├─ US_renewable_forecasting.csv # main dataset (monthly values)
│ └─ macro_vars.csv # data on WTI and natural gas (monthly values)
├─ forecasting/
│ ├─ __init__.py
│ ├─ preprocessing.py # data loading, cleaning, reshaping
│ ├─ models.py # time series forecasting models
│ └─ evaluation.py # accuracy metrics and summaries
├─ outputs/
│ ├─ plots/
│ │   ├─ ACF_PACF/
│ │   ├─ actual_vs_best_benchmark/
│ │   ├─ actual_vs_VAR/
│ │   ├─ actual_vs_VARx/
│ │   ├─ model_predictions_comparison/
| |   └─ RMSFE/
│ └─ forecast_summary.csv # RMSFE, MAVE by sector for each model
├─ main.py # orchestrates the full pipeline
├─ analysis_plots.ipynb
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Setup and Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/leilacielok/time-series-forecasting.git
   cd time-series-forecasting
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS / Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Place your dataset**
   Copy your CSV file (.csv) into the ```data/``` folder.
---
## 🧩 Data Description and Preprocessing
The US_renewable_forecasting dataset contains monthly renewable energy consumption for multiple U.S. sectors from 1970–2022, with columns such as:

| Year | Month | Sector | Hydroelectric Power | Geothermal Energy | Solar Energy | Biomass Energy | Total Renewable Energy | ... |

All preprocessing is handled through modular functions in ``` forecasting/preprocessing.py ```:
1. Date construction — creates a proper monthly timestamp (```date```) from ```Year``` and ```Month```.
2. Sector normalization — fixes typographical inconsistencies (e.g. “Commerical” → “Commercial”).
3. Transportation adjustment — replaces the Transportation sector’s total with the sum of biofuels (```Biodiesel```, ```Ethanol```, ```Other Biofuels```, ```Renewable Diesel Fuel```).
4. Pivoting — converts the dataset so that each sector becomes a time series column.
5. Trimming — restricts analysis to the period after 1995 for stable, modern data.
6. Transformation — applies a log transformation to stabilize variance (optional).

---
## 📈 Models Implemented
```
| Model                           | Description                                                                | Reference     |
|---------------------------------|----------------------------------------------------------------------------|---------------|
| **Seasonal Naïve**              | Uses last year’s same-month value as forecast.                             | Baseline      |
| **Random Walk (with drift)**    | Predicts next value as last observed + average drift.                      | ARIMA(0,1,0)  |
| **ARIMA (p,d,q)**               | Non-seasonal autoregressive integrated moving average.                     | Box–Jenkins   |
| **SARIMA (p,d,q)(P,D,Q)[s]**    | Seasonal ARIMA with seasonality `s = 12`.                                  | Statsmodels   |
| **VAR (Vector Autoregression)** | Multivariate model capturing interdependencies among sector time series.   | statsmodels   |
| **VARX**                        | VAR model extended with exogenous regressors (e.g., oil and gas prices).   | statsmodels   |
```
Each univariate benchmark model (up to SARIMA) is recursively re-estimated in a rolling one-step forecast fashion. Model selection within these benchmarks is automated via AIC minimization across a predefined parameter grid. For each sector, the best-performing benchmark is selected to compare against multivariate models VAR and VARX.
---
## 📊 Evaluation Metrics
The forecasting performance is evaluated on a hold-out test set (typically 20% of observations) using the following metrics:
- MAFE — Mean Absolute Forecast Error
- RMSFE — Root Mean Squared Forecast Error
- Results for all models and sectors are automatically exported to ```outputs/forecast_summary.csv```
---
## 🧠 Code Overview
```
| Module / Notebook                 | Description                                                                                |
|----------------------------------|--------------------------------------------------------------------------------------------|
| **`forecasting/preprocessing.py`** | Handles data ingestion, cleaning, and transformation (date creation, sector normalization, pivoting). |
| **`forecasting/models.py`**        | Implements Random Walk, ARIMA, SARIMA, and Seasonal Naïve models with recursive one-step forecasting, and also provides VAR/VARX for multivariate forecasting. |
| **`forecasting/evaluation.py`**    | Provides forecast error metrics (MAFE, RMSFE) and summary table construction.            |
| **`analysis_plots.ipynb`**         | Jupyter Notebook used for exploratory data analysis and visualization of model outputs. Creates and saves visual comparisons of actual vs predicted values, benchmark vs multivariate models, and residual diagnostics.  |
| **`main.py`**                      | The central pipeline controller — loads data, runs models, evaluates forecasts, and exports summary results. |
```
##🚀 Running the Full Pipeline
To run the entire forecasting pipeline:
``` bash
python main.py
```
If everything is configured correctly, you should see:
``` bash
Saved: outputs/forecast_summary.csv
```
The script will automatically create the ```outputs/``` folder if it does not exist.
---
## 🔍 Example Use Case
Once the pipeline is running, you can explore:
* Forecast accuracy by sector (Industrial, Residential, etc.)
* Model comparison: baseline forecasts (Seasonal Naïve, Random Walk, ARIMA, SARIMA) vs advanced multivariate models (VAR, VARX)
* Export and visualize forecast errors over time
---
## 🧰 Reproducibility
This repository follows best practices for reproducible forecasting research:
* All file paths are relative (e.g., ```data/```, ```outputs/```).
* A ```requirements.txt``` file freezes exact dependency versions.
* The code produces deterministic results (no random components).
---
## 🪪 License
This project is released under the **MIT License**.
You are free to use, modify, and distribute it for research, educational, or personal projects.
---
## 🧾 Citation
If you use this repository or adapt it for your research, please cite:
```
@software{time_series_forecasting_2025,
  author       = {Leila Cielok},
  title        = {Time Series Forecasting of Renewable Energy Consumption},
  year         = {2025},
  url          = {https://github.com/leilacielok/time-series-forecasting},
  note         = {Modular forecasting pipeline using univariate and multivariate models.}
}
```
---
## ✨ Author
**Leila Cielok**

Developed as part of an advanced time series forecasting project exploring renewable energy trends.
