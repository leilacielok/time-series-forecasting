from pathlib import Path
import pandas as pd
import numpy as np

from forecasting.preprocessing import (
    load_csv, build_month_start_date, normalize_sectors,
    transportation_from_biofuels, pivot_sector_timeseries,
    trim_period, detect_numeric_targets,
    maybe_log_transform
)
from forecasting.models import (
    seasonal_naive, forecast_rw_recursive,
    forecast_arima_recursive, forecast_sarima_recursive,
    forecast_var_recursive,      word
    forecast_varx_recursive,     
    forecast_varx_wti_recursive,
)
from forecasting.evaluation import summarize_series, rmsfe, granger_causality_matrix

CSV_PATH = 'data/US_renewable_forecasting.csv'
MACRO_PATH = 'data/macro_vars.csv'
ARIMA_ORDER_GRID = [(0,1,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1)]

TEST_SIZE_RATIO = 0.2
INCLUDE_DRIFT_IN_RW = False  # o True se vuoi il drift

VAR_SECTOR_COLS = [
    "Commercial",
    "Residential",
    "Transportation",
    "Industrial",
    "Electric Power",
]


def run():
    # === 0) Cartella output ===
    out = Path('outputs')
    out.mkdir(exist_ok=True)

    # === 1) Dati energia rinnovabile ===
    df = load_csv(CSV_PATH)
    df = build_month_start_date(df)
    df = normalize_sectors(df)
    df = transportation_from_biofuels(df)

    ts = pivot_sector_timeseries(df)
    ts = trim_period(ts, start='1995-01-01')

    targets = detect_numeric_targets(ts)

    # === 2) Dati macro (WTI + gas) ===
    # usa pd.read_csv, NON load_csv
    macro = pd.read_csv(MACRO_PATH, parse_dates=['date'])
    macro.set_index('date', inplace=True)

    # riallinea alle date di ts
    macro = macro.reindex(ts.index)

    # log-prezzi
    macro['log_WTI'] = np.log(macro['WTI'])
    macro['log_gas'] = np.log(macro['gas'])

    # regressori esogeni (anche con NaN)
    exog = macro[['log_WTI', 'log_gas']]

    # SOLO per VAR-X: taglio 1997+ (righe dove ho sia WTI che gas)
    mask = exog.notna().all(axis=1)
    ts_vx   = ts.loc[mask]     # endogene per VAR-X (1997+)
    exog_vx = exog.loc[mask]   # esogene per VAR-X (1997+)
    exog_wti_vx  = exog[['log_WTI']].loc[mask]  # esogene solo WTI

    # === 3) Benchmark: sNaive, RW, ARIMA, SARIMA ===
    rows = []

    for col in targets:
        y_raw = ts[col].dropna()
        y_t, logged = maybe_log_transform(y_raw)

        # split
        h = max(1, int(round(len(y_t) * TEST_SIZE_RATIO)))
        test_index = y_t.index[-h:]

        # s-naive
        yhat_s = seasonal_naive(y_t).loc[test_index]

        # RW
        yhat_rw = forecast_rw_recursive(
            y_t,
            include_drift=INCLUDE_DRIFT_IN_RW
        ).loc[test_index]

        # ARIMA
        yhat_ar, spec_ar = forecast_arima_recursive(y_t, ARIMA_ORDER_GRID)
        yhat_ar = yhat_ar.loc[test_index]

        # SARIMA
        yhat_sar, spec_s, spec_S = forecast_sarima_recursive(y_t)
        yhat_sar = yhat_sar.loc[test_index]

        rows.append(summarize_series(
            y_raw,
            {'sNaive': yhat_s, 'RW': yhat_rw, 'ARIMA': yhat_ar, 'SARIMA': yhat_sar},
            {'Spec_ARIMA': str(spec_ar), 'Spec_SARIMA': f'{spec_s} x {spec_S}'},
            logged
        ))

    summary_df = pd.DataFrame(rows).set_index('Series').sort_values('RMSFE_SARIMA')

    # === 4) VAR “puro”: solo i 5 settori ===
    var_rows = []
    for col in VAR_SECTOR_COLS:
        y_test_var, var_hat_lvl = forecast_var_recursive(
            ts=ts,
            target_col=col,
            var_cols=VAR_SECTOR_COLS,
            test_size_ratio=TEST_SIZE_RATIO,
            maxlags=4,
        )
        var_rows.append({
            "Series": col,
            "RMSFE_VAR": rmsfe(y_test_var, var_hat_lvl),
        })

    var_df = pd.DataFrame(var_rows).set_index("Series")

    # === 5) VAR-X con WTI + gas (sample 1997+) ===
    varx_rows = []
    for col in VAR_SECTOR_COLS:
        y_test_vx, varx_hat_lvl = forecast_varx_recursive(
            ts=ts_vx,      # solo 1997+
            exog=exog_vx,  # log_WTI, log_gas
            target_col=col,
            var_cols=VAR_SECTOR_COLS,
            test_size_ratio=TEST_SIZE_RATIO,
            maxlags=4,
        )
        varx_rows.append({
            "Series": col,
            "RMSFE_VARX": rmsfe(y_test_vx, varx_hat_lvl),
        })

    varx_df = pd.DataFrame(varx_rows).set_index("Series")

    # === 6) VAR-X con solo WTI (stesso sample 1997+) ===
    varx_wti_rows = []
    for col in VAR_SECTOR_COLS:
        y_test_vx_wti, varx_wti_hat_lvl = forecast_varx_wti_recursive(
            ts=ts_vx,          # solo 1997+
            exog=exog_wti_vx,  # SOLO log_WTI
            target_col=col,
            var_cols=VAR_SECTOR_COLS,
            test_size_ratio=TEST_SIZE_RATIO,
            maxlags=4,
        )
        varx_wti_rows.append({
            "Series": col,
            "RMSFE_VARX_WTI": rmsfe(y_test_vx_wti, varx_wti_hat_lvl),
        })

    varx_wti_df = pd.DataFrame(varx_wti_rows).set_index("Series")

    # === 7) Full RMSFE summary table (benchmarks + VAR + VARX) ===
    summary_all_df = (
        summary_df
        .join(var_df, how="left")
        .join(varx_df, how="left")
        .join(varx_wti_df, how="left")
    )

    summary_all_df.to_csv(out / 'forecast_summary.csv', float_format='%.3f')
    print('Saved:', (out / 'forecast_summary.csv').resolve())

    # === 8) Granger causality analysis (VAR sectors, TRAIN sample only) ===
    # Usiamo le 5 serie settoriali e il medesimo test_size_ratio
    sector_data = ts[VAR_SECTOR_COLS].dropna()

    # stessa logica di split degli univariati: usiamo solo il train per Granger
    h_gc = max(1, int(round(len(sector_data) * TEST_SIZE_RATIO)))
    gc_train = sector_data.iloc[:-h_gc]

    print("\n[Granger] Running Granger causality analysis on VAR sectors (train sample)...")
    gc_pvalues = granger_causality_matrix(
        data=gc_train,
        maxlags=4,     # coerente con il maxlags usato nei VAR
        alpha=0.05,
        ic="aic",
        verbose=True,
    )

    gc_path = out / "granger_causality_pvalues.csv"
    gc_pvalues.to_csv(gc_path, float_format="%.4f")
    print("Saved:", gc_path.resolve())

if __name__ == '__main__':
    run()
