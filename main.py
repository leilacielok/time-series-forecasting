from pathlib import Path
import pandas as pd

from forecasting.preprocessing import (load_csv, build_month_start_date, normalize_sectors,
                                       transportation_from_biofuels, pivot_sector_timeseries,
                                       trim_period, detect_numeric_targets,
                                       maybe_log_transform, train_test_split_ts)
from forecasting.models import (seasonal_naive, forecast_rw_recursive,
                                forecast_arima_recursive, forecast_sarima_recursive)
from forecasting.evaluation import summarize_series

CSV_PATH = 'data/summary_1970_2022.csv'
ARIMA_ORDER_GRID = [(0,1,0),(1,1,0),(0,1,1),(1,1,1),(2,1,1)]

def run():
    df = load_csv(CSV_PATH)
    df = build_month_start_date(df)
    df = normalize_sectors(df)
    df = transportation_from_biofuels(df)

    ts = pivot_sector_timeseries(df)
    ts = trim_period(ts, start='1995-01-01')

    targets = detect_numeric_targets(ts)
    rows = []

    for col in targets:
        y_raw = ts[col].dropna()
        y_t, logged = maybe_log_transform(y_raw)

        # split
        h = max(1, int(round(len(y_t)*0.2)))
        test_index = y_t.index[-h:]

        # s-naive
        yhat_s = seasonal_naive(y_t).loc[test_index]

        # RW
        yhat_rw = forecast_rw_recursive(y_t).loc[test_index]

        # ARIMA
        yhat_ar, spec_ar = forecast_arima_recursive(y_t, ARIMA_ORDER_GRID)
        yhat_ar = yhat_ar.loc[test_index]

        # SARIMA
        yhat_sar, spec_s, spec_S = forecast_sarima_recursive(y_t)
        yhat_sar = yhat_sar.loc[test_index]

        # summary row
        rows.append(summarize_series(
            y_raw,
            {'sNaive': yhat_s, 'RW': yhat_rw, 'ARIMA': yhat_ar, 'SARIMA': yhat_sar},
            {'Spec_ARIMA': str(spec_ar), 'Spec_SARIMA': f'{spec_s} x {spec_S}'},
            logged
        ))

    summary_df = pd.DataFrame(rows).set_index('Series').sort_values('RMSFE_SARIMA')
    out = Path('outputs'); out.mkdir(exist_ok=True)
    summary_df.to_csv(out / 'forecast_summary.csv', float_format='%.3f')
    print('Saved:', (out / 'forecast_summary.csv').resolve())

if __name__ == '__main__':
    run()