import pandas as pd
from typing import Tuple, List, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def seasonal_naive(y: pd.Series, season_lag=12) -> pd.Series:
    return y.shift(season_lag).dropna()

def forecast_rw_recursive(y: pd.Series, include_drift: bool = True) -> pd.Series:
    """Random Walk con/ senza drift via media differenze in-sample, rollando out-of-sample."""
    y_tr = y.iloc[:-int(round(len(y)*0.2))]
    y_te = y.iloc[-int(round(len(y)*0.2)):]
    preds, h = [], y_tr.copy()
    for t in y_te.index:
        pred = h.iloc[-1] + h.diff().dropna().mean() if include_drift else h.iloc[-1]
        preds.append((t, pred)); h.loc[t] = y.loc[t]
    return pd.Series(dict(preds), name=y.name)

def forecast_arima_recursive(y: pd.Series,
                             order_grid: List[Tuple[int,int,int]]) -> Tuple[pd.Series, Tuple[int,int,int]]:
    y_tr = y.iloc[:-int(round(len(y)*0.2))]
    y_te = y.iloc[-int(round(len(y)*0.2)):]
    best, best_aic = None, float('inf')
    for order in order_grid:
        try:
            aic = ARIMA(y_tr, order=order).fit().aic
            if aic < best_aic: best, best_aic = order, aic
        except Exception:
            pass
    preds, h = [], y_tr.copy()
    for t in y_te.index:
        fc = ARIMA(h, order=best).fit().get_forecast(1).predicted_mean.iloc[0]
        preds.append((t, fc)); h.loc[t] = y.loc[t]
    return pd.Series(dict(preds), name=y.name), best

def forecast_sarima_recursive(y: pd.Series,
                              orders=((1,1,1),(2,1,1),(0,1,1)),
                              seas=((1,1,1,12),(0,1,1,12))) -> Tuple[pd.Series, Tuple, Tuple]:
    y_tr = y.iloc[:-int(round(len(y)*0.2))]
    y_te = y.iloc[-int(round(len(y)*0.2)):]
    best_s, best_S, best_aic = None, None, float('inf')
    for o in orders:
        for S in seas:
            try:
                aic = SARIMAX(y_tr, order=o, seasonal_order=S,
                              enforce_stationarity=False, enforce_invertibility=False
                             ).fit(disp=False).aic
                if aic < best_aic: best_s, best_S, best_aic = o, S, aic
            except Exception:
                pass
    preds, h = [], y_tr.copy()
    for t in y_te.index:
        fc = SARIMAX(h, order=best_s, seasonal_order=best_S,
                     enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False).get_forecast(1).predicted_mean.iloc[0]
        preds.append((t, fc)); h.loc[t] = y.loc[t]
    return pd.Series(dict(preds), name=y.name), best_s, best_S