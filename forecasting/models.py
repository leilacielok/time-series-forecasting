# Benchmarks: RW, ARIMA, SARIMA

from typing import Tuple, List
import pandas as pd
import numpy as np
from forecasting.preprocessing import maybe_log_transform, invert_log_transform
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX


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

def forecast_arima_recursive(y: pd.Series, order_grid: List[Tuple[int,int,int]]) -> Tuple[pd.Series, Tuple[int,int,int]]:
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

## Model to test: VAR
from statsmodels.tsa.api import VAR


def forecast_var_recursive(
    ts: pd.DataFrame,
    target_col: str,
    var_cols: list[str],
    test_size_ratio: float = 0.2,
    maxlags: int = 4,
):
    """
    Stima un VAR(p) sulle colonne `var_cols` (in log) e produce
    previsioni 1-step-ahead ricorsive per il periodo di test.
    Ritorna:
        y_test (livelli originali),
        y_hat_lvl (previsioni in livelli per la sola colonna target_col).
        :type test_size_ratio: object
    """

    # 1) Seleziona solo le colonne del sistema VAR
    df = ts[var_cols].copy()

    # 2) Log-transform per tutte le serie, usando maybe_log_transform
    df_log = pd.DataFrame(index=df.index)
    log_flags = {}

    for col in var_cols:
        y_raw = df[col].dropna()
        y_log, logged = maybe_log_transform(y_raw)

        # riallinea sull'indice completo (qui assumiamo niente buchi seri)
        df_log[col] = y_log
        log_flags[col] = logged

    # 3) Split train/test in base al rapporto scelto
    n = len(df_log)
    n_test = int(np.floor(n * test_size_ratio))
    n_train = n - n_test

    train = df_log.iloc[:n_train]
    test = df_log.iloc[n_train:]

    # 4) Stima VAR sul train
    model = VAR(train)
    # usa maxlags come limite e lascia scegliere il p ottimale con AIC
    results = model.fit(maxlags=maxlags, ic="aic")
    p = results.k_ar

    # 5) Previsioni 1-step-ahead ricorsive sul periodo di test
    history = train.values.copy()
    forecasts_log = []

    for i in range(len(test)):
        # previsioni in log per tutte le serie nel sistema
        pred = results.forecast(y=history[-p:], steps=1)[0]
        forecasts_log.append(pred)

        # aggiorna la "storia" con il valore OSSERVATO (test) al tempo t
        history = np.vstack([history, test.values[i]])

    forecasts_log = np.array(forecasts_log)
    forecasts_log_df = pd.DataFrame(
        forecasts_log,
        index=test.index,
        columns=var_cols,
    )

    # 6) Recupera y_test in livelli e converte le previsioni del target
    y_test = ts[target_col].reindex(test.index)

    y_hat_log = forecasts_log_df[target_col]
    y_hat_lvl = invert_log_transform(y_hat_log, log_flags[target_col])

    return y_test, y_hat_lvl


# Model to test: VAR-X
def forecast_varx_recursive(
    ts: pd.DataFrame,
    exog: pd.DataFrame,
    target_col: str,
    var_cols: list[str],
    test_size_ratio: float = 0.2,
    maxlags: int = 1,
):
    """
    Stima un VAR-X(p) usando VARMAX (order=(p,0)) sulle colonne `var_cols`
    (endogene, in log) con regressori esogeni `exog` (già trasformati, es. log prezzi),
    e produce previsioni out-of-sample per il periodo di test (1-step ahead
    su tutto l'orizzonte) per la sola colonna target_col.

    maxlags qui viene interpretato direttamente come p (ordine del VAR-X).
    """

    # 1) Endogene: solo le colonne del sistema VAR
    df = ts[var_cols].copy()

    # 2) Log-transform per tutte le serie endogene
    df_log = pd.DataFrame(index=df.index)
    log_flags = {}
    for col in var_cols:
        y_raw = df[col].dropna()
        y_log, logged = maybe_log_transform(y_raw)
        df_log[col] = y_log
        log_flags[col] = logged

    # 3) Allinea exog e endog (per sicurezza)
    exog = exog.loc[df_log.index]

    # 4) Split train/test
    n = len(df_log)
    n_test = int(np.floor(n * test_size_ratio))
    n_train = n - n_test

    endog_train = df_log.iloc[:n_train]
    endog_test  = df_log.iloc[n_train:]
    exog_train  = exog.iloc[:n_train]
    exog_test   = exog.iloc[n_train:]

    # 5) Stima VAR-X(p) con VARMAX (order=(p, 0))
    p = maxlags  # interpretiamo maxlags come ordine p
    model = VARMAX(endog_train, exog=exog_train, order=(p, 0), trend="c")
    results = model.fit(disp=False)

    # 6) Previsioni out-of-sample per tutto il periodo di test
    # (usa storia di train e exog_test per generare forecast dinamici)
    forecast_log = results.forecast(steps=len(endog_test), exog=exog_test)

    # DataFrame con stesse colonne/indice
    forecast_log_df = pd.DataFrame(
        forecast_log,
        index=endog_test.index,
        columns=var_cols,
    )

    # 7) Torna ai livelli per il target
    y_test = ts[target_col].reindex(endog_test.index)
    y_hat_log = forecast_log_df[target_col]
    y_hat_lvl = invert_log_transform(y_hat_log, log_flags[target_col])

    return y_test, y_hat_lvl

def forecast_varx_wti_recursive(
    ts: pd.DataFrame,
    exog: pd.DataFrame,
    target_col: str,
    var_cols: list[str],
    test_size_ratio: float = 0.2,
    maxlags: int = 1,
):
    """
    Convenience wrapper around `forecast_varx_recursive` that uses only
    the `log_WTI` column as exogenous regressor.

    Parameters
    ----------
    ts : DataFrame
        Endogenous series (sectors), same structure as for `forecast_varx_recursive`.
    exog : DataFrame
        DataFrame containing at least a 'log_WTI' column. Any additional
        columns are ignored.
    target_col : str
        Name of the target series (one of `var_cols`).
    var_cols : list of str
        List of endogenous columns in the VAR-X system.
    test_size_ratio : float
        Fraction of the sample used for the test set.
    maxlags : int
        VAR order p (interpreted as in `forecast_varx_recursive`).

    Returns
    -------
    y_test : Series
        Realised values in levels for the target series.
    y_hat_lvl : Series
        Forecasts in levels for the target series.
    """
    # keep only log_WTI, even if exog has more columns
    exog_wti = exog[['log_WTI']].copy()

    return forecast_varx_recursive(
        ts=ts,
        exog=exog_wti,
        target_col=target_col,
        var_cols=var_cols,
        test_size_ratio=test_size_ratio,
        maxlags=maxlags,
    )
