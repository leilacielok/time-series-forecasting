from typing import Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def rmsfe(y, yhat) -> float:
    return float(np.sqrt(((y - yhat)**2).mean()))

def mafe(y, yhat) -> float:
    return float((y - yhat).abs().mean())

def summarize_series(y_raw: pd.Series,
                     yhat_dict: Dict[str, pd.Series],
                     specs: Dict[str, str],
                     logged: bool) -> Dict[str, object]:
    # assume che gli yhat abbiano lo stesso index del test
    row = {'Series': y_raw.name, 'log_transform': logged}
    for name, yhat in yhat_dict.items():
        row[f'MAFE_{name}'] = mafe(y_raw.loc[yhat.index], yhat)
        row[f'RMSFE_{name}'] = rmsfe(y_raw.loc[yhat.index], yhat)
    row.update(specs)
    return row

# Granger causality matrix for all variables in a VAR
def granger_causality_matrix(
    data: pd.DataFrame,
    maxlags: int = 4,
    alpha: float = 0.05,
    ic: str = "aic",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise Granger causality tests for all variables in 'data'
    using a VAR model.

    Parameters
    ----------
    data : pd.DataFrame
        Time series dataframe, columns = variables, index = dates.
        (Usa tipicamente il TRAIN set, non l'intero campione.)
    maxlags : int, optional
        Maximum lag order for VAR lag selection (if ic is not None).
    alpha : float, optional
        Significance level used to flag Granger causality.
    ic : {"aic","bic",None}, optional
        Information criterion for lag selection. If None, use maxlags.
    verbose : bool, optional
        If True, prints a small summary of significant relations.

    Returns
    -------
    pvals : pd.DataFrame
        Matrix of p-values, where entry (i, j) is the p-value for the test:
        H0: column j does NOT Granger cause row i.
    """
    # no NaN
    data = data.dropna()

    # Var estimation + lag choice
    model = VAR(data)

    if ic is not None:
        results = model.fit(maxlags=maxlags, ic=ic)
    else:
        results = model.fit(maxlags)  # fixed lag

    p = results.k_ar
    variables = list(data.columns)

    pvals = pd.DataFrame(
        np.nan,
        index=variables,
        columns=variables,
    )

    # Loop on all couples (ordered)
    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue
            test_res = results.test_causality(
                caused,
                causing,   # H0: causing does NOT Granger-cause caused
                kind="f"
            )
            pvals.loc[caused, causing] = test_res.pvalue

    if verbose:
        print(f"[Granger] Selected VAR lag order (ic={ic}): p = {p}")
        print(f"[Granger] Significance level alpha = {alpha}\n")
        for caused in variables:
            for causing in variables:
                if caused == causing:
                    continue
                pvalue = pvals.loc[caused, causing]
                if pvalue is not None and pvalue < alpha:
                    print(f"  {causing}  →  {caused}  (p = {pvalue:.4f})")

    return pvals
