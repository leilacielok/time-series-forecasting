import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

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