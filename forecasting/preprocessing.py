import pandas as pd
import numpy as np
from typing import Tuple, List

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_month_start_date(df: pd.DataFrame, year='Year', month='Month') -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime({'year': df[year].astype(int),
                                 'month': df[month].astype(int),
                                 'day': 1})
    return df

def normalize_sectors(df: pd.DataFrame, col='Sector') -> pd.DataFrame:
    df = df.copy()
    df[col] = (df[col]
               .str.strip()
               .str.replace('Commerical', 'Commercial', regex=False))
    return df

def transportation_from_biofuels(df: pd.DataFrame) -> pd.DataFrame:
    """Sostituisce i valori di Transportation con somma biocarburanti del mese."""
    df = df.copy()
    bio_cols = ['Biodiesel', 'Other Biofuels', 'Renewable Diesel Fuel', 'Ethanol, Excluding Denaturants']
    df['bio_sum'] = df[bio_cols].sum(axis=1)
    mask = df['Sector'].eq('Transportation')
    df.loc[mask, 'Total Renewable Energy'] = df.loc[mask, 'bio_sum']
    df.drop(columns=['bio_sum'], inplace=True)
    return df

def pivot_sector_timeseries(df: pd.DataFrame,
                            value_col='Total Renewable Energy',
                            sector_col='Sector',
                            date_col='date') -> pd.DataFrame:
    ts = (df.pivot_table(index=date_col, columns=sector_col, values=value_col, aggfunc='sum')
            .asfreq('MS')
            .sort_index())
    return ts

def trim_period(ts: pd.DataFrame, start='1995-01-01') -> pd.DataFrame:
    return ts.loc[start:].copy()

def train_test_split_ts(y: pd.Series, test_ratio=0.2) -> Tuple[pd.Series, pd.Series]:
    n = len(y); h = max(1, int(round(n*test_ratio)))
    return y.iloc[:-h], y.iloc[-h:]

def maybe_log_transform(y: pd.Series, force=True) -> Tuple[pd.Series, bool]:
    if force and (y > 0).all():
        return np.log(y), True
    return y.copy(), False

def invert_log_transform(yhat: pd.Series, was_logged: bool) -> pd.Series:
    return np.exp(yhat) if was_logged else yhat

def detect_numeric_targets(ts: pd.DataFrame) -> List[str]:
    return ts.select_dtypes('number').columns.tolist()
