import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

def plot_stl(y: pd.Series, period=12, title=None):
    res = STL(y, period=period).fit()
    res.plot()
    if title: plt.suptitle(title);
    plt.show()

def quick_plot(ts: pd.DataFrame, cols=None, title=None):
    ts[cols or ts.columns].plot(figsize=(10,4))
    if title: plt.title(title)
    plt.tight_layout(); plt.show()