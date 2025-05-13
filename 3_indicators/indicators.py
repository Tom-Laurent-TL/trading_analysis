import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy
from pykalman import KalmanFilter

def add_sma(df: pd.DataFrame, period: int, column: str = "Close", sma_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne SMA (Simple Moving Average) au DataFrame.
    """
    if sma_col is None:
        sma_col = f"SMA_{period}"
    df[sma_col] = df[column].rolling(window=period, min_periods=1).mean()
    return df

def add_ema(df: pd.DataFrame, period: int, column: str = "Close", ema_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne EMA (Exponential Moving Average) au DataFrame.
    """
    if ema_col is None:
        ema_col = f"EMA_{period}"
    df[ema_col] = df[column].ewm(span=period, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close", rsi_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne RSI (Relative Strength Index) au DataFrame.
    """
    if rsi_col is None:
        rsi_col = f"RSI_{period}"
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df[rsi_col] = rsi
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close") -> pd.DataFrame:
    """
    Ajoute les colonnes MACD, Signal et Histogram au DataFrame.
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    df[f"MACD_{fast}_{slow}"] = macd
    df[f"MACD_signal_{signal}"] = signal_line
    df[f"MACD_hist_{fast}_{slow}_{signal}"] = hist
    return df

def add_shannon_entropy(df: pd.DataFrame, period: int, column: str = "Close", entropy_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne Shannon Entropy au DataFrame.
    """
    if entropy_col is None:
        entropy_col = f"ShannonEntropy_{period}"
    def rolling_entropy(x):
        # Histogram with 10 bins, density=True for probability
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
        return shannon_entropy(hist, base=2)
    df[entropy_col] = df[column].rolling(window=period, min_periods=period).apply(rolling_entropy, raw=True)
    return df

def add_permutation_entropy(df: pd.DataFrame, period: int, order: int = 3, column: str = "Close", entropy_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne Permutation Entropy au DataFrame.
    """
    if entropy_col is None:
        entropy_col = f"PermutationEntropy_{period}_{order}"
    def permutation_entropy(x, order):
        # Generate all possible permutations
        from math import factorial
        n = len(x)
        if n < order:
            return np.nan
        perms = {}
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(x[i:i+order]))
            perms[pattern] = perms.get(pattern, 0) + 1
        counts = np.array(list(perms.values()), dtype=float)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))
    df[entropy_col] = df[column].rolling(window=period, min_periods=period).apply(lambda x: permutation_entropy(x, order), raw=True)
    return df


def add_rolling_mean(df: pd.DataFrame, col, window, new_col=None):
    """
    Add rolling mean column.
    """
    if new_col is None:
        new_col = f"{col}_rollmean_{window}"
    df[new_col] = df[col].rolling(window=window).mean()
    return df

def add_rolling_std(df: pd.DataFrame, col, window, new_col=None):
    """
    Add rolling std column.
    """
    if new_col is None:
        new_col = f"{col}_rollstd_{window}"
    df[new_col] = df[col].rolling(window=window).std()
    return df

def add_lag(df: pd.DataFrame, col, lags=1, new_col=None):
    """
    Add lagged feature column(s).
    """
    if isinstance(lags, int):
        lags = [lags]
    for lag in lags:
        colname = new_col or f"{col}_lag_{lag}"
        df[colname] = df[col].shift(lag)
    return df

def z_score_normalize(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Apply z-score normalization to specified columns.
    """
    df_normalized = df.copy()
    if cols is None:
        cols = df.columns
    for col in cols:
        df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
    return df_normalized

def add_rolling_z_normalize(df: pd.DataFrame, col, window, new_col=None) -> pd.DataFrame:
    """
    Add rolling z-score normalization column.
    """
    if new_col is None:
        new_col = f"{col}_rollz_{window}"
    rolling_mean = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    df[new_col] = (df[col] - rolling_mean) / rolling_std
    return df

def apply_kalman_filter(df: pd.DataFrame, col: str, new_col: str = None) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth a column.
    """
    if new_col is None:
        new_col = f"{col}_kalman"
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(df[col].dropna(), n_iter=10)
    state_means, _ = kf.filter(df[col].fillna(0).values)
    df[new_col] = state_means[:, 0]
    return df

def add_exponential_moving_average(df: pd.DataFrame, col: str, span: int, new_col: str = None) -> pd.DataFrame:
    """
    Add exponential moving average (EMA) column.
    """
    if new_col is None:
        new_col = f"{col}_ema_{span}"
    df[new_col] = df[col].ewm(span=span, adjust=False).mean()
    return df

def add_momentum(df: pd.DataFrame, col: str, period: int, new_col: str = None) -> pd.DataFrame:
    """
    Add momentum indicator column.
    """
    if new_col is None:
        new_col = f"{col}_momentum_{period}"
    df[new_col] = df[col] - df[col].shift(period)
    return df

def add_bollinger_bands(df: pd.DataFrame, col: str, window: int, num_std: float = 2.0, upper_col: str = None, lower_col: str = None) -> pd.DataFrame:
    """
    Add Bollinger Bands columns (upper and lower bands).
    """
    if upper_col is None:
        upper_col = f"{col}_bollinger_upper_{window}"
    if lower_col is None:
        lower_col = f"{col}_bollinger_lower_{window}"
    rolling_mean = df[col].rolling(window=window).mean()
    rolling_std = df[col].rolling(window=window).std()
    df[upper_col] = rolling_mean + (rolling_std * num_std)
    df[lower_col] = rolling_mean - (rolling_std * num_std)
    return df