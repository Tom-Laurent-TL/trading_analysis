import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def add_return(df: pd.DataFrame, period: int = 1, col: str = "Close", ret_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne de rendement (return) sur 'period' périodes.
    """
    if ret_col is None:
        ret_col = f"Return_{period}"
    df[ret_col] = df[col].pct_change(periods=period)
    return df

def add_direction(df: pd.DataFrame, period: int = 1, col: str = "Close", dir_col: str = None) -> pd.DataFrame:
    """
    Ajoute une colonne direction (1 si hausse, 0 sinon) sur 'period' périodes.
    """
    if dir_col is None:
        dir_col = f"Direction_{period}"
    df[dir_col] = (df[col].shift(-period) > df[col]).astype(int)
    return df

def train_test_split_df(df: pd.DataFrame, feature_cols, target_col, test_size=0.2, shuffle=False):
    """
    Sépare le DataFrame en X_train, X_test, y_train, y_test.
    """
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

def fill_missing(df: pd.DataFrame, cols=None, method='ffill', value=None):
    """
    Fill missing values in specified columns using method or value.
    """
    df_filled = df.copy()
    if cols is None:
        cols = df.columns
    if value is not None:
        df_filled[cols] = df_filled[cols].fillna(value)
    else:
        df_filled[cols] = df_filled[cols].fillna(method=method)
    return df_filled

def remove_outliers(df: pd.DataFrame, cols, z_thresh=3):
    """
    Remove rows where specified columns have z-score > z_thresh.
    """
    df_out = df.copy()
    for col in cols:
        z = np.abs((df_out[col] - df_out[col].mean()) / df_out[col].std())
        df_out = df_out[z < z_thresh]
    return df_out