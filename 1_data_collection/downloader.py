import os
import requests
import pandas as pd
from typing import Optional
import yfinance as yf

def download_csv(url: str, dest_path: str, overwrite: bool = False) -> None:
    """
    Télécharge un fichier CSV depuis une URL et l'enregistre localement.
    """
    if not overwrite and os.path.exists(dest_path):
        return
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)

def load_csv(path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV local dans un DataFrame pandas.
    """
    return pd.read_csv(path)

def download_yahoo_finance(symbol: str, start: str, end: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Télécharge les données historiques d'un symbole depuis Yahoo Finance.
    start, end: format 'YYYY-MM-DD'
    interval: '1d', '1wk', '1mo'
    """
    df = yf.download(symbol, start=start, end=end, interval=interval)
    if df.empty:
        return None
    return df

def get_stock_info(symbol: str) -> Optional[dict]:
    """
    Récupère les informations générales d'une action (secteur, industrie, capitalisation, etc.) depuis Yahoo Finance.
    Retourne un dictionnaire ou None si non disponible.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info:
        return None
    return info

def download_news(symbol: str, count: int = 10) -> Optional[pd.DataFrame]:
    """
    Télécharge les dernières actualités concernant un symbole depuis Yahoo Finance.
    Retourne un DataFrame pandas contenant les titres et liens des actualités.
    """
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        news_items = data.get("news", [])
        news_list = []
        for item in news_items[:count]:
            news_list.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "publisher": item.get("publisher"),
                "providerPublishTime": item.get("providerPublishTime")
            })
        if not news_list:
            return None
        return pd.DataFrame(news_list)
    except Exception:
        return None

