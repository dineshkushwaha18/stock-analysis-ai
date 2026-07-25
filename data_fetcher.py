"""
Data fetching module — retrieves stock price data and news headlines.
"""

import yfinance as yf
import pandas as pd
import feedparser
from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, MAX_NEWS_HEADLINES


def get_stock_data(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Returns a DataFrame with OHLCV data, or empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock info (name, sector, market cap, etc.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "industry": "N/A",
                "market_cap": "N/A", "currency": "USD", "exchange": "N/A"}


def get_news_headlines(ticker: str, max_headlines: int = MAX_NEWS_HEADLINES) -> list[dict]:
    """
    Fetch recent news headlines for a stock using Google News RSS.
    Returns list of dicts with 'title', 'link', 'published'.
    """
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:max_headlines]:
            headlines.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        return headlines
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []
