"""
Data fetching module — retrieves stock price data and news headlines.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
import feedparser
from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, MAX_NEWS_HEADLINES

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# US market timezone — NYSE/NASDAQ operate in Eastern Time
_MARKET_TZ = ZoneInfo("America/New_York")

# Map period strings to approximate days for explicit date range
_PERIOD_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
}


def get_stock_data(ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Uses yf.download() with market-timezone-aware dates.
    Returns a DataFrame with OHLCV data, or empty DataFrame on failure.
    """
    try:
        # Use the US market timezone to determine "today"
        market_now = datetime.now(_MARKET_TZ)
        end_date = market_now.date() + timedelta(days=1)  # tomorrow in market tz
        days = _PERIOD_DAYS.get(period, 365)
        start_date = end_date - timedelta(days=days)

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()

        # yf.download() returns MultiIndex columns for single tickers
        # e.g. ('Close', 'AAPL') — flatten to just 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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
