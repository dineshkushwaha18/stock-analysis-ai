"""
Technical analysis module — computes indicators from price data.
"""

import pandas as pd
import ta as ta_lib
from config import SHORT_MA, LONG_MA, RSI_PERIOD


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the price DataFrame.
    """
    if df.empty:
        return df

    # Moving Averages
    df[f"SMA_{SHORT_MA}"] = ta_lib.trend.sma_indicator(df["Close"], window=SHORT_MA)
    df[f"SMA_{LONG_MA}"] = ta_lib.trend.sma_indicator(df["Close"], window=LONG_MA)

    # RSI
    df["RSI"] = ta_lib.momentum.rsi(df["Close"], window=RSI_PERIOD)

    # MACD
    macd = ta_lib.trend.MACD(df["Close"])
    df["MACD_12_26_9"] = macd.macd()
    df["MACDs_12_26_9"] = macd.macd_signal()
    df["MACDh_12_26_9"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta_lib.volatility.BollingerBands(df["Close"], window=20)
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBL_20_2.0"] = bb.bollinger_lband()

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    """
    Extract the latest indicator values and generate simple signals.
    Returns a dict summarising current technical state.
    """
    if df.empty:
        return {"error": "No data available"}

    if len(df) < 2:
        return {"error": "Insufficient data for analysis"}

    # Drop rows where Close is NaN (can happen with yfinance)
    df_clean = df.dropna(subset=["Close"])
    if len(df_clean) < 2:
        return {"error": "Insufficient data for analysis"}

    latest = df_clean.iloc[-1]
    prev = df_clean.iloc[-2]

    close = latest["Close"]
    sma_short = latest.get(f"SMA_{SHORT_MA}")
    sma_long = latest.get(f"SMA_{LONG_MA}")
    rsi = latest.get("RSI")
    macd_val = latest.get("MACD_12_26_9")
    macd_signal = latest.get("MACDs_12_26_9")

    signals = {
        "price": round(close, 2),
        "sma_short": round(sma_short, 2) if pd.notna(sma_short) else None,
        "sma_long": round(sma_long, 2) if pd.notna(sma_long) else None,
        "rsi": round(rsi, 2) if pd.notna(rsi) else None,
        "macd": round(macd_val, 4) if pd.notna(macd_val) else None,
        "macd_signal": round(macd_signal, 4) if pd.notna(macd_signal) else None,
    }

    # Interpret signals
    interpretations = []

    # Moving average crossover
    if sma_short and sma_long:
        if sma_short > sma_long:
            interpretations.append("Short-term MA is ABOVE long-term MA (bullish trend)")
        else:
            interpretations.append("Short-term MA is BELOW long-term MA (bearish trend)")

    # Price vs MAs
    if sma_short:
        if close > sma_short:
            interpretations.append("Price is above short-term MA (bullish)")
        else:
            interpretations.append("Price is below short-term MA (bearish)")

    # RSI
    if rsi:
        if rsi > 70:
            interpretations.append(f"RSI is {rsi:.1f} — OVERBOUGHT (potential reversal down)")
        elif rsi < 30:
            interpretations.append(f"RSI is {rsi:.1f} — OVERSOLD (potential reversal up)")
        else:
            interpretations.append(f"RSI is {rsi:.1f} — neutral range")

    # MACD
    if pd.notna(macd_val) and pd.notna(macd_signal):
        if macd_val > macd_signal:
            interpretations.append("MACD is above signal line (bullish momentum)")
        else:
            interpretations.append("MACD is below signal line (bearish momentum)")

    signals["interpretations"] = interpretations
    return signals
