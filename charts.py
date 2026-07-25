"""
Chart generation module — creates interactive Plotly charts.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from config import SHORT_MA, LONG_MA


def create_stock_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Create an interactive candlestick chart with technical indicators.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    # Create subplots: price on top, RSI + MACD below
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "RSI", "MACD"),
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1, col=1,
    )

    # Moving Averages
    sma_short_col = f"SMA_{SHORT_MA}"
    sma_long_col = f"SMA_{LONG_MA}"

    if sma_short_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[sma_short_col], name=f"SMA {SHORT_MA}",
                       line=dict(color="orange", width=1.5)),
            row=1, col=1,
        )

    if sma_long_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[sma_long_col], name=f"SMA {LONG_MA}",
                       line=dict(color="blue", width=1.5)),
            row=1, col=1,
        )

    # Bollinger Bands
    if "BBU_20_2.0" in df.columns and "BBL_20_2.0" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BBU_20_2.0"], name="BB Upper",
                       line=dict(color="gray", width=1, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BBL_20_2.0"], name="BB Lower",
                       line=dict(color="gray", width=1, dash="dash"),
                       fill="tonexty", fillcolor="rgba(128,128,128,0.1)"),
            row=1, col=1,
        )

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                       line=dict(color="purple", width=1.5)),
            row=2, col=1,
        )
        # Overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if "MACD_12_26_9" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_12_26_9"], name="MACD",
                       line=dict(color="blue", width=1.5)),
            row=3, col=1,
        )
    if "MACDs_12_26_9" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACDs_12_26_9"], name="Signal",
                       line=dict(color="red", width=1.5)),
            row=3, col=1,
        )
    if "MACDh_12_26_9" in df.columns:
        colors = ["green" if v >= 0 else "red" for v in df["MACDh_12_26_9"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACDh_12_26_9"], name="Histogram",
                   marker_color=colors),
            row=3, col=1,
        )

    # Layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
