# 📈 Stock Analysis AI Assistant

An AI-powered stock analysis tool that combines **technical indicators**, **news sentiment analysis**, and **machine learning predictions** to provide actionable insights on any publicly traded stock.

Built as a solo project for **BMA5278 — AI and Analytics in Business Practice**.

## Live Demo

🔗 [https://stock-analysis-ai.streamlit.app](https://stock-analysis-ai.streamlit.app)

## Features

- **AI-Powered Analysis** — LLM-generated market outlook combining all signals into a decisive recommendation
- **Technical Indicators** — SMA (20/50), RSI, MACD, Bollinger Bands with plain-English interpretations
- **News Sentiment** — Real-time headline scraping scored by AI on a -1 to +1 scale
- **ML Prediction** — GradientBoosting classifier predicting UP/DOWN direction with confidence scores and AUC-ROC evaluation
- **Interactive Charts** — Candlestick + RSI + MACD panels via Plotly
- **Combined Overall Signal** — Weighted aggregation of technicals, sentiment, and ML into a single bullish/bearish indicator
- **Explainable Predictions** — Feature importance and human-readable reasons for each prediction

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B) via OpenAI-compatible API |
| Stock Data | yfinance |
| Technical Analysis | `ta` library |
| ML Model | scikit-learn (GradientBoostingClassifier) |
| Charts | Plotly |
| News | Google News RSS via feedparser |

## Project Structure

```
├── app.py                  # Main Streamlit application
├── config.py               # Configuration and secrets management
├── data_fetcher.py         # Stock price & news data retrieval
├── technical_analysis.py   # Technical indicator computation
├── sentiment.py            # LLM-based news sentiment scoring
├── llm_engine.py           # AI analysis generation & chat
├── ml_predictor.py         # ML prediction with explainability
├── charts.py               # Interactive Plotly charts
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── .gitignore              # Git ignore rules
```

## Setup (Local)

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-analysis-ai.git
   cd stock-analysis-ai
   ```

2. **Install dependencies** (Python 3.10–3.13 recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key (free at https://console.groq.com)
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## How It Works

1. Enter a stock ticker (e.g., AAPL, TSLA, MU)
2. The app fetches 1 year of price data and latest news headlines
3. Technical indicators are computed and interpreted
4. News headlines are scored for sentiment by the LLM
5. A GradientBoosting model is trained on historical features to predict direction
6. All signals are combined into an overall assessment
7. The LLM generates a comprehensive analysis explaining its outlook

## API Key

This app uses [Groq](https://console.groq.com) for LLM inference (free tier). Sign up and get your API key, then add it to `.env` or Streamlit Cloud secrets.

## License

MIT
