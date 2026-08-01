# 📈 Stock Analysis AI Assistant

An AI-powered stock analysis tool that combines **technical indicators** and **news sentiment analysis** to provide actionable insights on any publicly traded stock.

Built as a solo project for **BMA5278 — AI and Analytics in Business Practice**.

## Live Demo

🔗 [https://stock-ai-analysis.streamlit.app](https://stock-ai-analysis.streamlit.app)

## Features

- **AI-Powered Analysis** — LLM-generated market outlook combining all signals into a decisive recommendation
- **Technical Indicators** — SMA (20/50), RSI, MACD, Bollinger Bands with plain-English interpretations
- **News Sentiment** — Real-time headline scraping scored by AI on a -1 to +1 scale
- **Interactive Charts** — 3-panel candlestick + RSI + MACD via Plotly
- **Combined Overall Signal** — Weighted aggregation of technicals and sentiment into a single bullish/bearish indicator
- **Follow-up Chat** — Ask the AI follow-up questions about the analysis
- **Cloud-Ready** — Handles Yahoo Finance data quirks on cloud servers via real-time price fallback

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit (deployed on Streamlit Community Cloud) |
| LLM | Groq (Llama 3.3 70B) via OpenAI-compatible API |
| Stock Data | yfinance (with `fast_info` real-time fallback) |
| Technical Analysis | `ta` library |
| Charts | Plotly |
| News | Google News RSS via feedparser |

## Project Structure

```
├── app.py                  # Main Streamlit application
├── config.py               # Configuration and secrets management
├── data_fetcher.py         # Stock price & news data retrieval (with cloud fixes)
├── technical_analysis.py   # Technical indicator computation
├── sentiment.py            # LLM-based news sentiment scoring
├── llm_engine.py           # AI analysis generation & follow-up chat
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

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set main file to `app.py`
4. In **Advanced settings**, add your secret: `GROQ_API_KEY = "your-key-here"`
5. Set Python version to **3.12**
6. Click **Deploy** — you'll get a shareable URL

## How It Works

1. Enter a stock ticker (e.g., AAPL, TSLA, MU)
2. The app fetches 1 year of price data and latest news headlines
3. Technical indicators are computed and interpreted
4. News headlines are scored for sentiment by the LLM
5. Technicals and sentiment are combined into an overall signal
6. The LLM generates a comprehensive analysis explaining its outlook
7. Ask follow-up questions via the built-in chat

## API Key

This app uses [Groq](https://console.groq.com) for LLM inference (free tier). Sign up and get your API key, then add it to `.env` locally or Streamlit Cloud secrets for deployment.

## License

MIT
