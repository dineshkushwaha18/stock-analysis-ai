"""
Stock Analysis AI Assistant — Streamlit Application
BMA5278 AI and Analytics in Business Practice
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from data_fetcher import get_stock_data, get_stock_info, get_news_headlines
from technical_analysis import compute_indicators, get_latest_signals
from sentiment import analyze_sentiment
from llm_engine import generate_analysis, chat_followup
from charts import create_stock_chart
from ml_predictor import train_and_evaluate

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analysis AI Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Analysis AI Assistant")
st.caption("AI-powered technical & sentiment analysis with explainable insights")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g. AAPL, TSLA, MSFT").upper().strip()
    prediction_horizon = st.selectbox("Prediction Horizon", [1, 5, 10], index=1,
                                       format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
                                       help="How many trading days ahead to predict")
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("""
    1. Enter a stock ticker symbol
    2. Click **Analyze**
    3. Explore the tabs for insights
    4. Ask follow-up questions in the chat
    """)
    st.divider()
    st.caption("⚠️ This is not financial advice. For educational purposes only.")

# ─── Session State ────────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None

# ─── Main Analysis ────────────────────────────────────────────────────────────
if analyze_btn and ticker:
    st.session_state.messages = []  # Reset chat on new analysis
    st.session_state.current_ticker = ticker

    with st.spinner(f"Fetching data for {ticker}..."):
        # Step 1: Fetch data
        stock_info = get_stock_info(ticker)
        df = get_stock_data(ticker, period="1y")

        if df.empty:
            st.error(f"❌ Could not fetch data for '{ticker}'. Please check the ticker symbol.")
            st.stop()

        if len(df) < 50:
            st.warning(f"⚠️ Only {len(df)} days of data available for {ticker} (recently listed). Some indicators may be limited.")

        # Step 2: Compute technical indicators
        df = compute_indicators(df)
        signals = get_latest_signals(df)

        # Step 3: Fetch and analyze news sentiment
        headlines = get_news_headlines(ticker)
        sentiment = analyze_sentiment(headlines, ticker)

        # Step 3b: Train ML prediction model
        ml_result = train_and_evaluate(df, horizon=prediction_horizon)

        # Step 4: Generate AI analysis (graceful on LLM failure)
        try:
            analysis = generate_analysis(ticker, stock_info, signals, sentiment)
        except Exception as e:
            analysis = f"⚠️ AI analysis unavailable: {e}"

        # Store results
        st.session_state.analysis_result = {
            "ticker": ticker,
            "stock_info": stock_info,
            "df": df,
            "signals": signals,
            "sentiment": sentiment,
            "analysis": analysis,
            "headlines": headlines,
            "ml_result": ml_result,
        }

# ─── Display Results ──────────────────────────────────────────────────────────
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    ticker = result["ticker"]
    stock_info = result["stock_info"]
    df = result["df"]
    signals = result["signals"]
    sentiment = result["sentiment"]
    analysis = result["analysis"]
    headlines = result["headlines"]
    ml_result = result["ml_result"]

    # Compute combined overall signal from technicals + sentiment + ML
    def compute_overall_signal(signals, sentiment, ml_result):
        score = 0
        count = 0
        # Technical signals (each worth 1 point)
        for interp in signals.get('interpretations', []):
            if 'bullish' in interp.lower():
                score += 1
                count += 1
            elif 'bearish' in interp.lower():
                score -= 1
                count += 1
        # News sentiment (only if it didn't fail)
        sent_label = sentiment.get('label', '')
        if sent_label not in ('error', 'unavailable', ''):
            sent_score = sentiment.get('score', 0)
            score += sent_score * 2
            count += 2
        # ML prediction
        if ml_result and not ml_result.get('error'):
            ml_pred = ml_result['latest_prediction']
            ml_weight = (ml_pred['confidence'] / 100) * 2
            if ml_pred['direction'] == 'UP':
                score += ml_weight
            else:
                score -= ml_weight
            count += 2
        avg = score / count if count > 0 else 0
        if avg > 0.2:
            return '🟢 BULLISH', avg
        elif avg > 0.05:
            return '🟢 Slightly Bullish', avg
        elif avg > -0.05:
            return '🟡 Neutral', avg
        elif avg > -0.2:
            return '🔴 Slightly Bearish', avg
        else:
            return '🔴 BEARISH', avg

    overall_label, overall_score = compute_overall_signal(signals, sentiment, ml_result)

    # Header info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stock", f"{stock_info['name']}")
    with col2:
        price = signals.get('price', 'N/A')
        price_display = f"${price}" if price and price != 'N/A' and str(price) != 'nan' else "N/A"
        st.metric("Price", price_display)
    with col3:
        st.metric("Overall Signal", overall_label)
    with col4:
        rsi = signals.get("rsi")
        rsi_display = f"{rsi}" if rsi else "N/A"
        st.metric("RSI", rsi_display)
    data_date = signals.get('data_date', '')
    st.caption(f"📅 Price as of {data_date} · Based on last 1 year of trading data ({len(df)} trading days)")

    # Chart
    st.plotly_chart(create_stock_chart(df, ticker), use_container_width=True)

    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Analysis", "📰 News & Sentiment", "📊 Technical Signals", "🧠 ML Prediction"])

    with tab1:
        st.markdown(analysis)

    with tab2:
        st.subheader("Sentiment Summary")
        score = sentiment.get("score", 0)
        st.progress((score + 1) / 2, text=f"Sentiment Score: {score:.2f} ({sentiment.get('label', 'N/A')})")
        st.markdown(sentiment.get("summary", ""))

        st.subheader("Recent Headlines")
        for h in headlines:
            st.markdown(f"- [{h['title']}]({h['link']})")

    with tab3:
        st.subheader("Current Indicator Values")
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "Price": signals.get("price"),
                "SMA 20": signals.get("sma_short"),
                "SMA 50": signals.get("sma_long"),
                "RSI": signals.get("rsi"),
            })
        with col2:
            st.json({
                "MACD": signals.get("macd"),
                "MACD Signal": signals.get("macd_signal"),
            })

        st.subheader("Signal Interpretations")
        for interp in signals.get("interpretations", []):
            st.markdown(f"- {interp}")

    with tab4:
        if ml_result.get("error"):
            st.warning(ml_result["error"])
        else:
            # Prediction header
            pred = ml_result["latest_prediction"]
            direction_emoji = "🟢 UP" if pred["direction"] == "UP" else "🔴 DOWN"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Predicted Direction ({ml_result['prediction_horizon']}-day)", direction_emoji)
            with col2:
                st.metric("Confidence", f"{pred['confidence']}%")
            with col3:
                st.metric("Model AUC-ROC", f"{ml_result['auc_roc']:.4f}")

            st.divider()

            # Model performance
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model Performance")
                auc_label = '✅ Good' if ml_result['auc_roc'] > 0.55 else '⚠️ Weak signal'
                st.markdown(f"""
                - **AUC-ROC:** {ml_result['auc_roc']:.4f} {auc_label}
                - **Accuracy:** {ml_result['accuracy']:.1%}
                - **Training samples:** {ml_result['train_size']}
                - **Test samples:** {ml_result['test_size']}
                - **Prediction horizon:** {ml_result['prediction_horizon']} trading days
                """)

            with col2:
                st.subheader("Top Feature Importance")
                feat_df = pd.DataFrame({
                    "Feature": list(ml_result["feature_importance"].keys()),
                    "Importance": list(ml_result["feature_importance"].values()),
                })
                fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                             template="plotly_dark")
                fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            # Probability breakdown
            st.subheader("Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.progress(pred["up_probability"], text=f"📈 UP probability: {pred['up_probability']:.1%}")
            with prob_col2:
                st.progress(pred["down_probability"], text=f"📉 DOWN probability: {pred['down_probability']:.1%}")

            # Prediction explanation
            if pred.get("explanation"):
                st.divider()
                st.subheader(f"Why the model predicts {pred['direction']}")
                st.markdown("The top factors driving this prediction (ranked by importance):")
                for i, reason in enumerate(pred["explanation"], 1):
                    st.markdown(f"{i}. {reason}")

    # ─── Chat Interface ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask Follow-up Questions")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_question := st.chat_input(f"Ask anything about {ticker}..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_followup(ticker, analysis, user_question)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Landing state
    st.info("👈 Enter a stock ticker in the sidebar and click **Analyze** to get started.")
    st.markdown("""
    ### What this tool does:
    - **Technical Analysis**: Computes RSI, MACD, Moving Averages, Bollinger Bands
    - **Sentiment Analysis**: Scores recent news headlines for bullish/bearish sentiment
    - **ML Prediction**: Trained classifier predicts stock direction with AUC-ROC evaluation
    - **AI Reasoning**: Combines all signals to explain why a stock may move up or down
    - **Interactive Chat**: Ask follow-up questions about the analysis
    
    ### Example tickers to try:
    `AAPL` · `TSLA` · `MSFT` · `NVDA` · `GOOGL` · `AMZN` · `META`
    """)
