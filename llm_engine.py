"""
LLM reasoning module — generates the final analysis explanation.
"""

from openai import OpenAI
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL


def generate_analysis(ticker: str, stock_info: dict, signals: dict, sentiment: dict) -> str:
    """
    Use the LLM to generate a comprehensive, explainable stock analysis.
    Combines technical signals and news sentiment into a coherent narrative.
    """
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured. Please add your key to the `.env` file."

    # Build the context for the LLM
    interpretations = "\n".join([f"  • {i}" for i in signals.get("interpretations", [])])

    # Count bullish vs bearish signals for guidance
    bull_count = sum(1 for i in signals.get('interpretations', []) if 'bullish' in i.lower())
    bear_count = sum(1 for i in signals.get('interpretations', []) if 'bearish' in i.lower())
    sentiment_score = sentiment.get('score', 0)

    prompt = f"""You are an experienced stock analyst. Provide a DECISIVE analysis — take a clear position based on the data.

**Stock:** {stock_info.get('name', ticker)} ({ticker})
**Sector:** {stock_info.get('sector', 'N/A')} | **Industry:** {stock_info.get('industry', 'N/A')}

**Current Price:** ${signals.get('price', 'N/A')}
**20-day SMA:** ${signals.get('sma_short', 'N/A')}
**50-day SMA:** ${signals.get('sma_long', 'N/A')}
**RSI (14):** {signals.get('rsi', 'N/A')}
**MACD:** {signals.get('macd', 'N/A')} | **Signal:** {signals.get('macd_signal', 'N/A')}

**Technical Interpretations:**
{interpretations}

**Signal Summary:** {bull_count} bullish signals, {bear_count} bearish signals

**News Sentiment Score:** {sentiment_score:.2f} ({sentiment.get('label', 'N/A')})
**News Summary:** {sentiment.get('summary', 'No news available')}

IMPORTANT: Do NOT default to "neutral" unless the bullish and bearish signals are exactly equal AND news sentiment is near zero. The data above has clear signals — use them.

Based on the above data, provide:
1. **Overall Outlook** — State clearly: BULLISH, SLIGHTLY BULLISH, SLIGHTLY BEARISH, or BEARISH. Explain the primary reason in 1-2 sentences. Only say NEUTRAL if signals are genuinely split 50/50.
2. **Technical Analysis Summary** — What do the indicators tell us about current momentum and trend direction? Be specific with numbers.
3. **Sentiment Analysis Summary** — What is the market/news mood and what is driving it?
4. **Key Risks** — 2-3 specific risks that could change the outlook.
5. **Confidence Level** — Low / Medium / High — based on how many signals agree with each other.

Keep it concise and easy to understand. End with a disclaimer that this is not financial advice."""

    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a decisive stock analyst AI. You MUST take a clear bullish or bearish position based on the data — do not hedge everything as neutral. Support your view with specific data points. Always include a disclaimer that this is not financial advice.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error generating analysis: {str(e)}"


def chat_followup(ticker: str, analysis: str, user_question: str) -> str:
    """
    Handle follow-up questions about the analysis.
    """
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured."

    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a stock analyst AI. You previously provided this analysis for {ticker}:\n\n{analysis}\n\nAnswer follow-up questions based on this analysis. Be concise and helpful. Always remind that this is not financial advice.",
                },
                {"role": "user", "content": user_question},
            ],
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"
