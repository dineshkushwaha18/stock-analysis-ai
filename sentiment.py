"""
Sentiment analysis module — scores news headlines using the LLM.
"""

from openai import OpenAI
from config import GROQ_API_KEY, LLM_MODEL, LLM_BASE_URL


def analyze_sentiment(headlines: list[dict], ticker: str) -> dict:
    """
    Use the LLM to score the overall sentiment of recent headlines.
    Returns a dict with sentiment_score (-1 to 1), label, and summary.
    """
    if not headlines:
        return {
            "score": 0,
            "label": "neutral",
            "summary": "No recent news found.",
            "headline_sentiments": [],
        }

    if not GROQ_API_KEY:
        return {
            "score": 0,
            "label": "unavailable",
            "summary": "API key not configured. Sentiment analysis skipped.",
            "headline_sentiments": [],
        }

    headline_text = "\n".join([f"- {h['title']}" for h in headlines])

    prompt = f"""You are a financial sentiment analyst. Analyze these news headlines about {ticker} stock.

Headlines:
{headline_text}

IMPORTANT RULES:
- You MUST take a clear position. Avoid defaulting to neutral unless the evidence is truly mixed.
- Score each headline individually first, then compute the overall score.
- A score of exactly 0.0 is only acceptable if positive and negative headlines are perfectly balanced.
- Consider the financial impact: earnings beats, analyst upgrades, new products = positive. Lawsuits, misses, downgrades = negative.
- Ignore the news source name in the headline (e.g. "- Reuters") and focus on the content.

Respond in this exact JSON format:
{{
    "overall_score": <float from -1.0 (very bearish) to 1.0 (very bullish). MUST NOT be 0.0 unless truly balanced>,
    "overall_label": "<bearish|slightly_bearish|neutral|slightly_bullish|bullish>",
    "summary": "<2-3 sentence summary explaining WHY the sentiment is positive or negative>",
    "headline_sentiments": [
        {{"headline": "<headline text>", "sentiment": "<positive|negative|neutral>", "reason": "<brief reason>"}}
    ]
}}

Only return valid JSON, nothing else."""

    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        import json
        import re
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        # Try to fix common JSON issues from LLM output
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from the response
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                # Replace unescaped control characters
                cleaned = match.group(0)
                cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                result = json.loads(cleaned)
            else:
                raise ValueError("No valid JSON found in response")
        return {
            "score": float(result.get("overall_score", 0)),
            "label": result.get("overall_label", "neutral"),
            "summary": result.get("summary", ""),
            "headline_sentiments": result.get("headline_sentiments", []),
        }
    except Exception as e:
        return {
            "score": 0,
            "label": "error",
            "summary": f"Sentiment analysis failed: {str(e)}",
            "headline_sentiments": [],
        }
