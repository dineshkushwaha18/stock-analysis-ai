"""
evaluate.py — Evaluation harness for the Stock Analysis AI Assistant.

Produces the exact numbers needed for the "Evaluation" slide:
  1. Signal-consistency (face validity): does the combined signal agree with the
     stock's recent price direction?
  2. Anti-neutral prompting: what % of runs are directional (non-Neutral)
     BEFORE vs AFTER the anti-neutral prompt rules?

It imports the project's OWN modules so the numbers reflect the real system,
not a reimplementation. The only thing reproduced here is compute_overall_signal,
because that function lives inline in app.py and isn't importable.

USAGE
  # 1. Make sure your Groq key is set (same as the app):
  #    put GROQ_API_KEY=... in .env, OR: export GROQ_API_KEY=...
  # 2. From the repo root:
  python evaluate.py                 # AFTER: current anti-neutral prompt
  python evaluate.py --baseline      # BEFORE: permissive prompt (no anti-neutral rules)
  python evaluate.py --tickers AAPL TSLA NVDA ...   # custom ticker list

OUTPUT
  - Prints a per-ticker table and a summary block mapping 1:1 to the slide blanks
  - Writes eval_results_after.csv (or eval_results_before.csv with --baseline)
"""

import argparse
import csv
import json
import re
import time

import pandas as pd

# ── The project's own modules ────────────────────────────────────────────────
from data_fetcher import get_stock_data, get_stock_info, get_news_headlines
from technical_analysis import compute_indicators, get_latest_signals
from sentiment import analyze_sentiment
from config import GROQ_API_KEY, LLM_MODEL, LLM_BASE_URL


# ── compute_overall_signal: copied VERBATIM from app.py so the label matches ──
def compute_overall_signal(signals, sentiment):
    score = 0
    count = 0
    for interp in signals.get("interpretations", []):
        if "bullish" in interp.lower():
            score += 1
            count += 1
        elif "bearish" in interp.lower():
            score -= 1
            count += 1
    sent_label = sentiment.get("label", "")
    if sent_label not in ("error", "unavailable", ""):
        sent_score = sentiment.get("score", 0)
        score += sent_score * 2
        count += 2
    avg = score / count if count > 0 else 0
    # Labels match app.py verbatim (including the emoji prefixes).
    if avg > 0.2:
        return "🟢 BULLISH", avg
    elif avg > 0.05:
        return "🟢 Slightly Bullish", avg
    elif avg > -0.05:
        return "🟡 Neutral", avg
    elif avg > -0.2:
        return "🔴 Slightly Bearish", avg
    else:
        return "🔴 BEARISH", avg


def technical_subscore(signals):
    """The technical-only portion of the score (for the sample-output rows)."""
    s = 0
    for interp in signals.get("interpretations", []):
        low = interp.lower()
        if "bullish" in low:
            s += 1
        elif "bearish" in low:
            s -= 1
    return s


# ── BEFORE state: a permissive sentiment call with NO anti-neutral rules ──────
def analyze_sentiment_baseline(headlines, ticker):
    """
    Mirrors sentiment.analyze_sentiment but strips the anti-neutral rules,
    so you can measure how often the OLD prompt defaulted to neutral.
    """
    from openai import OpenAI

    if not headlines or not GROQ_API_KEY:
        return {"score": 0, "label": "neutral", "summary": "", "headline_sentiments": []}

    headline_text = "\n".join([f"- {h['title']}" for h in headlines])
    prompt = f"""You are a financial sentiment analyst. Analyze these news headlines about {ticker} stock.

Headlines:
{headline_text}

Respond in this exact JSON format:
{{
    "overall_score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "overall_label": "<bearish|slightly_bearish|neutral|slightly_bullish|bullish>",
    "summary": "<2-3 sentence summary>",
    "headline_sentiments": []
}}

Only return valid JSON, nothing else."""
    try:
        client = OpenAI(api_key=GROQ_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            result = json.loads(m.group(0).replace("\n", " ")) if m else {}
        return {
            "score": float(result.get("overall_score", 0)),
            "label": result.get("overall_label", "neutral"),
            "summary": result.get("summary", ""),
            "headline_sentiments": [],
        }
    except Exception as e:
        return {"score": 0, "label": "error", "summary": str(e), "headline_sentiments": []}


# ── Ground truth: recent price direction from the SAME df the app uses ─────────
def recent_direction(df, lookback=20, flat_threshold=0.03):
    """
    Classify recent price direction over `lookback` trading days.
    Returns ('uptrend'|'downtrend'|'sideways', pct_change).
    This is a FACE-VALIDITY check (does the label agree with what the price
    has recently done), NOT a predictive-accuracy test — describe it that way
    on the slide.
    """
    closes = df["Close"].dropna()
    if len(closes) < lookback + 1:
        lookback = len(closes) - 1
    if lookback < 1:
        return "unknown", 0.0
    old, new = closes.iloc[-lookback - 1], closes.iloc[-1]
    pct = (new - old) / old
    if pct > flat_threshold:
        return "uptrend", pct
    elif pct < -flat_threshold:
        return "downtrend", pct
    return "sideways", pct


def label_direction(label):
    if "BULLISH" in label.upper() or "Bullish" in label:
        return "uptrend"
    if "BEARISH" in label.upper() or "Bearish" in label:
        return "downtrend"
    return "sideways"


# A spread of tickers across market conditions. Swap in your own if you like.
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL",
                   "AMZN", "META", "JPM", "XOM", "PFE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--baseline", action="store_true",
                    help="Use the permissive prompt (measures the BEFORE state)")
    ap.add_argument("--lookback", type=int, default=20)
    args = ap.parse_args()

    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY not set. Put it in .env or export it, then rerun.")

    mode = "before" if args.baseline else "after"
    sent_fn = analyze_sentiment_baseline if args.baseline else analyze_sentiment
    print(f"\n=== Evaluation run: {mode.upper()} anti-neutral prompting ===\n")

    rows = []
    for t in args.tickers:
        try:
            df = get_stock_data(t, period="1y")
            if df.empty:
                print(f"  {t}: no data, skipped"); continue
            df = compute_indicators(df)
            signals = get_latest_signals(df)
            headlines = get_news_headlines(t)
            sentiment = sent_fn(headlines, t)
            label, avg = compute_overall_signal(signals, sentiment)
            tech = technical_subscore(signals)
            direction, pct = recent_direction(df, lookback=args.lookback)
            aligned = (label_direction(label) == direction)
            rows.append({
                "ticker": t,
                "technical_score": tech,
                "sentiment_score": round(sentiment.get("score", 0), 3),
                "combined_avg": round(avg, 3),
                "combined_label": label,
                "recent_direction": direction,
                "recent_pct": f"{pct*100:+.1f}%",
                "aligned": aligned,
            })
            print(f"  {t:6s} | tech {tech:+d} | sent {sentiment.get('score',0):+.2f} "
                  f"| {label:16s} | price {pct*100:+5.1f}% ({direction}) "
                  f"| {'MATCH' if aligned else 'miss'}")
            time.sleep(1)  # be gentle on the Groq free tier
        except Exception as e:
            print(f"  {t}: error {e}")

    if not rows:
        raise SystemExit("No rows produced — check network / API key.")

    # ── Summary block: these map 1:1 to the slide blanks ─────────────────────
    n = len(rows)
    # Labels may carry emoji prefixes (e.g. "🟡 Neutral"), so match by substring.
    directional = [r for r in rows if "Neutral" not in r["combined_label"]]
    neutral = [r for r in rows if "Neutral" in r["combined_label"]]
    # Alignment only meaningful for directional calls vs non-sideways price
    testable = [r for r in rows if r["recent_direction"] != "sideways"]
    aligned = [r for r in testable if r["aligned"]]

    print("\n" + "=" * 62)
    print(f"  SUMMARY  ({mode} anti-neutral prompting)")
    print("=" * 62)
    print(f"  Tickers tested ............... {n}")
    print(f"  Directional (non-Neutral) .... {len(directional)}/{n} "
          f"= {100*len(directional)/n:.0f}%")
    print(f"  Neutral / non-committal ...... {len(neutral)}/{n} "
          f"= {100*len(neutral)/n:.0f}%")
    if testable:
        print(f"  Signal matched recent price .. {len(aligned)}/{len(testable)} "
              f"of tickers in a clear trend")
    print("\n  Two sample outputs for the slide:")
    for r in rows[:2]:
        print(f"    {r['ticker']} — Technical score: {r['technical_score']:+d} | "
              f"Sentiment score: {r['sentiment_score']:+.2f} | "
              f"Output: {r['combined_label']}")
    print("=" * 62 + "\n")

    out = f"eval_results_{mode}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {out}\n")


if __name__ == "__main__":
    main()
