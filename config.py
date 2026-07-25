"""
Configuration settings for the Stock Analysis AI Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys — works with .env locally AND st.secrets on Streamlit Cloud
def _get_secret(key):
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None

GROQ_API_KEY = _get_secret("GROQ_API_KEY")

# Default settings
DEFAULT_PERIOD = "6mo"  # How far back to pull price data
DEFAULT_INTERVAL = "1d"  # Daily candles

# Technical analysis parameters
SHORT_MA = 20
LONG_MA = 50
RSI_PERIOD = 14

# LLM settings — using Groq (free tier, works globally)
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"  # Free, much better reasoning than 8B
LLM_TEMPERATURE = 0.4  # Slightly higher for more decisive outputs

# News settings
MAX_NEWS_HEADLINES = 10
