"""
ML prediction module — binary classifier predicting stock direction.
Optimized for AUC-ROC using walk-forward validation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import ta as ta_lib


PREDICTION_HORIZON = 5  # Predict if price goes up/down in next 5 trading days
MIN_TRAINING_ROWS = 100  # Minimum data needed to train

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "return_1d": ("1-day return", "positive momentum", "negative momentum"),
    "return_5d": ("5-day return", "upward trend this week", "downward trend this week"),
    "return_10d": ("10-day return", "strong recent uptrend", "recent decline"),
    "volatility_10d": ("10-day volatility", "high volatility (bigger moves likely)", "low volatility (stable)"),
    "volatility_20d": ("20-day volatility", "elevated volatility", "calm market"),
    "volume_change": ("volume change", "increasing trading volume", "declining trading volume"),
    "volume_sma_ratio": ("volume vs average", "above-average volume (strong interest)", "below-average volume (low interest)"),
    "price_vs_sma20": ("price vs 20-day MA", "price above short-term average (bullish)", "price below short-term average (bearish)"),
    "price_vs_sma50": ("price vs 50-day MA", "price above long-term average (bullish)", "price below long-term average (bearish)"),
    "sma20_vs_sma50": ("MA crossover", "short MA above long MA (golden cross)", "short MA below long MA (death cross)"),
    "rsi": ("RSI", "RSI indicates overbought or strong momentum", "RSI indicates oversold or weak momentum"),
    "macd": ("MACD", "positive MACD (bullish momentum)", "negative MACD (bearish momentum)"),
    "macd_signal": ("MACD signal", "MACD above signal line", "MACD below signal line"),
    "macd_diff": ("MACD histogram", "expanding bullish momentum", "expanding bearish momentum"),
    "bb_position": ("Bollinger Band position", "price near upper band (strong/overbought)", "price near lower band (weak/oversold)"),
    "bb_width": ("Bollinger Band width", "wide bands (high volatility)", "narrow bands (potential breakout)"),
    "stoch_k": ("Stochastic %K", "high stochastic (overbought zone)", "low stochastic (oversold zone)"),
    "stoch_d": ("Stochastic %D", "stochastic signal trending up", "stochastic signal trending down"),
}


def _explain_prediction(latest_values: pd.Series, feat_importance: pd.Series, direction: str, horizon: int) -> list:
    """
    Generate human-readable reasons for the ML prediction.
    Uses the top important features and their current values.
    """
    reasons = []
    top_features = feat_importance.head(5)

    for feature_name, importance in top_features.items():
        value = latest_values.get(feature_name)
        if value is None or pd.isna(value):
            continue

        desc = FEATURE_DESCRIPTIONS.get(feature_name)
        if not desc:
            continue

        label, pos_meaning, neg_meaning = desc

        # Determine if the feature value is positive or negative signal
        if feature_name == "rsi":
            if value > 70:
                reasons.append(f"**{label}** is {value:.1f} — overbought, may reverse down")
            elif value < 30:
                reasons.append(f"**{label}** is {value:.1f} — oversold, may bounce up")
            else:
                reasons.append(f"**{label}** is {value:.1f} — neutral range")
        elif feature_name == "bb_position":
            if value > 0.8:
                reasons.append(f"**{label}**: price near upper band — {pos_meaning}")
            elif value < 0.2:
                reasons.append(f"**{label}**: price near lower band — {neg_meaning}")
            else:
                reasons.append(f"**{label}**: price in middle of bands — neutral")
        elif feature_name in ("stoch_k", "stoch_d"):
            if value > 80:
                reasons.append(f"**{label}** is {value:.1f} — {pos_meaning}")
            elif value < 20:
                reasons.append(f"**{label}** is {value:.1f} — {neg_meaning}")
            else:
                reasons.append(f"**{label}** is {value:.1f} — neutral zone")
        else:
            if value > 0:
                reasons.append(f"**{label}**: {pos_meaning} ({value:+.4f})")
            else:
                reasons.append(f"**{label}**: {neg_meaning} ({value:+.4f})")

    return reasons


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from price data."""
    feat = pd.DataFrame(index=df.index)

    # Price momentum
    feat["return_1d"] = df["Close"].pct_change(1)
    feat["return_5d"] = df["Close"].pct_change(5)
    feat["return_10d"] = df["Close"].pct_change(10)

    # Volatility
    feat["volatility_10d"] = df["Close"].pct_change().rolling(10).std()
    feat["volatility_20d"] = df["Close"].pct_change().rolling(20).std()

    # Volume
    if "Volume" in df.columns:
        feat["volume_change"] = df["Volume"].pct_change()
        feat["volume_sma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Moving average features
    sma_20 = ta_lib.trend.sma_indicator(df["Close"], window=20)
    sma_50 = ta_lib.trend.sma_indicator(df["Close"], window=50)
    feat["price_vs_sma20"] = (df["Close"] - sma_20) / sma_20
    feat["price_vs_sma50"] = (df["Close"] - sma_50) / sma_50
    feat["sma20_vs_sma50"] = (sma_20 - sma_50) / sma_50

    # RSI
    feat["rsi"] = ta_lib.momentum.rsi(df["Close"], window=14)

    # MACD
    macd = ta_lib.trend.MACD(df["Close"])
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_diff"] = macd.macd_diff()

    # Bollinger Band position
    bb = ta_lib.volatility.BollingerBands(df["Close"], window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_range = bb_upper - bb_lower
    feat["bb_position"] = np.where(bb_range != 0, (df["Close"] - bb_lower) / bb_range, 0.5)
    feat["bb_width"] = np.where(df["Close"] != 0, bb_range / df["Close"], 0)

    # Stochastic oscillator
    feat["stoch_k"] = ta_lib.momentum.stoch(df["High"], df["Low"], df["Close"])
    feat["stoch_d"] = ta_lib.momentum.stoch_signal(df["High"], df["Low"], df["Close"])

    return feat


def create_target(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.Series:
    """Binary target: 1 if price UP in next `horizon` days, 0 otherwise."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > 0).astype(int)


def train_and_evaluate(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> dict:
    """Train a GradientBoosting classifier with walk-forward validation."""
    if len(df) < MIN_TRAINING_ROWS:
        return {
            "error": f"Need at least {MIN_TRAINING_ROWS} trading days. Got {len(df)}. Try '1y' or '2y' time period.",
            "model": None,
        }

    features = prepare_features(df)
    target = create_target(df, horizon=horizon)

    combined = features.copy()
    combined["target"] = target
    combined = combined.dropna()

    if len(combined) < 60:
        return {"error": "Not enough clean data after computing indicators.", "model": None}

    X = combined.drop("target", axis=1)
    y = combined["target"]

    # Walk-forward split: train on first 70%, test on last 30%
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.8, random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    feat_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(8)

    # Latest prediction
    latest_features = X.iloc[[-1]]
    latest_scaled = scaler.transform(latest_features)
    latest_proba = model.predict_proba(latest_scaled)[0]
    direction = "UP" if latest_proba[1] > 0.5 else "DOWN"

    # Generate human-readable explanation of why the model predicted this direction
    explanation = _explain_prediction(latest_features.iloc[0], feat_importance, direction, horizon)

    return {
        "error": None,
        "model": model,
        "scaler": scaler,
        "auc_roc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importance": feat_importance.to_dict(),
        "latest_prediction": {
            "up_probability": round(latest_proba[1], 4),
            "down_probability": round(latest_proba[0], 4),
            "direction": direction,
            "confidence": round(max(latest_proba) * 100, 1),
            "explanation": explanation,
        },
        "test_predictions": {
            "dates": X_test.index.tolist(),
            "actual": y_test.tolist(),
            "predicted_proba": y_pred_proba.tolist(),
        },
        "prediction_horizon": horizon,
    }
