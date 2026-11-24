"""Sentiment scoring.

Default backend is VADER: fast, offline, and tuned for short informal text such
as reviews and social posts. A Hugging Face transformer backend is available for
higher accuracy (`MOODMUSE_SENTIMENT=transformer`), at the cost of a model download.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05
LABELS = ("positive", "neutral", "negative")


def label_for(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "positive"
    if score <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


# VADER was built for social media and misses words that are clearly negative in
# product feedback ("crashes", "laggy"). These general-purpose additions use VADER's
# -4..+4 scale. They are deliberately generic, not tuned to any one dataset.
PRODUCT_LEXICON = {
    "crash": -2.0, "crashes": -2.0, "crashed": -2.0, "crashing": -2.0,
    "freeze": -1.8, "freezes": -1.8, "froze": -1.8, "freezing": -1.5,
    "lag": -1.5, "laggy": -1.8, "lags": -1.5, "slow": -1.3, "sluggish": -1.5,
    "bug": -1.5, "bugs": -1.5, "buggy": -2.0, "glitch": -1.5, "glitchy": -1.8,
    "broken": -2.0, "unusable": -2.5, "unreliable": -1.8, "fails": -1.5, "failing": -1.5,
    "drain": -1.3, "drains": -1.5, "overpriced": -2.0, "expensive": -1.2, "pricey": -1.0,
    "lost": -1.5, "disappear": -1.2, "disappears": -1.2, "spam": -1.8, "spammy": -1.8,
    "confusing": -1.5, "cluttered": -1.3, "late": -0.8, "wrong": -1.5, "error": -1.3,
    "refund": -0.8, "charged": -0.8, "locked": -0.8,
    "intuitive": 1.8, "seamless": 1.8, "smooth": 1.5, "fast": 1.0, "reliable": 1.5,
    "worth": 1.5, "value": 1.0, "fixed": 1.2, "polished": 1.5,
}


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(PRODUCT_LEXICON)
    return analyzer


@lru_cache(maxsize=1)
def _transformer():
    from transformers import pipeline  # optional dependency

    model = os.getenv("MOODMUSE_TRANSFORMER_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    return pipeline("sentiment-analysis", model=model, truncation=True)


def _transformer_scores(texts: list[str]) -> list[float]:
    out = []
    for r in _transformer()(texts, batch_size=32):
        label = r["label"].lower()
        sign = 1 if "pos" in label else -1 if "neg" in label else 0
        out.append(sign * float(r["score"]))
    return out


def score_texts(texts: pd.Series, backend: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with `score` in [-1, 1] and `sentiment` label, aligned to `texts`' index."""
    backend = (backend or os.getenv("MOODMUSE_SENTIMENT", "vader")).lower()
    if backend == "transformer":
        scores = _transformer_scores(texts.tolist())
    else:
        analyzer = _vader()
        scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    df = pd.DataFrame({"score": scores}, index=texts.index)
    df["sentiment"] = df["score"].map(label_for)
    return df
