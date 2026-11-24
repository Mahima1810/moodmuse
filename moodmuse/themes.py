"""Theme discovery.

1. Embed each text: sentence-transformers (all-MiniLM-L6-v2) when installed,
   otherwise TF-IDF reduced with SVD. Embeddings group texts by meaning, so
   "app keeps crashing" and "unusable, freezes constantly" land together.
2. Cluster with k-means; k is chosen by silhouette score unless given.
3. Describe each cluster with class-based TF-IDF (c-TF-IDF): words frequent in
   that cluster but rare in the others, so labels are distinctive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

MIN_DOCS_TO_CLUSTER = 10
EMBED_MODEL = os.getenv("MOODMUSE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Words too generic to describe a theme in feedback data.
EXTRA_STOP = {"app", "really", "just", "like", "use", "using", "used", "good", "great", "bad", "time", "feels"}


@dataclass
class Theme:
    id: int
    label: str
    keywords: list[str]
    size: int
    examples: list[str] = field(default_factory=list)


@dataclass
class ThemeResult:
    assignments: pd.Series  # theme id per text, aligned to the input index
    themes: list[Theme]
    coords: np.ndarray  # 2-D projection for plotting, one row per text
    k: int
    silhouette: float | None
    method: str  # "embeddings" or "tfidf"


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str], method: str = "auto") -> tuple[np.ndarray, str]:
    """Return L2-normalised vectors and the method actually used."""
    if method in ("auto", "embeddings"):
        try:
            vecs = _encoder().encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(vecs), "embeddings"
        except Exception:  # library missing or model unavailable offline
            if method == "embeddings":
                raise
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    n_comp = max(2, min(100, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    reduced = TruncatedSVD(n_components=n_comp, random_state=0).fit_transform(tfidf)
    return normalize(reduced), "tfidf"


def _project(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 3:
        return np.zeros((vectors.shape[0], 2))
    return PCA(n_components=2, random_state=0).fit_transform(vectors)


def _ctfidf_keywords(docs: list[str], labels: np.ndarray, n_themes: int, top_n: int = 8) -> list[list[str]]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    joined = [" ".join(d for d, lab in zip(docs, labels, strict=False) if lab == t) for t in range(n_themes)]
    try:
        cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=1)
        counts = cv.fit_transform(joined).toarray().astype(float)
    except ValueError:
        return [[] for _ in range(n_themes)]
    terms = cv.get_feature_names_out()
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    avg_words = counts.sum() / n_themes
    idf = np.log(1 + avg_words / np.maximum(counts.sum(axis=0), 1))
    scores = tf * idf
    out = []
    for row in scores:
        top = [terms[i] for i in np.argsort(-row)[: top_n * 2] if row[i] > 0]
        out.append(_dedupe_terms(top)[:top_n])
    return out


def find_themes(
    texts: pd.Series, k: int | None = None, max_k: int = 10, seed: int = 42, method: str = "auto"
) -> ThemeResult:
    docs = texts.tolist()
    vectors, used = embed(docs, method)
    coords = _project(vectors)

    best_k, best_sil = 1, None
    labels = np.zeros(len(docs), dtype=int)
    if len(docs) >= MIN_DOCS_TO_CLUSTER:
        candidates = [k] if k else range(3, max(4, min(max_k, len(docs) // 6) + 1))
        best_score = -1.0
        for kk in candidates:
            kk = min(kk, len(docs) - 1)
            model = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(vectors)
            if len(set(model.labels_)) < 2:
                continue
            sil = float(silhouette_score(vectors, model.labels_, metric="cosine"))
            if sil > best_score:
                best_k, best_sil, best_score, labels = kk, sil, sil, model.labels_

    # Order themes by size so theme 0 is always the biggest.
    order = pd.Series(labels).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[x] for x in labels])
    keywords = _ctfidf_keywords(docs, labels, len(order))

    themes = []
    for tid in range(len(order)):
        mask = labels == tid
        centroid = vectors[mask].mean(axis=0)
        sims = vectors[mask] @ centroid
        idx = np.where(mask)[0][np.argsort(-sims)[:3]]  # most typical texts
        kw = keywords[tid]
        themes.append(Theme(
            id=tid,
            label=" · ".join(kw[:3]) if kw else f"Theme {tid + 1}",
            keywords=kw,
            size=int(mask.sum()),
            examples=[docs[i] for i in idx],
        ))

    return ThemeResult(
        assignments=pd.Series(labels, index=texts.index, name="theme"),
        themes=themes, coords=coords, k=best_k, silhouette=best_sil, method=used,
    )


def _dedupe_terms(terms: list[str]) -> list[str]:
    """Drop unigrams already covered by a higher-ranked bigram ("battery" when "battery drain" is present)."""
    out: list[str] = []
    for t in terms:
        words = t.split()
        if len(words) == 1 and any(t in o.split() for o in out):
            continue
        if len(words) == 2 and all(w in out for w in words):
            continue
        out.append(t)
    return out
