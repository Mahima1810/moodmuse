"""Loading and cleaning free-text feedback."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

MIN_CHARS = 3


def read_table(source: str | Path | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read CSV, TSV, XLSX or plain text (one entry per line) into a DataFrame."""
    name = (filename or str(source)).lower()
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".txt"):
        lines = [ln.strip() for ln in raw.decode("utf-8", errors="replace").splitlines()]
        return pd.DataFrame({"text": [ln for ln in lines if ln]})
    sep = "\t" if name.endswith(".tsv") else None
    return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding_errors="replace")


def text_columns(df: pd.DataFrame) -> list[str]:
    """Columns that can hold text. pandas 2 stores text as `object`; pandas 3 uses a dedicated `str` dtype."""
    return [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def guess_text_column(df: pd.DataFrame) -> str:
    """Pick the column that most looks like free text: string-typed with the longest average length."""
    candidates = []
    for col in df.columns:
        series = df[col].dropna()
        if series.empty or not series.map(lambda v: isinstance(v, str)).mean() > 0.8:
            continue
        candidates.append((series.astype(str).str.len().mean(), col))
    if not candidates:
        raise ValueError("No text column found. Make sure the file has a column of written feedback.")
    return max(candidates)[1]


def clean_texts(texts: pd.Series) -> pd.Series:
    """Normalise whitespace, drop empty/very short entries and exact duplicates. Keeps the original index."""
    s = texts.dropna().astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    s = s[s.str.len() >= MIN_CHARS]
    return s[~s.str.lower().duplicated()]
