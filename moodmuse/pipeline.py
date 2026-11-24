"""End-to-end analysis: clean → sentiment → themes → summary."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import clean_texts
from .sentiment import score_texts
from .summarize import summarize
from .themes import ThemeResult, find_themes


@dataclass
class Analysis:
    table: pd.DataFrame  # one row per entry: text, score, sentiment, theme, theme_label, x, y
    stats: dict
    theme_rows: list[dict]
    themes: ThemeResult
    summary: str
    summary_mode: str
    dropped: int  # empty, too-short or duplicate entries removed during cleaning


def _pct(series: pd.Series, label: str) -> float:
    return float((series == label).mean() * 100) if len(series) else 0.0


def analyze(
    texts: pd.Series, k: int | None = None, sentiment_backend: str | None = None, theme_method: str = "auto"
) -> Analysis:
    cleaned = clean_texts(texts)
    if len(cleaned) == 0:
        raise ValueError("No usable text entries found after cleaning.")

    sent = score_texts(cleaned, backend=sentiment_backend)
    themes = find_themes(cleaned, k=k, method=theme_method)
    labels = {t.id: t.label for t in themes.themes}

    table = pd.DataFrame({"text": cleaned})
    table = table.join(sent)
    table["theme"] = themes.assignments
    table["theme_label"] = table["theme"].map(labels)
    table["x"], table["y"] = themes.coords[:, 0], themes.coords[:, 1]

    stats = {
        "total": int(len(table)),
        "avg_score": float(table["score"].mean()),
        **{f"{s}_pct": _pct(table["sentiment"], s) for s in ("positive", "neutral", "negative")},
    }

    theme_rows = []
    for t in themes.themes:
        part = table[table["theme"] == t.id]
        theme_rows.append({
            "id": t.id,
            "label": t.label,
            "keywords": t.keywords,
            "size": t.size,
            "share": t.size / len(table) * 100,
            "avg_score": float(part["score"].mean()),
            **{f"{s}_pct": _pct(part["sentiment"], s) for s in ("positive", "neutral", "negative")},
            "examples": t.examples,
            "most_negative": part.sort_values("score").iloc[0]["text"],
            "most_positive": part.sort_values("score").iloc[-1]["text"],
        })

    summary, mode = summarize(stats, theme_rows)
    return Analysis(
        table=table, stats=stats, theme_rows=theme_rows, themes=themes,
        summary=summary, summary_mode=mode, dropped=int(len(texts) - len(cleaned)),
    )


def to_markdown(a: Analysis, title: str = "MoodMuse report") -> str:
    lines = [f"# {title}", "", f"Entries analysed: {a.stats['total']} (removed {a.dropped} empty/duplicate)", ""]
    lines += ["## Summary", "", a.summary, "", "## Themes", ""]
    lines += ["| Theme | Share | Positive | Negative | Keywords |", "|---|---|---|---|---|"]
    for t in a.theme_rows:
        lines.append(
            f"| {t['label']} | {t['share']:.0f}% | {t['positive_pct']:.0f}% | {t['negative_pct']:.0f}% "
            f"| {', '.join(t['keywords'][:6])} |"
        )
    lines += ["", "## Example quotes", ""]
    for t in a.theme_rows:
        lines.append(f"**{t['label']}**")
        lines += [f"- \"{q}\"" for q in t["examples"]]
        lines.append("")
    if a.themes.silhouette is not None:
        lines.append(
            f"_Clustering: {a.themes.method}, k={a.themes.k}, silhouette (cosine) = {a.themes.silhouette:.3f}_"
        )
    return "\n".join(lines)


__all__ = ["Analysis", "analyze", "to_markdown"]
