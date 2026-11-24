"""Turn the numbers into a short written summary.

With an LLM configured (any OpenAI-compatible API), the model writes the summary
from the computed statistics and real example quotes only. Without one, a
template summary is built directly from the statistics, so the output is always
traceable to the data.
"""

from __future__ import annotations

import os

import httpx

SYSTEM_PROMPT = """You are an analyst summarising customer or user feedback.
Use ONLY the statistics and quotes provided. Do not invent numbers, themes or quotes.
Write 4-6 short bullet points: overall mood, the biggest praise, the biggest complaints,
and one concrete recommendation. Mention theme names and percentages from the data."""


def llm_configured() -> bool:
    return bool(os.getenv("LLM_MODEL", "").strip())


def _chat(system: str, user: str) -> str:
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if key := os.getenv("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    resp = httpx.post(
        f"{base}/chat/completions",
        headers=headers,
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        json={
            "model": os.environ["LLM_MODEL"],
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def facts_block(stats: dict, theme_rows: list[dict]) -> str:
    lines = [
        f"Total entries: {stats['total']}",
        f"Positive: {stats['positive_pct']:.0f}% | Neutral: {stats['neutral_pct']:.0f}% | "
        f"Negative: {stats['negative_pct']:.0f}%",
        "",
        "Themes:",
    ]
    for t in theme_rows:
        lines.append(
            f"- {t['label']} ({t['share']:.0f}% of entries; {t['positive_pct']:.0f}% positive, "
            f"{t['negative_pct']:.0f}% negative)"
        )
        lines.append(f'    most negative quote: "{t["most_negative"]}"')
        lines.append(f'    most positive quote: "{t["most_positive"]}"')
    return "\n".join(lines)


def template_summary(stats: dict, theme_rows: list[dict]) -> str:
    mood = max(("positive", "neutral", "negative"), key=lambda s: stats[f"{s}_pct"])
    bullets = [
        f"**Overall mood is mostly {mood}**: {stats['positive_pct']:.0f}% positive, "
        f"{stats['neutral_pct']:.0f}% neutral, {stats['negative_pct']:.0f}% negative across {stats['total']} entries."
    ]
    if theme_rows:
        biggest = max(theme_rows, key=lambda t: t["share"])
        bullets.append(f"**Most discussed theme:** {biggest['label']} ({biggest['share']:.0f}% of entries).")
    if len(theme_rows) > 1:
        best = max(theme_rows, key=lambda t: t["avg_score"])
        worst = min(theme_rows, key=lambda t: t["avg_score"])
        bullets.append(f"**Most positive theme:** {best['label']} ({best['positive_pct']:.0f}% positive).")
        if worst["negative_pct"] > 0:
            bullets.append(
                f"**Biggest pain point:** {worst['label']} ({worst['negative_pct']:.0f}% negative), "
                f'e.g. "{worst["most_negative"]}"'
            )
            bullets.append(f"**Suggested focus:** investigate {worst['label'].split(' · ')[0]} issues first.")
    return "\n".join(f"- {b}" for b in bullets)


def summarize(stats: dict, theme_rows: list[dict]) -> tuple[str, str]:
    """Return (summary_markdown, mode) where mode is "llm" or "template"."""
    if llm_configured():
        try:
            return _chat(SYSTEM_PROMPT, facts_block(stats, theme_rows)), "llm"
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            pass  # fall through to the template so the app keeps working
    return template_summary(stats, theme_rows), "template"
