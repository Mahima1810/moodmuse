import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moodmuse import analyze, to_markdown  # noqa: E402
from moodmuse.__main__ import main as cli_main  # noqa: E402
from moodmuse.data import clean_texts, guess_text_column, read_table  # noqa: E402
from moodmuse.sentiment import label_for, score_texts  # noqa: E402
from moodmuse.themes import _dedupe_terms, find_themes  # noqa: E402

SAMPLE = ROOT / "data" / "sample_reviews.csv"


@pytest.fixture(scope="module")
def reviews() -> pd.Series:
    return pd.read_csv(SAMPLE)["review"]


# ---------- data ----------
def test_clean_drops_empty_short_and_duplicates():
    s = pd.Series(["Great app", "  great   APP ", "", None, "ok", "Crashes a lot"])
    out = clean_texts(s)
    assert out.tolist() == ["Great app", "Crashes a lot"]


def test_guess_text_column_picks_longest_strings():
    df = pd.DataFrame({"id": [1, 2], "tag": ["a", "b"], "comment": ["this is long feedback", "another long one"]})
    assert guess_text_column(df) == "comment"


def test_guess_text_column_errors_without_text():
    with pytest.raises(ValueError):
        guess_text_column(pd.DataFrame({"a": [1, 2]}))


def test_read_txt_one_entry_per_line():
    df = read_table(b"first line\n\nsecond line\n", "notes.txt")
    assert df["text"].tolist() == ["first line", "second line"]


# ---------- sentiment ----------
def test_thresholds():
    assert label_for(0.5) == "positive"
    assert label_for(0.0) == "neutral"
    assert label_for(-0.5) == "negative"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I love this app, it is wonderful", "positive"),
        ("The app crashes every time I open it", "negative"),  # needs the product lexicon
        ("Laggy and buggy on my phone", "negative"),
        ("The meeting is on Tuesday", "neutral"),
    ],
)
def test_vader_with_product_lexicon(text, expected):
    assert score_texts(pd.Series([text]), backend="vader")["sentiment"].iloc[0] == expected


def test_sentiment_matches_star_ratings_reasonably():
    df = pd.read_csv(SAMPLE)
    pred = score_texts(df["review"], backend="vader")["sentiment"]
    polar = df["rating"] != 3
    gold = df.loc[polar, "rating"].map(lambda r: "positive" if r >= 4 else "negative")
    assert (pred[polar] == gold).mean() >= 0.75


# ---------- themes ----------
def test_dedupe_terms():
    assert _dedupe_terms(["battery drain", "battery", "drain", "slow"]) == ["battery drain", "slow"]


def test_tfidf_themes_cover_every_text(reviews):
    res = find_themes(reviews, method="tfidf")
    assert len(res.assignments) == len(reviews)
    assert sum(t.size for t in res.themes) == len(reviews)
    assert res.themes == sorted(res.themes, key=lambda t: -t.size)  # biggest first
    assert all(t.keywords for t in res.themes)


def test_fixed_k_is_respected(reviews):
    assert find_themes(reviews, k=4, method="tfidf").k == 4


def test_tiny_input_is_one_theme():
    res = find_themes(pd.Series(["good", "bad app", "fine"]), method="tfidf")
    assert res.k == 1 and len(res.themes) == 1


# ---------- pipeline ----------
def test_analyze_end_to_end(reviews, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    a = analyze(reviews, theme_method="tfidf")
    assert a.stats["total"] == len(reviews)
    assert abs(a.stats["positive_pct"] + a.stats["neutral_pct"] + a.stats["negative_pct"] - 100) < 1e-6
    assert a.summary_mode == "template"
    assert "Overall mood" in a.summary
    report = to_markdown(a)
    assert "## Themes" in report and "## Example quotes" in report


def test_analyze_rejects_empty():
    with pytest.raises(ValueError):
        analyze(pd.Series(["", None]))


def test_llm_failure_falls_back_to_template(reviews, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setenv("LLM_TIMEOUT", "2")
    a = analyze(reviews.head(20), theme_method="tfidf")
    assert a.summary_mode == "template"


def test_cli_writes_report(tmp_path):
    out = tmp_path / "report.md"
    assert cli_main([str(SAMPLE), "--column", "review", "--k", "4", "--out", str(out), "--themes", "tfidf"]) == 0
    assert out.read_text(encoding="utf-8").startswith("# MoodMuse report")
