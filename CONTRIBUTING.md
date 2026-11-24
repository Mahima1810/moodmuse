# Contributing to MoodMuse

Thanks for your interest! Bug reports, ideas and pull requests are all welcome.

## Setup

```bash
git clone https://github.com/Mahima1810/moodmuse.git
cd moodmuse
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Before opening a pull request

1. `pytest` and `ruff check .` must pass.
2. Add or update a test for any behaviour change.
3. If you change sentiment or clustering, run `python scripts/evaluate_sentiment.py` and include the before/after numbers in the PR.
4. Keep PRs focused, one change per PR.

## Project layout

| Path | What it does |
|---|---|
| `moodmuse/data.py` | Reading CSV/TSV/XLSX/TXT, picking the text column, cleaning |
| `moodmuse/sentiment.py` | VADER + product-feedback lexicon, optional transformer backend |
| `moodmuse/themes.py` | Embeddings → k-means (silhouette-selected k) → c-TF-IDF labels |
| `moodmuse/summarize.py` | LLM or template summary built only from computed statistics |
| `moodmuse/pipeline.py` | Ties it together; Markdown report |
| `app.py` | Streamlit dashboard |

## Ground rule

Every number and quote MoodMuse shows must come from the user's data. Summaries must never invent themes, figures or quotes.
