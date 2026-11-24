"""MoodMuse dashboard: streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from moodmuse import analyze, to_markdown
from moodmuse.data import guess_text_column, read_table

SAMPLE = Path(__file__).parent / "data" / "sample_reviews.csv"
COLORS = {"positive": "#20b486", "neutral": "#9aa0b4", "negative": "#ef5b5b"}
PALETTE = ["#6d5efc", "#20b486", "#f59e0b", "#ef5b5b", "#3b82f6", "#ec4899", "#14b8a6", "#8b5cf6", "#84cc16", "#f97316"]

st.set_page_config(page_title="MoodMuse · feedback insights", page_icon="🎭", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 2rem; max-width: 1280px; }
  h1 { letter-spacing: -0.02em; }
  .hero-sub { color: #6b7088; font-size: 1.05rem; margin-top: -0.6rem; }
  .kpi { border: 1px solid rgba(120,120,150,.18); border-radius: 14px; padding: 16px 18px; }
  .kpi .v { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
  .kpi .l { color: #6b7088; font-size: .85rem; text-transform: uppercase; letter-spacing: .06em; }
  .theme-card { border: 1px solid rgba(120,120,150,.18); border-radius: 14px; padding: 16px 18px; height: 100%; }
  .theme-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
  .chip { display: inline-block; font-size: .78rem; padding: 2px 9px; margin: 2px 4px 2px 0;
          border-radius: 99px; background: rgba(109,94,252,.10); color: #6d5efc; }
  .quote { font-size: .9rem; color: #555b73; border-left: 3px solid rgba(109,94,252,.45);
           padding: 2px 0 2px 10px; margin: 8px 0; }
  .bar { display: flex; height: 8px; border-radius: 99px; overflow: hidden; margin: 8px 0 4px; }
</style>
""",
    unsafe_allow_html=True,
)


def esc(text: str) -> str:
    """Escape user-provided text before placing it inside our HTML snippets."""
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


@st.cache_data(show_spinner=False)
def run_analysis(texts: tuple[str, ...], k: int | None, backend: str):
    return analyze(pd.Series(texts), k=k, sentiment_backend=backend)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🎭 MoodMuse")
    st.caption("Upload feedback (reviews, survey answers, support tickets) and see the mood and the themes.")
    upload = st.file_uploader("CSV, TSV, XLSX or TXT", type=["csv", "tsv", "xlsx", "txt"])
    use_sample = upload is None
    if use_sample:
        st.info("Showing the built-in sample: 70 fictional reviews of a study-planner app.", icon="💡")

    try:
        df = read_table(SAMPLE) if use_sample else read_table(upload.getvalue(), upload.name)
    except Exception as exc:  # malformed upload
        st.error(f"Couldn't read that file: {exc}")
        st.stop()

    try:
        default_col = "review" if use_sample else guess_text_column(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    columns = [c for c in df.columns if df[c].dtype == object]
    column = st.selectbox("Text column", columns, index=columns.index(default_col) if default_col in columns else 0)

    auto_k = st.toggle("Choose number of themes automatically", value=True)
    k = None if auto_k else st.slider("Number of themes", 2, 12, 6)
    accurate = st.toggle(
        "Accurate sentiment (transformer)", value=False,
        help="Uses a RoBERTa sentiment model: more accurate, slower, and downloads ~500 MB the first time. "
             "Off: VADER + product lexicon (instant, offline).",
    )
    backend = "transformer" if accurate else "vader"
    st.divider()
    sent_name = "RoBERTa (cardiffnlp)" if accurate else "VADER + product-feedback lexicon"
    st.caption(f"Sentiment: {sent_name} · Themes: MiniLM embeddings + k-means · Labels: c-TF-IDF")

# ---------------- analysis ----------------
texts = tuple(df[column].dropna().astype(str).tolist())
if not texts:
    st.warning("That column is empty.")
    st.stop()

with st.spinner("Reading the room…"):
    try:
        a = run_analysis(texts, k, backend)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

table, stats = a.table, a.stats
theme_names = {t["id"]: t["label"] for t in a.theme_rows}
color_map = {t["label"]: PALETTE[i % len(PALETTE)] for i, t in enumerate(a.theme_rows)}

# ---------------- header + KPIs ----------------
st.title("MoodMuse")
st.markdown('<p class="hero-sub">What people feel, and what they are talking about.</p>', unsafe_allow_html=True)

kpis = [
    (f"{stats['total']}", "entries analysed"),
    (f"{stats['positive_pct']:.0f}%", "positive"),
    (f"{stats['negative_pct']:.0f}%", "negative"),
    (f"{len(a.theme_rows)}", "themes found"),
]
for col, (value, label) in zip(st.columns(4), kpis, strict=False):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)
if a.dropped:
    st.caption(f"{a.dropped} empty, very short or duplicate entries were removed before analysis.")

st.write("")
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Summary")
    st.markdown(a.summary)
    st.caption("Written by an LLM from the statistics below." if a.summary_mode == "llm"
               else "Generated directly from the statistics (no LLM configured).")
with right:
    counts = table["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    donut = go.Figure(go.Pie(
        labels=counts.index.str.capitalize(), values=counts.values, hole=0.62, sort=False,
        marker=dict(colors=[COLORS[s] for s in counts.index]), textinfo="percent",
    ))
    donut.update_layout(
        title="Overall sentiment", height=300, margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(orientation="h", y=-0.05), annotations=[dict(
            text=f"<b>{stats['avg_score']:+.2f}</b><br>avg score", showarrow=False, font=dict(size=15))],
    )
    st.plotly_chart(donut, use_container_width=True)

# ---------------- themes ----------------
st.subheader("Themes")
rows = pd.DataFrame(a.theme_rows)
stack = rows.melt(id_vars=["label", "share"], value_vars=["positive_pct", "neutral_pct", "negative_pct"],
                  var_name="sentiment", value_name="pct")
stack["sentiment"] = stack["sentiment"].str.replace("_pct", "")
stack["entries"] = (stack["pct"] / 100 * stack["share"] / 100 * stats["total"]).round().astype(int)
bars = px.bar(
    stack, y="label", x="entries", color="sentiment", orientation="h", color_discrete_map=COLORS,
    category_orders={"label": rows.sort_values("share")["label"].tolist()[::-1],
                     "sentiment": ["positive", "neutral", "negative"]},
    labels={"label": "", "entries": "entries"},
)
bars.update_layout(height=60 + 42 * len(rows), margin=dict(t=10, b=10, l=10, r=10), barmode="stack",
                   legend=dict(orientation="h", y=1.08, title=None))
st.plotly_chart(bars, use_container_width=True)

cards = st.columns(min(3, len(a.theme_rows)))
for i, t in enumerate(a.theme_rows):
    with cards[i % len(cards)]:
        bar = "".join(
            f'<div style="width:{t[f"{s}_pct"]:.1f}%;background:{COLORS[s]}"></div>'
            for s in ("positive", "neutral", "negative")
        )
        chips = "".join(f'<span class="chip">{esc(kw)}</span>' for kw in t["keywords"][:6])
        quotes = "".join(f'<div class="quote">“{esc(q)}”</div>' for q in t["examples"][:2])
        st.markdown(
            f'<div class="theme-card"><h4>{esc(t["label"])}</h4>'
            f'<div style="color:#6b7088;font-size:.85rem">{t["size"]} entries · {t["share"]:.0f}%'
            f' · {t["positive_pct"]:.0f}% positive · {t["negative_pct"]:.0f}% negative</div>'
            f'<div class="bar">{bar}</div>{chips}{quotes}</div>',
            unsafe_allow_html=True,
        )
        st.write("")

# ---------------- map ----------------
st.subheader("Feedback map")
st.caption("Each dot is one entry. Entries with similar meaning sit close together.")
plot_df = table.assign(theme_name=table["theme"].map(theme_names), short=table["text"].str.slice(0, 120))
scatter = px.scatter(
    plot_df, x="x", y="y", color="theme_name", symbol="sentiment", color_discrete_map=color_map,
    hover_data={"short": True, "score": ":.2f", "x": False, "y": False, "theme_name": False},
    labels={"theme_name": "Theme", "short": "Text"},
)
scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="white"), opacity=0.9))
scatter.update_layout(height=480, margin=dict(t=10, b=10, l=10, r=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
st.plotly_chart(scatter, use_container_width=True)

# ---------------- table + downloads ----------------
st.subheader("All entries")
f1, f2, f3 = st.columns([2, 2, 3])
pick_sent = f1.multiselect("Sentiment", ["positive", "neutral", "negative"], default=[])
pick_theme = f2.multiselect("Theme", list(theme_names.values()), default=[])
query = f3.text_input("Search text")
view = plot_df
if pick_sent:
    view = view[view["sentiment"].isin(pick_sent)]
if pick_theme:
    view = view[view["theme_name"].isin(pick_theme)]
if query:
    view = view[view["text"].str.contains(query, case=False, regex=False)]
st.dataframe(
    view[["text", "sentiment", "score", "theme_name"]].rename(columns={"theme_name": "theme"}),
    use_container_width=True, hide_index=True,
    column_config={"score": st.column_config.ProgressColumn("score", min_value=-1, max_value=1, format="%.2f")},
)

d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("⬇ Results CSV", table.drop(columns=["x", "y"]).to_csv(index=False), "moodmuse_results.csv",
                   "text/csv", use_container_width=True)
d2.download_button("⬇ Report (Markdown)", to_markdown(a), "moodmuse_report.md", "text/markdown",
                   use_container_width=True)
