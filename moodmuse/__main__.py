"""Command line: python -m moodmuse feedback.csv [--column text] [--k 4] [--out report.md]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import guess_text_column, read_table
from .pipeline import analyze, to_markdown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moodmuse", description="Sentiment + themes + summary for written feedback.")
    p.add_argument("file", type=Path, help="CSV, TSV, XLSX or TXT (one entry per line)")
    p.add_argument("--column", help="text column (guessed if omitted)")
    p.add_argument("--k", type=int, help="number of themes (chosen automatically if omitted)")
    p.add_argument("--themes", choices=["auto", "embeddings", "tfidf"], default="auto",
                   help="theme method: sentence embeddings (default when installed) or TF-IDF")
    p.add_argument("--out", type=Path, help="write a Markdown report here")
    p.add_argument("--csv", type=Path, help="write per-entry results here")
    args = p.parse_args(argv)

    df = read_table(args.file)
    column = args.column or guess_text_column(df)
    if column not in df.columns:
        p.error(f"column '{column}' not found; available: {', '.join(map(str, df.columns))}")
    result = analyze(df[column], k=args.k, theme_method=args.themes)

    report = to_markdown(result, title=f"MoodMuse report: {args.file.name}")
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)
    if args.csv:
        result.table.drop(columns=["x", "y"]).to_csv(args.csv, index=False)
        print(f"Per-entry results written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
