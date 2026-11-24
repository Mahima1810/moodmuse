"""Check sentiment labels against star ratings in the sample data.

Ratings 4-5 count as positive, 3 as neutral, 1-2 as negative. This is a rough
proxy (a 3-star review can still be clearly negative), but it shows whether the
scorer is in the right direction.

Usage: python scripts/evaluate_sentiment.py [csv] [--backend vader|transformer]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moodmuse.sentiment import score_texts  # noqa: E402


def rating_to_label(r: int) -> str:
    return "positive" if r >= 4 else "negative" if r <= 2 else "neutral"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="data/sample_reviews.csv")
    p.add_argument("--backend", default="vader")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    pred = score_texts(df["review"], backend=args.backend)["sentiment"]
    gold = df["rating"].map(rating_to_label)
    acc = (pred == gold).mean()
    polar = gold != "neutral"
    polar_acc = (pred[polar] == gold[polar]).mean()
    print(f"backend={args.backend}  n={len(df)}")
    print(f"3-class accuracy:              {acc:.1%}")
    print(f"accuracy on 1-2 and 4-5 stars: {polar_acc:.1%}")
    print(pd.crosstab(gold, pred, rownames=["rating"], colnames=["predicted"]))


if __name__ == "__main__":
    main()
