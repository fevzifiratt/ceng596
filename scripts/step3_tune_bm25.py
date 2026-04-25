"""Step 3 — tune BM25 ``(k1, b)`` on the locked baseline configuration.

Tunes only on the winning Step 2 config (``stemmed`` index +
``title_desc`` queries), since tuning the loser configs is wasted effort.

Pipeline::

    1. Grid-search (k1, b) with pt.Experiment       -> evals/bm25_grid.csv
    2. Print a MAP heatmap, best per metric
    3. Re-run the best (k1, b) and write run file   -> runs/bm25_tuned_title_desc.trec
    4. Append a 'bm25_tuned' row                    -> evals/summary.csv

Run from the repo root::

    .venv/bin/python scripts/step3_tune_bm25.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyterrier as pt

from ir596.config import EVALS_DIR, QRELS_TREC
from ir596.evaluate import evaluate
from ir596.index.build_index import build
from ir596.retrieval.bm25 import bm25
from ir596.retrieval.topics import topics_df
from ir596.retrieval.tune_bm25 import (
    DEFAULT_B_GRID,
    DEFAULT_K1_GRID,
    DEFAULT_METRICS,
    grid_search,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MAP = 0.3344  # locked from Step 2 (bm25_stemmed_title_desc, defaults)


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def _load_qrels() -> pd.DataFrame:
    df = pd.read_csv(
        QRELS_TREC,
        sep=r"\s+",
        header=None,
        names=["qid", "iter", "docno", "label"],
        dtype={"qid": str, "docno": str},
    )
    return df[["qid", "docno", "label"]]


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    _hr("Setup")
    print(f"index           : stemmed (Porter + Terrier stoplist)")
    print(f"query field     : title + desc")
    print(f"k1 grid         : {list(DEFAULT_K1_GRID)}")
    print(f"b  grid         : {list(DEFAULT_B_GRID)}")
    n_configs = len(DEFAULT_K1_GRID) * len(DEFAULT_B_GRID)
    print(f"total configs   : {n_configs}")

    idx_ref = build("stemmed")
    topics = topics_df("title_desc")
    qrels = _load_qrels()
    print(f"queries / qrels : {len(topics)} / {len(qrels)}")

    _hr("Grid search")
    t0 = time.perf_counter()
    grid_df, best = grid_search(idx_ref, topics, qrels)
    elapsed = time.perf_counter() - t0
    print(f"  ran {len(grid_df)} configs in {elapsed:.1f}s "
          f"({elapsed / len(grid_df):.2f}s per config)")

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = EVALS_DIR / "bm25_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"  full grid       : {grid_path.relative_to(REPO_ROOT)}")

    _hr("MAP heatmap (rows = k1, cols = b)")
    pivot = grid_df.pivot(index="k1", columns="b", values="map")
    print(pivot.round(4).to_string())

    _hr("Best config per metric")
    for metric in DEFAULT_METRICS:
        row = grid_df.loc[grid_df[metric].idxmax()]
        marker = "  <-- selected" if metric == "map" else ""
        print(
            f"  best by {metric:14s} : k1={row['k1']:.2f}  b={row['b']:.2f}  "
            f"-> {row[metric]:.4f}{marker}"
        )

    _hr("Vs Step 2 baseline (defaults k1=1.20, b=0.75)")
    delta = best["map"] - BASELINE_MAP
    pct = 100 * delta / BASELINE_MAP
    print(
        f"  MAP : {BASELINE_MAP:.4f}  ->  {best['map']:.4f}  "
        f"({delta:+.4f}, {pct:+.1f}%)"
    )

    _hr("Re-run best config and append summary.csv row")
    evaluate(
        {"bm25_tuned_title_desc": bm25(idx_ref, k1=best["k1"], b=best["b"])},
        topics,
        extra_columns={
            "index": "stemmed",
            "query_field": "title_desc",
            "model": "BM25-tuned",
            "k1": float(best["k1"]),
            "b": float(best["b"]),
        },
    )
    print(f"  run             : runs/bm25_tuned_title_desc.trec")
    print(f"  summary updated : evals/summary.csv")

    _hr("Done")
    print(
        f"Tuned baseline locked: k1={best['k1']:.2f}, b={best['b']:.2f}, "
        f"MAP={best['map']:.4f}.  This is the new reference for Step 4 (PRF)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
