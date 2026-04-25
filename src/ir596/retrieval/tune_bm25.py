"""BM25 hyperparameter grid search.

Sweeps the standard BM25 knobs:

- ``k1``  : term-frequency saturation (Terrier default 1.2)
- ``b``   : document-length normalization weight (Terrier default 0.75)

The grid is intentionally small; we want a coarse-but-honest reference
point for the progress report, not a maximum-likelihood fit.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import pyterrier as pt

from ir596.retrieval.bm25 import bm25

DEFAULT_K1_GRID: tuple[float, ...] = (0.6, 0.9, 1.2, 1.5, 1.8, 2.1)
DEFAULT_B_GRID:  tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DEFAULT_METRICS: tuple[str, ...]   = ("map", "ndcg_cut_10", "P_10", "recall_1000")


def grid_search(
    index_ref: pt.IndexRef,
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    k1_grid: Iterable[float] = DEFAULT_K1_GRID,
    b_grid:  Iterable[float] = DEFAULT_B_GRID,
    metrics: Iterable[str]   = DEFAULT_METRICS,
    select_metric: str       = "map",
) -> tuple[pd.DataFrame, dict]:
    """Run all (k1, b) combinations in a single ``pt.Experiment`` call.

    Returns
    -------
    (grid_df, best_row)
        ``grid_df`` has one row per (k1, b) cell with the requested
        metrics plus ``k1`` and ``b`` columns.  ``best_row`` is the
        single winning row (per ``select_metric``) as a plain dict.
    """
    k1_grid = list(k1_grid)
    b_grid = list(b_grid)

    pipelines: dict[str, pt.Transformer] = {}
    for k1 in k1_grid:
        for b in b_grid:
            pipelines[f"k1={k1:.2f}_b={b:.2f}"] = bm25(index_ref, k1=k1, b=b)

    df = pt.Experiment(
        list(pipelines.values()),
        topics,
        qrels,
        eval_metrics=list(metrics),
        names=list(pipelines.keys()),
    )

    df["k1"] = df["name"].str.extract(r"k1=([\d.]+)").astype(float)
    df["b"]  = df["name"].str.extract(r"b=([\d.]+)").astype(float)

    best_row = df.loc[df[select_metric].idxmax()].to_dict()
    return df, best_row


__all__ = ("DEFAULT_K1_GRID", "DEFAULT_B_GRID", "DEFAULT_METRICS", "grid_search")
