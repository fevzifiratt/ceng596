"""Evaluation helpers: run a pipeline over topics, score with trec metrics,
write a TREC run file, and append a row to ``evals/summary.csv``.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import pyterrier as pt

from ir596.config import EVALS_DIR, QRELS_TREC, RUNS_DIR

DEFAULT_METRICS: tuple[str, ...] = ("map", "ndcg_cut_10", "P_10", "recall_1000")


def _load_qrels() -> pd.DataFrame:
    """Load qrels.trec as a DataFrame: qid, docno, label."""
    qrels = pd.read_csv(
        QRELS_TREC, sep=r"\s+", header=None,
        names=["qid", "iter", "docno", "label"], dtype={"qid": str, "docno": str},
    )
    return qrels[["qid", "docno", "label"]]


def write_run(
    results: pd.DataFrame,
    name: str,
    *,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Write a results DataFrame as a 6-column trec_eval run file."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = runs_dir / f"{name}.trec"
    df = results.copy()
    df["Q0"] = "Q0"
    df["tag"] = name
    df = df.sort_values(["qid", "rank"])
    df[["qid", "Q0", "docno", "rank", "score", "tag"]].to_csv(
        out, sep=" ", header=False, index=False
    )
    return out


def evaluate(
    pipelines: Mapping[str, pt.Transformer],
    topics: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    extra_columns: Mapping[str, str] | None = None,
    summary_path: Path = EVALS_DIR / "summary.csv",
) -> pd.DataFrame:
    """Run each named pipeline, write a run file, evaluate, append summary.

    Returns the evaluation DataFrame (one row per pipeline).
    """
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    qrels = _load_qrels()

    metrics = list(metrics)
    eval_df = pt.Experiment(
        list(pipelines.values()),
        topics,
        qrels,
        eval_metrics=metrics,
        names=list(pipelines.keys()),
    )

    for name, pipe in pipelines.items():
        results = pipe.transform(topics)
        write_run(results, name)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = eval_df.copy()
    rows.insert(0, "timestamp", stamp)
    if extra_columns:
        for k, v in extra_columns.items():
            rows[k] = v

    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        merged = pd.concat([prev, rows], ignore_index=True)
    else:
        merged = rows
    merged.to_csv(summary_path, index=False)

    return eval_df


__all__ = ("DEFAULT_METRICS", "evaluate", "write_run")
