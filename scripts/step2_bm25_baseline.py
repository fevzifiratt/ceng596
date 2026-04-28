"""Step 2 — BM25 baseline.

Builds two Terrier indexes (stemmed + unstemmed) and runs BM25 over each,
once with the topic ``title`` and once with ``title + description``.
That gives 4 baseline configurations in a single ablation table.

Outputs::

    index/stemmed/                       Terrier index files
    index/unstemmed/                     Terrier index files
    runs/bm25_<index>_<field>.trec       trec_eval-formatted run files
    evals/summary.csv                    appended with one row per config

Run from the repo root::

    .venv/bin/python scripts/step2_bm25_baseline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyterrier as pt

from ir596.evaluate import evaluate
from ir596.index.build_index import IndexVariant, build, index_path
from ir596.retrieval.bm25 import bm25
from ir596.retrieval.topics import QueryField, topics_df


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def ensure_index(variant: IndexVariant) -> pt.IndexRef:
    out = index_path(variant)
    if (out / "data.properties").exists():
        print(f"index '{variant}' already at {out.relative_to(REPO_ROOT)} — reusing")
        return build(variant)
    print(f"building '{variant}' index at {out.relative_to(REPO_ROOT)} ...")
    t0 = time.perf_counter()
    ref = build(variant, force=True)
    print(f"  built in {time.perf_counter() - t0:.1f}s")
    return ref


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    _hr("Indexes")
    refs: dict[IndexVariant, pt.IndexRef] = {
        "stemmed": ensure_index("stemmed"),
        "unstemmed": ensure_index("unstemmed"),
    }

    for variant, ref in refs.items():
        idx = pt.IndexFactory.of(ref)
        stats = idx.getCollectionStatistics()
        print(
            f"  {variant:9s} : {stats.getNumberOfDocuments():,} docs, "
            f"{stats.getNumberOfUniqueTerms():,} terms, "
            f"{stats.getNumberOfTokens():,} tokens"
        )

    _hr("Topics")
    fields: tuple[QueryField, ...] = ("title", "title_desc")
    topic_dfs = {f: topics_df(f) for f in fields}
    for f, df in topic_dfs.items():
        avg_len = df["query"].str.split().map(len).mean()
        print(f"  {f:12s} : {len(df)} queries, avg {avg_len:.1f} tokens/query")

    _hr("Run + evaluate")
    pipelines: dict[str, pt.Transformer] = {}
    for variant in refs:
        for f in fields:
            name = f"bm25_{variant}_{f}"
            pipelines[name] = bm25(refs[variant])
    flat_topics = topic_dfs["title"]
    flat_topics_td = topic_dfs["title_desc"]

    summary_rows = []
    for variant in refs:
        for f in fields:
            name = f"bm25_{variant}_{f}"
            print(f"  running {name} ...")
            df = evaluate(
                {name: bm25(refs[variant])},
                topic_dfs[f],
                extra_columns={"index": variant, "query_field": f, "model": "BM25"},
            )
            summary_rows.append(df)

    _hr("Summary")
    import pandas as pd
    summary = pd.concat(summary_rows, ignore_index=True)
    cols = ["name", "map", "ndcg_cut_10", "P_10", "recall_1000"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRun files written under:", (REPO_ROOT / "runs").relative_to(REPO_ROOT))
    print("Summary CSV:", (REPO_ROOT / "evals" / "summary.csv").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
