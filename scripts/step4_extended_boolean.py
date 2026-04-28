"""Step 4 — Extended Boolean retrieval.

Builds a lightweight postings cache directly from the AP corpus and
evaluates a small p-norm grid with the existing TREC evaluation flow.

Outputs::

    index/ext_boolean_nostop.pkl.gz       serialized postings cache
    runs/extbool_*.trec                   trec_eval-formatted run files
    evals/summary.csv                     appended with one row per config
"""
from __future__ import annotations

import sys
from pathlib import Path
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from ir596.evaluate import evaluate
from ir596.retrieval.extended_boolean import build_index, cache_path, extended_boolean
from ir596.retrieval.topics import QueryField, topics_df


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    _hr("Cache")
    out = cache_path(remove_stopwords=True)
    if out.exists():
        print(f"cache already at {out.relative_to(REPO_ROOT)} — reusing")
        cache = build_index(remove_stopwords=True)
    else:
        print(f"building cache at {out.relative_to(REPO_ROOT)} ...")
        t0 = time.perf_counter()
        cache = build_index(remove_stopwords=True, force=True)
        print(f"  built in {time.perf_counter() - t0:.1f}s")

    print(
        f"  docs={cache.num_docs:,} vocab={len(cache.postings):,} "
        f"stopwords_removed={cache.remove_stopwords}"
    )

    _hr("Topics")
    fields: tuple[QueryField, ...] = ("title", "title_desc")
    topic_dfs = {field: topics_df(field) for field in fields}
    for field, df in topic_dfs.items():
        avg_len = df["query"].str.split().map(len).mean()
        print(f"  {field:12s} : {len(df)} queries, avg {avg_len:.1f} tokens/query")

    _hr("Run + evaluate")
    p_values = (1.5, 2.0, 3.0)
    summary_rows = []
    for field in fields:
        for p in p_values:
            name = f"extbool_or_p{str(p).replace('.', '_')}_{field}"
            print(f"  running {name} ...")
            df = evaluate(
                {name: extended_boolean(remove_stopwords=True, mode='or', p=p)},
                topic_dfs[field],
                extra_columns={
                    "query_field": field,
                    "model": "extended_boolean",
                    "mode": "or",
                    "p": f"{p:.1f}",
                    "stopwords_removed": "true",
                },
            )
            summary_rows.append(df)

    _hr("Summary")
    summary = pd.concat(summary_rows, ignore_index=True)
    cols = ["name", "map", "ndcg_cut_10", "P_10", "recall_1000"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRun files written under:", (REPO_ROOT / "runs").relative_to(REPO_ROOT))
    print("Summary CSV:", (REPO_ROOT / "evals" / "summary.csv").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
