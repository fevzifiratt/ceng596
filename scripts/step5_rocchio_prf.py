"""Step 5 — Rocchio-style pseudo-relevance feedback (Stage 2).

Layered on top of the Step-3 tuned BM25 baseline. Sweeps three
Rocchio-family expansion models (Bo1, KL, RM3) with ``fb_docs`` ∈ {3, 5,
10} on the ``title_desc`` query field, plus a single ``title``-only
control that matches the proposal's exact recipe (Bo1, top-5).

Pipeline per config::

    BM25(k1=1.80, b=0.50) >> QE(fb_docs, fb_terms=10) >> BM25(k1=1.80, b=0.50)

Outputs::

    runs/prf_*.trec                  trec_eval-formatted run files
    evals/summary.csv                appended with one row per config
    evals/prf_grid.csv               compact grid table for the report

Run from the repo root::

    .venv/bin/python scripts/step5_rocchio_prf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import pyterrier as pt

from ir596.config import EVALS_DIR
from ir596.evaluate import evaluate
from ir596.index.build_index import build
from ir596.retrieval.prf import QEModel, rocchio_prf
from ir596.retrieval.topics import topics_df

# Locked tuned BM25 baseline from Step 3.
TUNED_K1 = 1.80
TUNED_B = 0.50
BASELINE_MAP = 0.3396  # bm25_tuned_title_desc

QE_MODELS: tuple[QEModel, ...] = ("Bo1", "KL", "RM3")
FB_DOCS_GRID: tuple[int, ...] = (3, 5, 10)
FB_TERMS = 10


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    _hr("Setup")
    print(f"index           : stemmed (Porter + Terrier stoplist)")
    print(f"BM25 hyperparams: k1={TUNED_K1}, b={TUNED_B}  (locked from Step 3)")
    print(f"qe models       : {list(QE_MODELS)}")
    print(f"fb_docs grid    : {list(FB_DOCS_GRID)}")
    print(f"fb_terms        : {FB_TERMS}")

    idx_ref = build("stemmed")
    topics_t = topics_df("title")
    topics_td = topics_df("title_desc")
    print(f"topics          : title={len(topics_t)}, title_desc={len(topics_td)}")

    configs: list[dict] = []
    for qe in QE_MODELS:
        for k in FB_DOCS_GRID:
            configs.append({
                "name": f"prf_{qe.lower()}_d{k}_t{FB_TERMS}_title_desc",
                "field": "title_desc",
                "qe_model": qe,
                "fb_docs": k,
                "transformer": rocchio_prf(
                    idx_ref, k1=TUNED_K1, b=TUNED_B,
                    qe_model=qe, fb_docs=k, fb_terms=FB_TERMS,
                ),
            })

    # Title-only control on the proposal's recommended config (Bo1 / top-5).
    configs.append({
        "name": f"prf_bo1_d5_t{FB_TERMS}_title",
        "field": "title",
        "qe_model": "Bo1",
        "fb_docs": 5,
        "transformer": rocchio_prf(
            idx_ref, k1=TUNED_K1, b=TUNED_B,
            qe_model="Bo1", fb_docs=5, fb_terms=FB_TERMS,
        ),
    })

    _hr("Run + evaluate")
    summary_rows: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for cfg in configs:
        print(f"  running {cfg['name']} ...")
        topics = topics_td if cfg["field"] == "title_desc" else topics_t
        df = evaluate(
            {cfg["name"]: cfg["transformer"]},
            topics,
            extra_columns={
                "index": "stemmed",
                "query_field": cfg["field"],
                "model": f"BM25+{cfg['qe_model']}",
                "qe_model": cfg["qe_model"],
                "fb_docs": cfg["fb_docs"],
                "fb_terms": FB_TERMS,
                "k1": TUNED_K1,
                "b": TUNED_B,
            },
        )
        summary_rows.append(df)
    elapsed = time.perf_counter() - t0
    print(
        f"  ran {len(configs)} configs in {elapsed:.1f}s "
        f"({elapsed / len(configs):.2f}s per config)"
    )

    _hr("Summary table")
    summary = pd.concat(summary_rows, ignore_index=True)
    cols = ["name", "map", "ndcg_cut_10", "P_10", "recall_1000"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = EVALS_DIR / "prf_grid.csv"
    summary[cols].to_csv(grid_path, index=False)
    print(f"\n  full grid       : {grid_path.relative_to(REPO_ROOT)}")

    _hr("Vs Step 3 tuned baseline (title_desc, MAP=0.3396)")
    td_only = summary[summary["name"].str.endswith("_title_desc")]
    best_row = td_only.loc[td_only["map"].idxmax()]
    delta = best_row["map"] - BASELINE_MAP
    pct = 100 * delta / BASELINE_MAP
    print(
        f"  best PRF : {best_row['name']}  MAP={best_row['map']:.4f}\n"
        f"  vs BM25t : {BASELINE_MAP:.4f}  ({delta:+.4f}, {pct:+.1f}%)"
    )

    print("\nRun files written under:", (REPO_ROOT / "runs").relative_to(REPO_ROOT))
    print("Summary CSV:", (REPO_ROOT / "evals" / "summary.csv").relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
