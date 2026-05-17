"""Step 8 — Tolerant retrieval with spelling variants and wildcard support."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyterrier as pt

from ir596.config import EVALS_DIR
from ir596.evaluate import evaluate
from ir596.index.build_index import build
from ir596.retrieval.tolerant import build_index as build_tolerant_index
from ir596.retrieval.tolerant import preview_rewrites, tolerant_bm25
from ir596.retrieval.topics import topics_df

TUNED_K1 = 1.8
TUNED_B = 0.5
TUNED_BM25_MAP = 0.3396335336530723
RARE_DF_THRESHOLD = 2
MAX_EDIT_DISTANCE = 2
MAX_SPELLING_VARIANTS = 2
MAX_WILDCARD_MATCHES = 10
EXPAND_FIRST_N = 8


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    _hr("Setup")
    print("index            : stemmed")
    print("query field      : title + desc")
    print(f"locked BM25      : k1={TUNED_K1:.2f}, b={TUNED_B:.2f}")
    print(
        f"tolerant config  : rare_df<={RARE_DF_THRESHOLD}, "
        f"max_edit={MAX_EDIT_DISTANCE}, max_variants={MAX_SPELLING_VARIANTS}, "
        f"wildcard_matches={MAX_WILDCARD_MATCHES}, expand_first_n={EXPAND_FIRST_N}"
    )

    _hr("Build or load tolerant index")
    t0 = time.perf_counter()
    tolerant_index = build_tolerant_index()
    elapsed = time.perf_counter() - t0
    print(f"  vocab size       : {len(tolerant_index.vocab):,}")
    print(f"  index ready in   : {elapsed:.1f}s")

    idx_ref = build("stemmed")
    topics = topics_df("title_desc")

    _hr("Preview rewrites")
    shown = 0
    for original, expanded in preview_rewrites(
        tolerant_index,
        topics["query"].head(15),
        rare_df_threshold=RARE_DF_THRESHOLD,
        max_edit_distance=MAX_EDIT_DISTANCE,
        max_spelling_variants=MAX_SPELLING_VARIANTS,
        max_wildcard_matches=MAX_WILDCARD_MATCHES,
        expand_only_first_n_terms=EXPAND_FIRST_N,
    ):
        if original != expanded:
            print(f"  q: {original}")
            print(f"  e: {expanded}")
            shown += 1
    if shown == 0:
        print("  no query rewrites triggered in the first 15 topics")

    _hr("Run + evaluate")
    eval_df = evaluate(
        {
            "tolerant_tuned_title_desc": tolerant_bm25(
                idx_ref,
                tolerant_index,
                k1=TUNED_K1,
                b=TUNED_B,
                rare_df_threshold=RARE_DF_THRESHOLD,
                max_edit_distance=MAX_EDIT_DISTANCE,
                max_spelling_variants=MAX_SPELLING_VARIANTS,
                max_wildcard_matches=MAX_WILDCARD_MATCHES,
                expand_only_first_n_terms=EXPAND_FIRST_N,
            )
        },
        topics,
        extra_columns={
            "index": "stemmed",
            "query_field": "title_desc",
            "model": "Tolerant-BM25",
            "k1": TUNED_K1,
            "b": TUNED_B,
            "rare_df_threshold": RARE_DF_THRESHOLD,
            "max_edit_distance": MAX_EDIT_DISTANCE,
            "max_spelling_variants": MAX_SPELLING_VARIANTS,
            "max_wildcard_matches": MAX_WILDCARD_MATCHES,
            "expand_only_first_n_terms": EXPAND_FIRST_N,
        },
    )

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALS_DIR / "tolerant_retrieval.csv"
    eval_df.to_csv(out, index=False)

    score = float(eval_df.iloc[0]["map"])
    delta = score - TUNED_BM25_MAP

    _hr("Summary")
    print(eval_df[["name", "map", "ndcg_cut_10", "P_10", "recall_1000"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print()
    print(
        f"MAP delta vs tuned BM25 : {TUNED_BM25_MAP:.4f} -> {score:.4f} "
        f"({delta:+.4f})"
    )
    print(f"\nSaved comparison table : {out.relative_to(REPO_ROOT)}")
    print("Run file written under : runs/tolerant_tuned_title_desc.trec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
