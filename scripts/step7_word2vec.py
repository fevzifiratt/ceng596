"""Step 7 — Word2Vec domain-specific expansion trained on AP88."""
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
from ir596.retrieval.topics import topics_df
from ir596.retrieval.word2vec import build_model, preview_expansions, word2vec_bm25

TUNED_K1 = 1.8
TUNED_B = 0.5
TUNED_BM25_MAP = 0.3396335336530723
MAX_NEIGHBORS_PER_TERM = 1
MAX_ADDED_TERMS = 4
MIN_TERM_LEN = 4
EXPAND_FIRST_N = 8
MIN_SIMILARITY = 0.45


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    _hr("Setup")
    print("index           : stemmed")
    print("query field     : title + desc")
    print(f"locked BM25     : k1={TUNED_K1:.2f}, b={TUNED_B:.2f}")
    print(
        f"Word2Vec config : neighbors/term={MAX_NEIGHBORS_PER_TERM}, "
        f"max_added={MAX_ADDED_TERMS}, min_len={MIN_TERM_LEN}, "
        f"expand_first_n={EXPAND_FIRST_N}, min_sim={MIN_SIMILARITY:.2f}"
    )

    _hr("Train or load model")
    t0 = time.perf_counter()
    model = build_model()
    elapsed = time.perf_counter() - t0
    print(f"  vocab size      : {len(model.wv):,}")
    print(f"  vector size     : {model.wv.vector_size}")
    print(f"  model ready in  : {elapsed:.1f}s")

    idx_ref = build("stemmed")
    topics = topics_df("title_desc")

    _hr("Preview expansions")
    for original, expanded in preview_expansions(
        model,
        topics["query"].head(5),
        max_neighbors_per_term=MAX_NEIGHBORS_PER_TERM,
        max_added_terms=MAX_ADDED_TERMS,
        min_term_len=MIN_TERM_LEN,
        expand_only_first_n_terms=EXPAND_FIRST_N,
        min_similarity=MIN_SIMILARITY,
    ):
        print(f"  q: {original}")
        print(f"  e: {expanded}")

    _hr("Run + evaluate")
    eval_df = evaluate(
        {
            "word2vec_tuned_title_desc": word2vec_bm25(
                idx_ref,
                model,
                k1=TUNED_K1,
                b=TUNED_B,
                max_neighbors_per_term=MAX_NEIGHBORS_PER_TERM,
                max_added_terms=MAX_ADDED_TERMS,
                min_term_len=MIN_TERM_LEN,
                expand_only_first_n_terms=EXPAND_FIRST_N,
                min_similarity=MIN_SIMILARITY,
            )
        },
        topics,
        extra_columns={
            "index": "stemmed",
            "query_field": "title_desc",
            "model": "Word2Vec-BM25",
            "k1": TUNED_K1,
            "b": TUNED_B,
            "max_neighbors_per_term": MAX_NEIGHBORS_PER_TERM,
            "max_added_terms": MAX_ADDED_TERMS,
            "min_term_len": MIN_TERM_LEN,
            "expand_only_first_n_terms": EXPAND_FIRST_N,
            "min_similarity": MIN_SIMILARITY,
        },
    )

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALS_DIR / "word2vec_ap88.csv"
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
    print("Run file written under : runs/word2vec_tuned_title_desc.trec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
