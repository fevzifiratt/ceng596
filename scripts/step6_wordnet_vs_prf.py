"""Step 6 — WordNet synonym expansion vs PRF baseline."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import pyterrier as pt

from ir596.config import EVALS_DIR
from ir596.evaluate import evaluate
from ir596.index.build_index import build
from ir596.retrieval.topics import topics_df
from ir596.retrieval.wordnet import preview_expansions, wordnet_bm25

TUNED_K1 = 1.8
TUNED_B = 0.5
TUNED_BM25_MAP = 0.3396335336530723
DEFAULT_PRF_NAME = "prf_bo1_d3_t10_title_desc"
DEFAULT_PRF_MAP = 0.3737088690924453
MAX_SYNONYMS = 1
MAX_ADDED = 4
MIN_TERM_LEN = 4
EXPAND_FIRST_N = 8


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def _best_prf_baseline() -> tuple[str, float]:
    grid = EVALS_DIR / "prf_grid.csv"
    if not grid.exists():
        return DEFAULT_PRF_NAME, DEFAULT_PRF_MAP
    df = pd.read_csv(grid)
    row = df.loc[df["map"].idxmax()]
    return str(row["name"]), float(row["map"])


def main() -> int:
    if not pt.java.started():
        pt.java.init()

    prf_name, prf_map = _best_prf_baseline()

    _hr("Setup")
    print("index           : stemmed")
    print("query field     : title + desc")
    print(f"locked BM25     : k1={TUNED_K1:.2f}, b={TUNED_B:.2f}")
    print(f"PRF baseline    : {prf_name}  MAP={prf_map:.4f}")
    print(
        f"WordNet config  : synonyms/term={MAX_SYNONYMS}, max_added={MAX_ADDED}, "
        f"min_len={MIN_TERM_LEN}, expand_first_n={EXPAND_FIRST_N}"
    )

    idx_ref = build("stemmed")
    topics = topics_df("title_desc")

    _hr("Preview expansions")
    for original, expanded in preview_expansions(
        topics["query"].head(5),
        max_synonyms_per_term=MAX_SYNONYMS,
        max_added_terms=MAX_ADDED,
        min_term_len=MIN_TERM_LEN,
        expand_only_first_n_terms=EXPAND_FIRST_N,
    ):
        print(f"  q: {original}")
        print(f"  e: {expanded}")

    _hr("Run + evaluate")
    eval_df = evaluate(
        {
            "wordnet_tuned_title_desc": wordnet_bm25(
                idx_ref,
                k1=TUNED_K1,
                b=TUNED_B,
                max_synonyms_per_term=MAX_SYNONYMS,
                max_added_terms=MAX_ADDED,
                min_term_len=MIN_TERM_LEN,
                expand_only_first_n_terms=EXPAND_FIRST_N,
            )
        },
        topics,
        extra_columns={
            "index": "stemmed",
            "query_field": "title_desc",
            "model": "WordNet-BM25",
            "k1": TUNED_K1,
            "b": TUNED_B,
            "max_synonyms_per_term": MAX_SYNONYMS,
            "max_added_terms": MAX_ADDED,
            "min_term_len": MIN_TERM_LEN,
            "expand_only_first_n_terms": EXPAND_FIRST_N,
        },
    )

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALS_DIR / "wordnet_vs_prf.csv"
    eval_df.to_csv(out, index=False)

    wn_map = float(eval_df.iloc[0]["map"])
    delta_tuned = wn_map - TUNED_BM25_MAP
    delta_prf = wn_map - prf_map

    _hr("Summary")
    print(eval_df[["name", "map", "ndcg_cut_10", "P_10", "recall_1000"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print()
    print(
        f"MAP delta vs tuned BM25 : {TUNED_BM25_MAP:.4f} -> {wn_map:.4f} "
        f"({delta_tuned:+.4f})"
    )
    print(
        f"MAP delta vs PRF        : {prf_map:.4f} -> {wn_map:.4f} "
        f"({delta_prf:+.4f})"
    )
    print(f"\nSaved comparison table : {out.relative_to(REPO_ROOT)}")
    print("Run file written under : runs/wordnet_tuned_title_desc.trec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
