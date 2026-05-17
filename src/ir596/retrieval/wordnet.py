"""WordNet synonym expansion on top of tuned BM25."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import pyterrier as pt

from ir596.retrieval.bm25 import bm25
from ir596.retrieval.topics import sanitise

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "NLTK is required for WordNet expansion. Install project dependencies first."
    ) from exc

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*$")
_REPO_NLTK_DATA = Path(__file__).resolve().parents[3] / "nltk_data"
_STOP_TERMS = {
    "about", "after", "before", "being", "between", "document", "documents",
    "discuss", "discusses", "documentary", "identify", "identifies",
    "information", "provide", "provides", "refers", "refer", "report",
    "reports", "reveal", "reveals", "using", "used", "would", "could",
    "should", "those", "these", "there", "their", "where", "which",
    "throughout", "because", "forced", "condition", "conditions", "plans",
    "major", "world", "process", "political", "implications", "will", "with",
}

if str(_REPO_NLTK_DATA) not in nltk.data.path:
    nltk.data.path.insert(0, str(_REPO_NLTK_DATA))


def _ensure_wordnet() -> None:
    try:
        wn.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK WordNet data. "
            f"Run nltk.download('wordnet', download_dir='{_REPO_NLTK_DATA}') "
            f"and nltk.download('omw-1.4', download_dir='{_REPO_NLTK_DATA}')."
        ) from exc


@lru_cache(maxsize=4096)
def _synonyms(term: str, max_synonyms: int) -> tuple[str, ...]:
    _ensure_wordnet()
    base = wn.morphy(term) or term
    synsets = wn.synsets(term)
    if len(synsets) > 6:
        return ()

    scores: dict[str, int] = {}
    for synset in synsets[:2]:
        for lemma in synset.lemmas():
            candidate = sanitise(lemma.name().replace("_", " "))
            if " " in candidate or not _TOKEN_RE.fullmatch(candidate):
                continue
            if candidate == term or candidate in _STOP_TERMS:
                continue
            if (wn.morphy(candidate) or candidate) == base:
                continue
            scores[candidate] = max(scores.get(candidate, 0), lemma.count())

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _ in ranked[:max_synonyms])


def expand_query(
    query: str,
    *,
    max_synonyms_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
) -> str:
    terms = [t for t in sanitise(query).split() if _TOKEN_RE.fullmatch(t)]
    if not terms:
        return ""

    expanded = list(terms)
    seen = set(terms)
    added = 0

    for idx, term in enumerate(terms):
        if added >= max_added_terms or idx >= expand_only_first_n_terms:
            break
        if len(term) < min_term_len or term in _STOP_TERMS:
            continue
        for synonym in _synonyms(term, max_synonyms_per_term):
            if synonym in seen or synonym in _STOP_TERMS:
                continue
            expanded.append(synonym)
            seen.add(synonym)
            added += 1
            if added >= max_added_terms:
                break

    return " ".join(expanded)


def expand_topics(
    *,
    max_synonyms_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
) -> pt.Transformer:
    return pt.apply.query(
        lambda row: expand_query(
            row.query,
            max_synonyms_per_term=max_synonyms_per_term,
            max_added_terms=max_added_terms,
            min_term_len=min_term_len,
            expand_only_first_n_terms=expand_only_first_n_terms,
        )
    )


def wordnet_bm25(
    index_ref: pt.IndexRef,
    *,
    k1: float,
    b: float,
    max_synonyms_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
    num_results: int = 1000,
) -> pt.Transformer:
    return expand_topics(
        max_synonyms_per_term=max_synonyms_per_term,
        max_added_terms=max_added_terms,
        min_term_len=min_term_len,
        expand_only_first_n_terms=expand_only_first_n_terms,
    ) >> bm25(index_ref, k1=k1, b=b, num_results=num_results)


def preview_expansions(
    queries: Iterable[str],
    *,
    max_synonyms_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
) -> list[tuple[str, str]]:
    return [
        (
            query,
            expand_query(
                query,
                max_synonyms_per_term=max_synonyms_per_term,
                max_added_terms=max_added_terms,
                min_term_len=min_term_len,
                expand_only_first_n_terms=expand_only_first_n_terms,
            ),
        )
        for query in queries
    ]


__all__ = ("expand_query", "expand_topics", "preview_expansions", "wordnet_bm25")
