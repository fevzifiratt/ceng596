"""AP88-trained Word2Vec query expansion on top of tuned BM25."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import re

from gensim.models import Word2Vec
import pyterrier as pt

from ir596.config import INDEX_DIR
from ir596.io.parse_corpus import iter_docs
from ir596.retrieval.bm25 import bm25
from ir596.retrieval.topics import sanitise

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*$")
_STOP_TERMS = {
    "about", "after", "before", "being", "between", "document", "documents",
    "discuss", "discusses", "identify", "identifies", "information", "provide",
    "provides", "refer", "refers", "report", "reports", "reveal", "reveals",
    "using", "used", "would", "could", "should", "those", "these", "there",
    "their", "where", "which", "throughout", "because", "forced", "condition",
    "conditions", "plans", "major", "world", "process", "political",
    "implications", "will", "with",
}

MODEL_FILENAME = "word2vec_ap88.model"


def model_path() -> Path:
    return INDEX_DIR / MODEL_FILENAME


def _tokenise(text: str, *, max_tokens_per_doc: int) -> list[str]:
    tokens = [
        token
        for token in sanitise(text).split()
        if len(token) >= 3 and _TOKEN_RE.fullmatch(token) and token not in _STOP_TERMS
    ]
    return tokens[:max_tokens_per_doc]


class _SentenceIterator:
    def __init__(self, *, max_tokens_per_doc: int = 80) -> None:
        self.max_tokens_per_doc = max_tokens_per_doc

    def __iter__(self) -> Iterator[list[str]]:
        for doc in iter_docs():
            tokens = _tokenise(
                f"{doc['head']} {doc['text']}",
                max_tokens_per_doc=self.max_tokens_per_doc,
            )
            if tokens:
                yield tokens


def build_model(
    *,
    force: bool = False,
    vector_size: int = 64,
    window: int = 4,
    min_count: int = 10,
    sg: int = 1,
    epochs: int = 3,
    negative: int = 8,
    workers: int = 1,
    seed: int = 42,
    max_tokens_per_doc: int = 80,
    path: Path | None = None,
) -> Word2Vec:
    out = path or model_path()
    if out.exists() and not force:
        model = Word2Vec.load(str(out))
        if (
            model.wv.vector_size == vector_size
            and model.window == window
            and model.min_count == min_count
            and model.sg == sg
            and model.negative == negative
        ):
            return model

    out.parent.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(
        sentences=_SentenceIterator(max_tokens_per_doc=max_tokens_per_doc),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        negative=negative,
        workers=workers,
        seed=seed,
        sample=1e-4,
    )
    model.save(str(out))
    return model


def load_model(*, path: Path | None = None) -> Word2Vec:
    src = path or model_path()
    if not src.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found at {src}. Run build_model() first."
        )
    return Word2Vec.load(str(src))


def _similar_terms(
    model: Word2Vec,
    term: str,
    *,
    topn: int = 10,
    min_similarity: float = 0.45,
    max_terms: int = 1,
) -> tuple[str, ...]:
    if term not in model.wv:
        return ()

    seen: list[str] = []
    for candidate, similarity in model.wv.most_similar(term, topn=topn):
        if similarity < min_similarity:
            continue
        if (
            not _TOKEN_RE.fullmatch(candidate)
            or candidate == term
            or candidate in _STOP_TERMS
            or candidate in seen
        ):
            continue
        seen.append(candidate)
        if len(seen) >= max_terms:
            break
    return tuple(seen)


def expand_query(
    model: Word2Vec,
    query: str,
    *,
    max_neighbors_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
    min_similarity: float = 0.45,
) -> str:
    terms = [term for term in sanitise(query).split() if _TOKEN_RE.fullmatch(term)]
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
        for candidate in _similar_terms(
            model,
            term,
            topn=10,
            min_similarity=min_similarity,
            max_terms=max_neighbors_per_term,
        ):
            if candidate in seen:
                continue
            expanded.append(candidate)
            seen.add(candidate)
            added += 1
            if added >= max_added_terms:
                break

    return " ".join(expanded)


def expand_topics(
    model: Word2Vec,
    *,
    max_neighbors_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
    min_similarity: float = 0.45,
) -> pt.Transformer:
    return pt.apply.query(
        lambda row: expand_query(
            model,
            row.query,
            max_neighbors_per_term=max_neighbors_per_term,
            max_added_terms=max_added_terms,
            min_term_len=min_term_len,
            expand_only_first_n_terms=expand_only_first_n_terms,
            min_similarity=min_similarity,
        )
    )


def word2vec_bm25(
    index_ref: pt.IndexRef,
    model: Word2Vec,
    *,
    k1: float,
    b: float,
    max_neighbors_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
    min_similarity: float = 0.45,
    num_results: int = 1000,
) -> pt.Transformer:
    return expand_topics(
        model,
        max_neighbors_per_term=max_neighbors_per_term,
        max_added_terms=max_added_terms,
        min_term_len=min_term_len,
        expand_only_first_n_terms=expand_only_first_n_terms,
        min_similarity=min_similarity,
    ) >> bm25(index_ref, k1=k1, b=b, num_results=num_results)


def preview_expansions(
    model: Word2Vec,
    queries: Iterable[str],
    *,
    max_neighbors_per_term: int = 1,
    max_added_terms: int = 4,
    min_term_len: int = 4,
    expand_only_first_n_terms: int = 8,
    min_similarity: float = 0.45,
) -> list[tuple[str, str]]:
    return [
        (
            query,
            expand_query(
                model,
                query,
                max_neighbors_per_term=max_neighbors_per_term,
                max_added_terms=max_added_terms,
                min_term_len=min_term_len,
                expand_only_first_n_terms=expand_only_first_n_terms,
                min_similarity=min_similarity,
            ),
        )
        for query in queries
    ]


__all__ = (
    "build_model",
    "expand_query",
    "expand_topics",
    "load_model",
    "model_path",
    "preview_expansions",
    "word2vec_bm25",
)
