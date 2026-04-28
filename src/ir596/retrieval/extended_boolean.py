"""Extended Boolean retrieval implemented in pure Python.

This module builds a lightweight postings cache directly from the AP
corpus, then scores queries with the p-norm Extended Boolean model.
It plugs into the existing evaluation flow by exposing a ``transform()``
method compatible with the rest of the codebase.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import math
import pickle
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import pyterrier as pt

from ir596.config import INDEX_DIR
from ir596.io.parse_corpus import iter_docs
from ir596.retrieval.topics import sanitise

ExtBooleanMode = Literal["or", "and"]

_CACHE_VERSION = 1
_DEFAULT_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
})


def _tokenise(text: str, *, remove_stopwords: bool) -> list[str]:
    tokens = sanitise(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS]
    return tokens


@dataclass(frozen=True)
class ExtBooleanIndex:
    """Compact postings cache for Extended Boolean retrieval."""

    docnos: list[str]
    postings: dict[str, list[tuple[int, int]]]
    dfs: dict[str, int]
    num_docs: int
    max_idf: float
    remove_stopwords: bool

    def idf(self, term: str) -> float:
        df = self.dfs.get(term)
        if not df:
            return 0.0
        return math.log1p(self.num_docs / df)

    def normalised_idf(self, term: str) -> float:
        if self.max_idf <= 0.0:
            return 0.0
        return self.idf(term) / self.max_idf


def cache_path(*, remove_stopwords: bool = True) -> Path:
    suffix = "nostop" if remove_stopwords else "raw"
    return INDEX_DIR / f"ext_boolean_{suffix}.pkl.gz"


def build_index(
    *,
    remove_stopwords: bool = True,
    force: bool = False,
    path: Path | None = None,
) -> ExtBooleanIndex:
    """Build (or load) the Extended Boolean postings cache."""
    out = path or cache_path(remove_stopwords=remove_stopwords)
    if out.exists() and not force:
        return load_index(remove_stopwords=remove_stopwords, path=out)

    out.parent.mkdir(parents=True, exist_ok=True)

    docnos: list[str] = []
    postings_acc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    dfs: Counter[str] = Counter()

    for doc_id, doc in enumerate(iter_docs()):
        docnos.append(doc["docno"])
        tokens = _tokenise(
            f"{doc['head']} {doc['text']}",
            remove_stopwords=remove_stopwords,
        )
        if not tokens:
            continue

        counts = Counter(tokens)
        for term, tf in counts.items():
            postings_acc[term].append((doc_id, tf))
            dfs[term] += 1

    num_docs = len(docnos)
    max_idf = max((math.log1p(num_docs / df) for df in dfs.values()), default=0.0)
    index = ExtBooleanIndex(
        docnos=docnos,
        postings=dict(postings_acc),
        dfs=dict(dfs),
        num_docs=num_docs,
        max_idf=max_idf,
        remove_stopwords=remove_stopwords,
    )

    with gzip.open(out, "wb") as f:
        pickle.dump(
            {"version": _CACHE_VERSION, "index": index},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return index


def load_index(
    *,
    remove_stopwords: bool = True,
    path: Path | None = None,
) -> ExtBooleanIndex:
    """Load a previously-built Extended Boolean postings cache."""
    src = path or cache_path(remove_stopwords=remove_stopwords)
    if not src.exists():
        raise FileNotFoundError(
            f"Extended Boolean cache not found at {src}. Run build_index() first."
        )
    with gzip.open(src, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != _CACHE_VERSION:
        raise ValueError(
            f"Unsupported Extended Boolean cache version at {src}. Rebuild it."
        )
    index: ExtBooleanIndex = payload["index"]
    if index.remove_stopwords != remove_stopwords:
        raise ValueError(
            "Cache preprocessing does not match requested remove_stopwords flag."
        )
    return index


class ExtendedBooleanRetriever(pt.Transformer):
    """Retriever compatible with the project's evaluation helper."""

    def __init__(
        self,
        index: ExtBooleanIndex,
        *,
        mode: ExtBooleanMode = "or",
        p: float = 2.0,
        num_results: int = 1000,
        tf_saturation: float = 2.0,
    ) -> None:
        if p <= 0:
            raise ValueError("p must be > 0")
        if tf_saturation <= 0:
            raise ValueError("tf_saturation must be > 0")
        self.index = index
        self.mode = mode
        self.p = p
        self.num_results = num_results
        self.tf_saturation = tf_saturation

    def _query_weights(self, query: str) -> dict[str, float]:
        counts = Counter(
            _tokenise(query, remove_stopwords=self.index.remove_stopwords)
        )
        if not counts:
            return {}

        raw_weights = {}
        for term, qtf in counts.items():
            idf = self.index.normalised_idf(term)
            if idf <= 0.0:
                continue
            raw_weights[term] = (1.0 + math.log1p(qtf)) * idf

        if not raw_weights:
            return {}

        max_weight = max(raw_weights.values())
        return {term: weight / max_weight for term, weight in raw_weights.items()}

    def _term_match_score(self, term: str, tf: int) -> float:
        tf_part = tf / (tf + self.tf_saturation)
        return tf_part * self.index.normalised_idf(term)

    def _score_query(self, query: str) -> list[tuple[int, float]]:
        weights = self._query_weights(query)
        if not weights:
            return []

        denom = sum(weight ** self.p for weight in weights.values())
        if denom <= 0.0:
            return []

        accum: dict[int, float] = defaultdict(float)
        for term, weight in weights.items():
            postings = self.index.postings.get(term, ())
            for doc_id, tf in postings:
                match = self._term_match_score(term, tf)
                if self.mode == "or":
                    accum[doc_id] += (weight * match) ** self.p
                else:
                    accum[doc_id] += (weight ** self.p) - (weight * (1.0 - match)) ** self.p

        scored: list[tuple[int, float]] = []
        for doc_id, value in accum.items():
            if self.mode == "or":
                score = (value / denom) ** (1.0 / self.p)
            else:
                penalty = max(0.0, denom - value)
                score = 1.0 - (penalty / denom) ** (1.0 / self.p)
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda item: (-item[1], self.index.docnos[item[0]]))
        return scored[: self.num_results]

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for row in topics.itertuples(index=False):
            ranking = self._score_query(row.query)
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                rows.append({
                    "qid": str(row.qid),
                    "query": row.query,
                    "docno": self.index.docnos[doc_id],
                    "score": float(score),
                    "rank": rank,
                })
        return pd.DataFrame(rows, columns=["qid", "query", "docno", "score", "rank"])


def extended_boolean(
    *,
    remove_stopwords: bool = True,
    mode: ExtBooleanMode = "or",
    p: float = 2.0,
    num_results: int = 1000,
    tf_saturation: float = 2.0,
    force_rebuild: bool = False,
) -> ExtendedBooleanRetriever:
    """Factory mirroring the BM25 helper style used elsewhere in the repo."""
    index = build_index(remove_stopwords=remove_stopwords, force=force_rebuild)
    return ExtendedBooleanRetriever(
        index,
        mode=mode,
        p=p,
        num_results=num_results,
        tf_saturation=tf_saturation,
    )


__all__: Iterable[str] = (
    "ExtBooleanIndex",
    "ExtBooleanMode",
    "ExtendedBooleanRetriever",
    "build_index",
    "cache_path",
    "extended_boolean",
    "load_index",
)
