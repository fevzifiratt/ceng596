"""Tolerant retrieval with BK-tree spelling variants and permuterm wildcards."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
import gzip
import pickle
from pathlib import Path
import re
from typing import Iterable

import pyterrier as pt

from ir596.config import INDEX_DIR
from ir596.io.parse_corpus import iter_docs
from ir596.retrieval.bm25 import bm25
from ir596.retrieval.topics import sanitise

_CACHE_VERSION = 1
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9*]*$")
_QUERY_RE = re.compile(r"[^a-z0-9* ]+")
_WS_RE = re.compile(r"\s+")
_STOP_TERMS = {
    "about", "after", "before", "being", "between", "document", "documents",
    "discuss", "discusses", "identify", "identifies", "information", "provide",
    "provides", "refer", "refers", "report", "reports", "reveal", "reveals",
    "using", "used", "would", "could", "should", "those", "these", "there",
    "their", "where", "which", "throughout", "because", "forced", "condition",
    "conditions", "plans", "major", "world", "process", "political",
    "implications", "with", "from", "that", "this", "have", "has", "will",
}


def _tokenise(text: str) -> list[str]:
    return [
        token
        for token in sanitise(text).split()
        if len(token) >= 3 and token not in _STOP_TERMS and "*" not in token
    ]


def _normalise_query(query: str) -> list[str]:
    query = _QUERY_RE.sub(" ", query.lower())
    query = _WS_RE.sub(" ", query).strip()
    if not query:
        return []
    return [term for term in query.split() if _TOKEN_RE.fullmatch(term)]


def _rotations(term: str) -> list[str]:
    base = f"{term}$"
    return [base[i:] + base[:i] for i in range(len(base))]


def _edit_distance_with_cutoff(a: str, b: str, cutoff: int) -> int:
    if abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            row_min = min(row_min, cur[-1])
        if row_min > cutoff:
            return cutoff + 1
        prev = cur
    return prev[-1]


@dataclass
class BKNode:
    term: str
    children: dict[int, "BKNode"] = field(default_factory=dict)

    def insert(self, term: str) -> None:
        node = self
        while True:
            dist = _edit_distance_with_cutoff(
                term,
                node.term,
                cutoff=max(len(term), len(node.term)),
            )
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = BKNode(term)
                return
            node = child

    def search(self, term: str, max_distance: int) -> list[tuple[int, str]]:
        matches: list[tuple[int, str]] = []
        stack = [self]
        while stack:
            node = stack.pop()
            dist = _edit_distance_with_cutoff(term, node.term, max_distance)
            if dist <= max_distance:
                matches.append((dist, node.term))
            low = dist - max_distance
            high = dist + max_distance
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return matches


@dataclass(frozen=True)
class TolerantIndex:
    vocab: tuple[str, ...]
    term_freqs: dict[str, int]
    bk_root: BKNode | None
    permuterm: tuple[tuple[str, str], ...]


def cache_path() -> Path:
    return INDEX_DIR / "tolerant_retrieval.pkl.gz"


def build_index(*, force: bool = False, path: Path | None = None) -> TolerantIndex:
    out = path or cache_path()
    if out.exists() and not force:
        return load_index(path=out)

    freqs: Counter[str] = Counter()
    for doc in iter_docs():
        freqs.update(_tokenise(f"{doc['head']} {doc['text']}"))

    vocab = tuple(sorted(term for term, freq in freqs.items() if freq >= 2))
    root = BKNode(vocab[0]) if vocab else None
    if root is not None:
        for term in vocab[1:]:
            root.insert(term)

    rotations: list[tuple[str, str]] = []
    for term in vocab:
        rotations.extend((rotation, term) for rotation in _rotations(term))
    rotations.sort()

    index = TolerantIndex(
        vocab=vocab,
        term_freqs=dict(freqs),
        bk_root=root,
        permuterm=tuple(rotations),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as f:
        pickle.dump(
            {"version": _CACHE_VERSION, "index": index},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return index


def load_index(*, path: Path | None = None) -> TolerantIndex:
    src = path or cache_path()
    if not src.exists():
        raise FileNotFoundError(
            f"Tolerant retrieval cache not found at {src}. Run build_index() first."
        )
    with gzip.open(src, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != _CACHE_VERSION:
        raise ValueError(
            f"Unsupported tolerant retrieval cache version at {src}. Rebuild it."
        )
    return payload["index"]


def _wildcard_matches(
    index: TolerantIndex,
    pattern: str,
    *,
    max_matches: int = 10,
) -> tuple[str, ...]:
    if pattern.count("*") != 1:
        return ()
    prefix, suffix = pattern.split("*", 1)
    key = f"{suffix}${prefix}"
    hi = key + "\uffff"
    left = bisect_left(index.permuterm, (key, ""))
    right = bisect_right(index.permuterm, (hi, ""))

    matches: list[str] = []
    seen: set[str] = set()
    for _, term in index.permuterm[left:right]:
        if term in seen:
            continue
        seen.add(term)
        matches.append(term)
        if len(matches) >= max_matches:
            break
    matches.sort(key=lambda term: (-index.term_freqs.get(term, 0), term))
    return tuple(matches[:max_matches])


def _spelling_variants(
    index: TolerantIndex,
    term: str,
    *,
    max_distance: int = 2,
    max_terms: int = 2,
) -> tuple[str, ...]:
    if index.bk_root is None:
        return ()
    matches = index.bk_root.search(term, max_distance)
    ranked = sorted(
        (
            (distance, candidate)
            for distance, candidate in matches
            if candidate != term and candidate not in _STOP_TERMS
        ),
        key=lambda item: (item[0], -index.term_freqs.get(item[1], 0), item[1]),
    )
    return tuple(candidate for _, candidate in ranked[:max_terms])


def rewrite_query(
    index: TolerantIndex,
    query: str,
    *,
    rare_df_threshold: int = 2,
    max_edit_distance: int = 2,
    max_spelling_variants: int = 2,
    max_wildcard_matches: int = 10,
    expand_only_first_n_terms: int = 8,
) -> str:
    terms = _normalise_query(query)
    if not terms:
        return ""

    expanded = list(terms)
    seen = set(terms)
    for idx, term in enumerate(terms):
        if idx >= expand_only_first_n_terms:
            break
        if len(term) < 3 or term in _STOP_TERMS:
            continue

        if "*" in term:
            candidates = _wildcard_matches(
                index,
                term,
                max_matches=max_wildcard_matches,
            )
        else:
            freq = index.term_freqs.get(term, 0)
            candidates = ()
            if freq == 0 or freq <= rare_df_threshold:
                candidates = _spelling_variants(
                    index,
                    term,
                    max_distance=max_edit_distance,
                    max_terms=max_spelling_variants,
                )

        for candidate in candidates:
            if candidate in seen:
                continue
            expanded.append(candidate)
            seen.add(candidate)

    return " ".join(expanded)


def expand_topics(
    index: TolerantIndex,
    *,
    rare_df_threshold: int = 2,
    max_edit_distance: int = 2,
    max_spelling_variants: int = 2,
    max_wildcard_matches: int = 10,
    expand_only_first_n_terms: int = 8,
) -> pt.Transformer:
    return pt.apply.query(
        lambda row: rewrite_query(
            index,
            row.query,
            rare_df_threshold=rare_df_threshold,
            max_edit_distance=max_edit_distance,
            max_spelling_variants=max_spelling_variants,
            max_wildcard_matches=max_wildcard_matches,
            expand_only_first_n_terms=expand_only_first_n_terms,
        )
    )


def tolerant_bm25(
    index_ref: pt.IndexRef,
    tolerant_index: TolerantIndex,
    *,
    k1: float,
    b: float,
    rare_df_threshold: int = 2,
    max_edit_distance: int = 2,
    max_spelling_variants: int = 2,
    max_wildcard_matches: int = 10,
    expand_only_first_n_terms: int = 8,
    num_results: int = 1000,
) -> pt.Transformer:
    return expand_topics(
        tolerant_index,
        rare_df_threshold=rare_df_threshold,
        max_edit_distance=max_edit_distance,
        max_spelling_variants=max_spelling_variants,
        max_wildcard_matches=max_wildcard_matches,
        expand_only_first_n_terms=expand_only_first_n_terms,
    ) >> bm25(index_ref, k1=k1, b=b, num_results=num_results)


def preview_rewrites(
    index: TolerantIndex,
    queries: Iterable[str],
    *,
    rare_df_threshold: int = 2,
    max_edit_distance: int = 2,
    max_spelling_variants: int = 2,
    max_wildcard_matches: int = 10,
    expand_only_first_n_terms: int = 8,
) -> list[tuple[str, str]]:
    return [
        (
            query,
            rewrite_query(
                index,
                query,
                rare_df_threshold=rare_df_threshold,
                max_edit_distance=max_edit_distance,
                max_spelling_variants=max_spelling_variants,
                max_wildcard_matches=max_wildcard_matches,
                expand_only_first_n_terms=expand_only_first_n_terms,
            ),
        )
        for query in queries
    ]


__all__ = (
    "TolerantIndex",
    "build_index",
    "cache_path",
    "expand_topics",
    "load_index",
    "preview_rewrites",
    "rewrite_query",
    "tolerant_bm25",
)
