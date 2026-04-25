"""Build a Terrier inverted index over the AP 1988 corpus.

Two named index variants are supported so that we can ablate the
contribution of preprocessing later:

- ``stemmed``    : Porter stemmer + Terrier English stoplist (default)
- ``unstemmed``  : no stemming, no stopwords (raw vocabulary)

The body field indexed for each document is ``HEAD + " " + TEXT``
(headlines often carry strong topical signal in newswire text).

If the index already exists on disk we just load it; rebuilding is an
explicit opt-in via ``force=True``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import pyterrier as pt

from ir596.config import INDEX_DIR
from ir596.io.parse_corpus import iter_docs

IndexVariant = Literal["stemmed", "unstemmed"]

_VARIANT_KWARGS: dict[IndexVariant, dict] = {
    "stemmed":   {"stemmer": "porter", "stopwords": "terrier"},
    "unstemmed": {"stemmer": "none",   "stopwords": "none"},
}


def index_path(variant: IndexVariant) -> Path:
    return INDEX_DIR / variant


def _ensure_pt() -> None:
    if not pt.java.started():
        pt.java.init()


def _doc_iter() -> Iterator[dict]:
    """Yield Terrier-shaped dicts: ``{'docno': ..., 'text': ...}``."""
    for d in iter_docs():
        body = d["head"] + " " + d["text"] if d["head"] else d["text"]
        yield {"docno": d["docno"], "text": body}


def build(variant: IndexVariant = "stemmed", *, force: bool = False) -> pt.IndexRef:
    """Build (or load) the named index. Returns a Terrier ``IndexRef``."""
    _ensure_pt()
    out = index_path(variant)
    props_file = out / "data.properties"

    if props_file.exists() and not force:
        return pt.IndexRef.of(str(props_file))

    out.mkdir(parents=True, exist_ok=True)
    indexer = pt.IterDictIndexer(
        str(out),
        meta={"docno": 32},
        text_attrs=["text"],
        overwrite=True,
        **_VARIANT_KWARGS[variant],
    )
    index_ref = indexer.index(_doc_iter())
    return index_ref


def load(variant: IndexVariant = "stemmed") -> pt.IndexRef:
    """Load a previously-built index. Raises if it doesn't exist."""
    _ensure_pt()
    props_file = index_path(variant) / "data.properties"
    if not props_file.exists():
        raise FileNotFoundError(
            f"Index variant '{variant}' not found at {props_file}. "
            "Run build() first."
        )
    return pt.IndexRef.of(str(props_file))


__all__ = ("IndexVariant", "build", "load", "index_path")
