"""Streaming parser for the Associated Press TREC-SGML corpus.

The files in ``AP_collection/coll/`` are *not* valid XML (e.g. the tag
``<1ST_LINE>`` starts with a digit, and ``&`` appears unescaped).  So we
use regex-based extraction over each daily file, one ``<DOC>..</DOC>``
block at a time, and yield plain dicts.

Usage::

    from ir596.io.parse_corpus import iter_docs
    for doc in iter_docs():
        print(doc["docno"], len(doc["text"]))
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, TypedDict

from ir596.config import CORPUS_DIR

_DOC_RE = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
_DOCNO_RE = re.compile(r"<DOCNO>\s*(\S+)\s*</DOCNO>")
_HEAD_RE = re.compile(r"<HEAD>(.*?)</HEAD>", re.DOTALL)
_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

_WS_RE = re.compile(r"\s+")


class Doc(TypedDict):
    docno: str
    head: str
    text: str


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def iter_doc_files(corpus_dir: Path = CORPUS_DIR) -> Iterator[Path]:
    """Yield the 322 daily files in deterministic order."""
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus dir not found: {corpus_dir}. "
            "Set AP_DATA_ROOT or fix config.CORPUS_DIR."
        )
    yield from sorted(p for p in corpus_dir.iterdir() if p.is_file())


def iter_docs(corpus_dir: Path = CORPUS_DIR) -> Iterator[Doc]:
    """Stream every <DOC>..</DOC> record across the whole corpus.

    Memory stays bounded: we read one daily file at a time (each is
    at most a few MB) rather than loading the full 94 MB corpus.
    """
    for fp in iter_doc_files(corpus_dir):
        raw = fp.read_text(encoding="latin-1", errors="replace")
        for m in _DOC_RE.finditer(raw):
            block = m.group(1)
            docno_m = _DOCNO_RE.search(block)
            if not docno_m:
                continue
            head_m = _HEAD_RE.search(block)
            text_m = _TEXT_RE.search(block)
            yield Doc(
                docno=docno_m.group(1),
                head=_clean(head_m.group(1)) if head_m else "",
                text=_clean(text_m.group(1)) if text_m else "",
            )


def count_docs(corpus_dir: Path = CORPUS_DIR) -> int:
    return sum(1 for _ in iter_docs(corpus_dir))


def collect_docnos(corpus_dir: Path = CORPUS_DIR) -> set[str]:
    """Return the full set of docnos (used to validate qrels)."""
    return {d["docno"] for d in iter_docs(corpus_dir)}


__all__: Iterable[str] = (
    "Doc",
    "iter_doc_files",
    "iter_docs",
    "count_docs",
    "collect_docnos",
)
