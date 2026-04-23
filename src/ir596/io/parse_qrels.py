"""Parse ``qrels1-50ap.docx`` into a plain ``qrels.trec`` file for trec_eval.

Each non-empty paragraph is already in the correct form::

    <topic> 0 <docno> <rel>

We just strip whitespace, validate the 4-column shape, and write it out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

from docx import Document

from ir596.config import QRELS_DOCX, QRELS_TREC


@dataclass(frozen=True)
class QrelLine:
    qid: str
    docno: str
    rel: int


def _iter_paragraph_text(path: Path) -> Iterator[str]:
    doc = Document(str(path))
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            yield text


def parse_qrels(path: Path = QRELS_DOCX) -> List[QrelLine]:
    out: List[QrelLine] = []
    for lineno, line in enumerate(_iter_paragraph_text(path), start=1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"qrels line {lineno} malformed: {line!r}")
        qid, dummy, docno, rel = parts
        if dummy != "0":
            raise ValueError(
                f"qrels line {lineno} dummy column expected '0', got {dummy!r}"
            )
        if rel not in ("0", "1"):
            raise ValueError(
                f"qrels line {lineno} relevance must be 0 or 1, got {rel!r}"
            )
        out.append(QrelLine(qid=qid, docno=docno, rel=int(rel)))
    return out


def write_trec_qrels(qrels: Iterable[QrelLine], out: Path = QRELS_TREC) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for q in qrels:
            f.write(f"{q.qid} 0 {q.docno} {q.rel}\n")
    return out


__all__ = ("QrelLine", "parse_qrels", "write_trec_qrels")
