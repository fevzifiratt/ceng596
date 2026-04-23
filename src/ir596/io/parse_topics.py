"""Parse the TREC topics from ``topics1-50.docx`` into a proper TREC topics file.

The source .docx contains 50 topic blocks in a TREC-like, but malformed,
SGML (open tags without closers, each block terminated by ``</top>``)::

    <top>
    <num>1
    <title>Coping with overcrowded prisons
    <desc>
    The document will ...
    <narr>
    A relevant document will ...
    </top>

We rebuild a well-formed TREC topics file::

    <top>
    <num> Number: 1 </num>
    <title> Coping with overcrowded prisons </title>
    <desc> Description: ... </desc>
    <narr> Narrative: ... </narr>
    </top>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from docx import Document

from ir596.config import TOPICS_DOCX, TOPICS_TREC

_NUM_RE = re.compile(r"<num>\s*(\d+)", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*(?=<desc>|<narr>|</top>|$)",
                       re.IGNORECASE | re.DOTALL)
_DESC_RE = re.compile(r"<desc>\s*(.*?)\s*(?=<narr>|</top>|$)",
                      re.IGNORECASE | re.DOTALL)
_NARR_RE = re.compile(r"<narr>\s*(.*?)\s*(?=</top>|$)",
                      re.IGNORECASE | re.DOTALL)

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Topic:
    qid: str
    title: str
    desc: str
    narr: str


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _read_docx_text(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def parse_topics(path: Path = TOPICS_DOCX) -> List[Topic]:
    raw = _read_docx_text(path)
    blocks = [b for b in raw.split("</top>") if "<num>" in b.lower()]
    topics: List[Topic] = []
    for block in blocks:
        num_m = _NUM_RE.search(block)
        title_m = _TITLE_RE.search(block)
        desc_m = _DESC_RE.search(block)
        narr_m = _NARR_RE.search(block)
        if not (num_m and title_m):
            continue
        topics.append(
            Topic(
                qid=num_m.group(1).strip(),
                title=_clean(title_m.group(1)),
                desc=_clean(desc_m.group(1)) if desc_m else "",
                narr=_clean(narr_m.group(1)) if narr_m else "",
            )
        )
    return topics


def write_trec_topics(topics: Iterable[Topic], out: Path = TOPICS_TREC) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in topics:
            f.write("<top>\n")
            f.write(f"<num> Number: {t.qid} </num>\n")
            f.write(f"<title> {t.title} </title>\n")
            f.write(f"<desc> Description: {t.desc} </desc>\n")
            f.write(f"<narr> Narrative: {t.narr} </narr>\n")
            f.write("</top>\n\n")
    return out


__all__ = ("Topic", "parse_topics", "write_trec_topics")
