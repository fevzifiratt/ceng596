"""Build query DataFrames from the parsed topics.

Two query "fields" are exposed:

- ``title``        : just the ``<title>`` text (very short, ~3–6 words)
- ``title_desc``   : ``<title>`` + ``<desc>``  (5–60 words; richer signal,
                     more vocabulary mismatch)

We sanitise queries for Terrier's parser: lowercase, keep only
alphanumerics + spaces.  This avoids the ``(``, ``)``, ``:``, ``"`` etc
that Terrier treats as query operators.
"""
from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from ir596.io.parse_topics import Topic, parse_topics

QueryField = Literal["title", "title_desc"]

_NON_QUERY_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def sanitise(text: str) -> str:
    text = text.lower()
    text = _NON_QUERY_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _query_text(t: Topic, field: QueryField) -> str:
    if field == "title":
        return sanitise(t.title)
    if field == "title_desc":
        return sanitise(f"{t.title} {t.desc}")
    raise ValueError(f"unknown query field: {field}")


def topics_df(field: QueryField = "title") -> pd.DataFrame:
    """Return a DataFrame with columns ``qid`` and ``query``."""
    topics = parse_topics()
    rows = [{"qid": t.qid, "query": _query_text(t, field)} for t in topics]
    df = pd.DataFrame(rows)
    return df[df["query"].str.len() > 0].reset_index(drop=True)


__all__ = ("QueryField", "sanitise", "topics_df")
