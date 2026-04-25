"""BM25 retrieval pipeline factory."""
from __future__ import annotations

from typing import Optional

import pyterrier as pt


def bm25(
    index_ref: pt.IndexRef,
    *,
    k1: Optional[float] = None,
    b: Optional[float] = None,
    num_results: int = 1000,
) -> pt.Transformer:
    """Return a BM25 retriever over ``index_ref``.

    Parameters ``k1`` and ``b`` are passed through to Terrier when
    provided; leaving them as ``None`` uses Terrier's defaults
    (k1=1.2, b=0.75) — the proper baseline.
    """
    controls = {}
    if k1 is not None:
        controls["bm25.k_1"] = str(k1)
    if b is not None:
        controls["bm25.b"] = str(b)

    return pt.terrier.Retriever(
        index_ref,
        wmodel="BM25",
        num_results=num_results,
        controls=controls or None,
    )


__all__ = ("bm25",)
