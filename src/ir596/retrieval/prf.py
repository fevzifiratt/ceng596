"""Pseudo-relevance feedback (Rocchio-family) pipelines.

Implements the Stage-2 PRF step from the project proposal: top-k BM25
results are treated as pseudo-relevant, used to re-weight / expand the
query vector, and a second BM25 retrieval is run with the expanded
query.

PyTerrier ships three Rocchio-family query expansion models out of the
box, all of which follow the same recipe at a high level (score
candidate expansion terms from feedback docs, append the top
``fb_terms``, retrieve again):

- ``Bo1``  : Bose-Einstein DFR weighting (Amati 2003) — Terrier's
             canonical "Rocchio-style" expansion
- ``KL``   : Kullback-Leibler divergence weighting
- ``RM3``  : relevance-model variant; mixes the original query with
             the expansion via ``fb_lambda``

The proposal targets the short TREC topics with top-5 feedback, so
``fb_docs`` defaults to 5. The pipeline returned is a plain
``pt.Transformer`` and plugs into ``ir596.evaluate.evaluate`` exactly
like ``bm25()`` does.
"""
from __future__ import annotations

from typing import Literal

import pyterrier as pt

from ir596.retrieval.bm25 import bm25

QEModel = Literal["Bo1", "KL", "RM3"]


def _qe_transformer(
    index_ref: pt.IndexRef,
    qe_model: QEModel,
    *,
    fb_docs: int,
    fb_terms: int,
    fb_lambda: float,
) -> pt.Transformer:
    if qe_model == "Bo1":
        return pt.rewrite.Bo1QueryExpansion(
            index_ref, fb_terms=fb_terms, fb_docs=fb_docs
        )
    if qe_model == "KL":
        return pt.rewrite.KLQueryExpansion(
            index_ref, fb_terms=fb_terms, fb_docs=fb_docs
        )
    if qe_model == "RM3":
        return pt.rewrite.RM3(
            index_ref,
            fb_terms=fb_terms,
            fb_docs=fb_docs,
            fb_lambda=fb_lambda,
        )
    raise ValueError(f"unknown QE model: {qe_model!r}")


def rocchio_prf(
    index_ref: pt.IndexRef,
    *,
    k1: float | None = None,
    b: float | None = None,
    qe_model: QEModel = "Bo1",
    fb_docs: int = 5,
    fb_terms: int = 10,
    fb_lambda: float = 0.6,
    num_results: int = 1000,
) -> pt.Transformer:
    """Build a ``BM25 >> QE >> BM25`` PRF pipeline.

    The first BM25 retrieves a working ranking, the QE rewriter takes
    the top ``fb_docs`` documents as pseudo-relevant and rewrites the
    query, and the final BM25 re-runs against that expanded query.
    Pass the tuned ``(k1, b)`` from Step 3 so the BM25 stages match the
    locked baseline.

    Parameters
    ----------
    qe_model :
        Which Rocchio-family expansion model to use. ``Bo1`` is the
        canonical Terrier "Rocchio-style" choice (DFR weighting, no
        extra knobs); ``RM3`` is the relevance-model variant and
        exposes ``fb_lambda``.
    fb_docs :
        Number of pseudo-relevant documents to feed back. The proposal
        calls for top-5 on these short TREC topics.
    fb_terms :
        Number of expansion terms to inject.
    fb_lambda :
        RM3-only mixture weight between the original query and the
        expansion (ignored for Bo1 / KL).
    """
    base = bm25(index_ref, k1=k1, b=b, num_results=num_results)
    qe = _qe_transformer(
        index_ref,
        qe_model,
        fb_docs=fb_docs,
        fb_terms=fb_terms,
        fb_lambda=fb_lambda,
    )
    final = bm25(index_ref, k1=k1, b=b, num_results=num_results)
    return base >> qe >> final


__all__ = ("QEModel", "rocchio_prf")
