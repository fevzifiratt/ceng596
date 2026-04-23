"""Step 1 — data prep + sanity checks.

Produces::

    data/topics/topics.trec
    data/qrels/qrels.trec

and prints corpus / topic / qrel statistics.  Expected numbers::

    79923 documents
    50    topics
    all qrel docnos present in the corpus

Run from the repo root::

    .venv/bin/python scripts/step1_sanity_check.py
"""
from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ir596 import config                                    # noqa: E402
from ir596.io.parse_corpus import iter_docs                 # noqa: E402
from ir596.io.parse_qrels import parse_qrels, write_trec_qrels     # noqa: E402
from ir596.io.parse_topics import parse_topics, write_trec_topics  # noqa: E402


def _hr(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def check_corpus() -> set[str]:
    _hr("Corpus")
    t0 = time.perf_counter()
    docnos: set[str] = set()
    total_chars = 0
    first_docno = last_docno = ""
    for i, d in enumerate(iter_docs()):
        if i == 0:
            first_docno = d["docno"]
        last_docno = d["docno"]
        docnos.add(d["docno"])
        total_chars += len(d["text"])
    elapsed = time.perf_counter() - t0
    n = len(docnos)
    print(f"docs parsed          : {n}")
    print(f"first docno          : {first_docno}")
    print(f"last  docno          : {last_docno}")
    print(f"avg text length      : {total_chars // max(n, 1)} chars")
    print(f"parse time           : {elapsed:.1f}s")
    if n != config.EXPECTED_NUM_DOCS:
        print(f"WARNING: expected {config.EXPECTED_NUM_DOCS} docs, got {n}")
    else:
        print(f"OK: matches expected {config.EXPECTED_NUM_DOCS}")
    return docnos


def check_topics() -> None:
    _hr("Topics")
    topics = parse_topics()
    print(f"topics parsed        : {len(topics)}")
    if topics:
        t0 = topics[0]
        print(f"first topic qid      : {t0.qid}")
        print(f"first topic title    : {t0.title}")
    if len(topics) != config.EXPECTED_NUM_TOPICS:
        print(
            f"WARNING: expected {config.EXPECTED_NUM_TOPICS} topics, got {len(topics)}"
        )
    else:
        print(f"OK: matches expected {config.EXPECTED_NUM_TOPICS}")
    out = write_trec_topics(topics)
    print(f"wrote                : {out.relative_to(REPO_ROOT)}")


def check_qrels(corpus_docnos: set[str]) -> None:
    _hr("Qrels")
    qrels = parse_qrels()
    topics_in_qrels = {q.qid for q in qrels}
    rel_counter: Counter[int] = Counter(q.rel for q in qrels)
    missing = [q for q in qrels if q.docno not in corpus_docnos]

    print(f"qrel lines           : {len(qrels)}")
    print(f"distinct topics      : {len(topics_in_qrels)}")
    print(f"  relevance=1        : {rel_counter.get(1, 0)}")
    print(f"  relevance=0        : {rel_counter.get(0, 0)}")
    print(f"docnos missing corpus: {len(missing)}")
    if missing:
        sample = random.sample(missing, k=min(5, len(missing)))
        for q in sample:
            print(f"  MISSING  {q.qid} 0 {q.docno} {q.rel}")

    out = write_trec_qrels(qrels)
    print(f"wrote                : {out.relative_to(REPO_ROOT)}")


def main() -> int:
    config.ensure_dirs()
    print(f"repo                 : {REPO_ROOT}")
    print(f"corpus dir           : {config.CORPUS_DIR}")
    print(f"topics .docx         : {config.TOPICS_DOCX}")
    print(f"qrels  .docx         : {config.QRELS_DOCX}")

    docnos = check_corpus()
    check_topics()
    check_qrels(docnos)

    _hr("Done")
    print("Step 1 complete. You can now proceed to Step 2 (BM25 baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
