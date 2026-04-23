"""Central configuration: all paths and global hyperparameters live here.

By default the dataset is read from the repo-local ``data/raw/`` folder
(where the corpus, topics, and qrels were copied during Step 1).
Override with the ``AP_DATA_ROOT`` environment variable if you keep the
dataset elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(
    os.environ.get("AP_DATA_ROOT", str(REPO_ROOT / "data" / "raw"))
)

CORPUS_DIR: Path = DATA_ROOT / "coll"
TOPICS_DOCX: Path = DATA_ROOT / "topics1-50.docx"
QRELS_DOCX: Path = DATA_ROOT / "qrels1-50ap.docx"

TOPICS_TREC: Path = REPO_ROOT / "data" / "topics" / "topics.trec"
QRELS_TREC: Path = REPO_ROOT / "data" / "qrels" / "qrels.trec"

INDEX_DIR: Path = REPO_ROOT / "index"
RUNS_DIR: Path = REPO_ROOT / "runs"
EVALS_DIR: Path = REPO_ROOT / "evals"

EXPECTED_NUM_DOCS: int = 79923
EXPECTED_NUM_TOPICS: int = 50


def ensure_dirs() -> None:
    for p in (TOPICS_TREC.parent, QRELS_TREC.parent, INDEX_DIR, RUNS_DIR, EVALS_DIR):
        p.mkdir(parents=True, exist_ok=True)
