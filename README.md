# Total Recall — Ad-Hoc IR on the Associated Press (AP88) Corpus

CENG 596 course project. Two-stage retrieval pipeline built over the
Associated Press 1988 newswire corpus, evaluated with standard TREC
metrics (MAP, NDCG@10, P@10).

## Layout

```
src/ir596/
  config.py              paths + hyperparameters
  io/                    corpus / topics / qrels parsers
  index/                 PyTerrier indexing (step 2+)
  retrieval/             BM25, PRF, WordNet, Word2Vec, tolerant, ext-boolean
  app/                   Streamlit demo UI
data/
  raw/coll               symlink to the AP collection (outside the repo)
  topics/topics.trec     parsed topics (produced by scripts/step1_sanity_check.py)
  qrels/qrels.trec       parsed qrels  (produced by scripts/step1_sanity_check.py)
index/                   Terrier index (gitignored)
runs/                    per-config .trec run files
evals/                   trec_eval summaries
scripts/                 one-shot entry points, one per milestone step
reports/                 progress + final report
notebooks/               EDA + ablation plots
```

## Data location

The corpus lives outside the repo. Default path:

```
~/Downloads/AssociatedPressDataset-20260423T115119Z-3-001/AssociatedPressDataset/
```

Override with the env var `AP_DATA_ROOT` if you keep it elsewhere.

## Step 1 — data prep (run once)

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/step1_sanity_check.py
```

Produces `data/topics/topics.trec`, `data/qrels/qrels.trec`, and prints
corpus stats. Expected: 79923 docs, 50 topics, qrels all valid.
