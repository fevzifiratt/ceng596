# Total Recall — Ad-Hoc IR on the Associated Press (AP88) Corpus

CENG 596 course project. Two-stage retrieval pipeline built over the
Associated Press 1988 newswire corpus, evaluated with standard TREC
metrics (MAP, NDCG@10, P@10).

## Layout

```
src/ir596/
  config.py              paths + hyperparameters
  io/                    corpus / topics / qrels parsers
  index/                 PyTerrier indexing
  retrieval/             BM25, PRF, WordNet, Word2Vec, tolerant, ext-boolean
  app/                   Streamlit demo UI
data/
  raw/coll               AP collection (gitignored)
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

## Step 2 — BM25 baseline

```bash
.venv/bin/pip install -e .
.venv/bin/python scripts/step2_bm25_baseline.py
```

Builds two Terrier indexes (stemmed + unstemmed) and runs BM25 with
two query fields (title, title+desc) → 4 baseline configs.
Produces `index/{stemmed,unstemmed}/`, `runs/bm25_*.trec`, and appends
rows to `evals/summary.csv`. First run ~2 min, re-runs ~30 s.

Locked baseline: `bm25_stemmed_title_desc` → MAP 0.3344, NDCG@10 0.5028.

## Step 3 — BM25 tuning (k1, b grid search)

```bash
.venv/bin/python scripts/step3_tune_bm25.py
```

Sweeps a 6×4 grid (`k1 ∈ {0.6..2.1}`, `b ∈ {0.25..1.0}`) on the locked
baseline config (stemmed index, title+desc queries) using a single
`pt.Experiment`. Produces `evals/bm25_grid.csv` (full grid),
`runs/bm25_tuned_title_desc.trec` (winner re-run), and appends a
`bm25_tuned` row to `evals/summary.csv`. Runtime ~2 min.

Tuned baseline: `k1=1.80, b=0.50` → MAP 0.3396, NDCG@10 0.5059
(+1.6% MAP over Terrier defaults). This is the reference Stage-2
techniques have to beat.

## Step 4 — Extended Boolean retrieval

```bash
.venv/bin/python scripts/step4_extended_boolean.py
```

Builds a repo-local postings cache from the AP corpus and evaluates a
small p-norm grid over both query fields (`title`, `title_desc`) using a
soft-OR Extended Boolean model. Produces
`index/ext_boolean_nostop.pkl.gz`, `runs/extbool_*.trec`, and appends
rows to `evals/summary.csv`.

## Step 5 — Pseudo-relevance feedback (Rocchio)

```bash
.venv/bin/python scripts/step5_rocchio_prf.py
```

Layers Rocchio-style PRF on top of the Step-3 tuned BM25 baseline
(`k1=1.80, b=0.50`). Pipeline per config:
`BM25 >> QE >> BM25`, where the QE rewriter takes the top `fb_docs`
results from the first BM25 as pseudo-relevant and rewrites the
query for the final retrieval. Sweeps three Rocchio-family expansion
models — `Bo1`, `KL`, `RM3` — at `fb_docs ∈ {3, 5, 10}` with
`fb_terms=10`, on `title_desc`, plus a `title`-only control matching
the proposal's recipe (Bo1, top-5). Produces `runs/prf_*.trec`,
`evals/prf_grid.csv`, and appends rows to `evals/summary.csv`.
