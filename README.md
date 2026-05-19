# Total Recall — Ad-Hoc IR on the Associated Press (AP88) Corpus

CENG 596 course project. Two-stage retrieval pipeline built over the
Associated Press 1988 newswire corpus, evaluated with standard TREC
metrics (MAP, NDCG@10, P@10).

## Setup

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Required before running the demo UI or any of the step scripts below.

## Demo UI

```bash
.venv/bin/python -m streamlit run app.py
```

Opens a local query UI for trying the project pipelines interactively:
tuned BM25, PRF, Extended Boolean, WordNet, Word2Vec, and tolerant
retrieval with snippets.

Only requires the AP corpus at `data/raw/coll/`. On first launch, the
Streamlit app will lazily build whatever caches the selected mode
needs — Terrier index (~2 min), Word2Vec model, tolerant vocabulary —
and reuse them on subsequent runs. The WordNet mode additionally
requires `nltk_data/` (see Step 6 for the one-time NLTK download).

## Layout

```
app.py                   Streamlit demo UI
src/ir596/
  config.py              paths + hyperparameters
  io/                    corpus / topics / qrels parsers
  index/                 PyTerrier indexing
  retrieval/             BM25, PRF, WordNet, Word2Vec, tolerant, ext-boolean
data/
  raw/coll               AP collection (gitignored)
  topics/topics.trec     parsed topics (produced by scripts/step1_sanity_check.py)
  qrels/qrels.trec       parsed qrels  (produced by scripts/step1_sanity_check.py)
index/                   Terrier index (gitignored)
runs/                    per-config .trec run files
evals/                   trec_eval summaries
scripts/                 one-shot entry points, one per milestone step
```

## Data location

The corpus lives in `data/raw/` (gitignored), with this layout:

```
data/raw/
  coll/                  AP collection files
  topics1-50.docx        topics
  qrels1-50ap.docx       qrels
```

## Step 1 — data prep (run once)

```bash
.venv/bin/python scripts/step1_sanity_check.py
```

Produces `data/topics/topics.trec`, `data/qrels/qrels.trec`, and prints
corpus stats. Expected: 79923 docs, 50 topics, qrels all valid.

## Step 2 — BM25 baseline

```bash
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

## Step 6 — WordNet Synonym Expansion

```bash
.venv/bin/python -c "import nltk; nltk.download('wordnet', download_dir='nltk_data'); nltk.download('omw-1.4', download_dir='nltk_data')"
.venv/bin/python scripts/step6_wordnet_vs_prf.py
```

Applies conservative WordNet synonym expansion on top of the tuned BM25
baseline (`k1=1.80`, `b=0.50`) and reports MAP deltas against both tuned
BM25 and the best Step-5 PRF baseline. Produces
`evals/wordnet_vs_prf.csv` and `runs/wordnet_tuned_title_desc.trec`.

## Step 7 — Word2Vec AP88 Expansion

```bash
.venv/bin/python scripts/step7_word2vec.py
```

Trains or reuses an AP88-specific Word2Vec model and applies
conservative nearest-neighbour query expansion on top of tuned BM25.
Produces `evals/word2vec_ap88.csv` and
`runs/word2vec_tuned_title_desc.trec`.

## Step 8 — Tolerant Retrieval

```bash
.venv/bin/python scripts/step8_tolerant_retrieval.py
```

Builds a tolerant vocabulary cache with BK-tree spelling lookup plus
permuterm wildcard expansion, then evaluates that rewrite independently
on top of tuned BM25. Produces `evals/tolerant_retrieval.csv` and
`runs/tolerant_tuned_title_desc.trec`.

## Results

Headline numbers on AP88 (50 topics, full qrels). Full per-config table
lives in `evals/summary.csv`.

| Config                                     | MAP    | NDCG@10 | Δ MAP vs. tuned BM25 |
| ------------------------------------------ | ------ | ------- | -------------------- |
| BM25 baseline (stemmed, title+desc)        | 0.3344 | 0.5028  | —                    |
| **BM25 tuned** (`k1=1.80`, `b=0.50`)       | 0.3396 | 0.5059  | reference            |
| Extended Boolean (best, `p=1.5`, t+d)      | 0.2290 | 0.3448  | −32.6%               |
| WordNet expansion                          | 0.3360 | 0.4987  | −1.1%                |
| Word2Vec AP88 expansion                    | 0.3320 | 0.4958  | −2.2%                |
| Tolerant retrieval (no triggered rewrites) | 0.3396 | 0.5059  | ±0.0%                |
| **PRF Bo1, `fb_docs=3`, t+d (winner)**     | 0.3737 | 0.5216  | **+10.0%**           |
| PRF KL, `fb_docs=3`, t+d                   | 0.3711 | 0.5134  | +9.3%                |
| PRF Bo1, `fb_docs=5`, title-only          | 0.3706 | 0.5261  | +9.1%                |

Stage-2 takeaway: classical Rocchio-style PRF (`Bo1` / `KL`) is the only
expansion technique that meaningfully beats the tuned BM25 baseline on
this corpus. WordNet and Word2Vec expansions stay close to baseline,
Extended Boolean underperforms, and tolerant retrieval is neutral on
clean topic strings (it's a recall-side safety net for typos and
wildcards, not a precision booster).
