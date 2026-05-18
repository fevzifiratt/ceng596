from __future__ import annotations

import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import pyterrier as pt
import streamlit as st

from ir596.index.build_index import build
from ir596.io.parse_corpus import iter_docs
from ir596.retrieval.bm25 import bm25
from ir596.retrieval.extended_boolean import extended_boolean
from ir596.retrieval.prf import QEModel, rocchio_prf
from ir596.retrieval.tolerant import (
    build_index as build_tolerant_index,
    rewrite_query as tolerant_rewrite_query,
    tolerant_bm25,
)
from ir596.retrieval.word2vec import (
    build_model as build_word2vec_model,
    expand_query as word2vec_expand_query,
    word2vec_bm25,
)
from ir596.retrieval.wordnet import (
    expand_query as wordnet_expand_query,
    wordnet_bm25,
)

TUNED_K1 = 1.8
TUNED_B = 0.5

MODEL_OPTIONS = {
    "BM25 (Tuned)": "bm25",
    "PRF Bo1": "prf_bo1",
    "PRF KL": "prf_kl",
    "PRF RM3": "prf_rm3",
    "Extended Boolean": "extended_boolean",
    "WordNet Expansion": "wordnet",
    "Word2Vec Expansion": "word2vec",
    "Tolerant Retrieval": "tolerant",
}

EXAMPLE_QUERIES = [
    "oil spills",
    "insurance coverage for long term care",
    "right wing christian fundamentalism",
    "accusations of cheating by contractors on defense projects",
    "prison overcrowding",
    "wildcard demo: insur* coverage",
]


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 3]


def _make_snippet(text: str, query: str, *, width: int = 240) -> str:
    source = re.sub(r"\s+", " ", text).strip()
    if not source:
        return ""

    terms = _query_terms(query)
    lower = source.lower()
    start = 0
    for term in terms:
        pos = lower.find(term)
        if pos >= 0:
            start = max(0, pos - width // 3)
            break
    snippet = source[start : start + width].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + width < len(source):
        snippet += "..."

    escaped = html.escape(snippet)
    for term in sorted(set(terms), key=len, reverse=True):
        escaped = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r"<mark>\1</mark>",
            escaped,
        )
    return escaped


@st.cache_resource(show_spinner=False)
def _ensure_pt() -> None:
    if not pt.java.started():
        pt.java.init()


@st.cache_resource(show_spinner=False)
def _stemmed_index_ref() -> pt.IndexRef:
    _ensure_pt()
    return build("stemmed")


@st.cache_resource(show_spinner="Loading AP88 documents for snippets...")
def _doc_store() -> dict[str, dict[str, str]]:
    return {
        doc["docno"]: {"head": doc["head"], "text": doc["text"]}
        for doc in iter_docs()
    }


@st.cache_resource(show_spinner="Loading AP88 Word2Vec model...")
def _word2vec_model():
    return build_word2vec_model()


@st.cache_resource(show_spinner="Loading tolerant retrieval index...")
def _tolerant_index():
    return build_tolerant_index()


def _pipeline_for(
    mode: str,
    *,
    top_k: int,
    prf_docs: int,
    prf_model: QEModel,
    ext_p: float,
):
    index_ref = _stemmed_index_ref()
    if mode == "bm25":
        return bm25(index_ref, k1=TUNED_K1, b=TUNED_B, num_results=top_k)
    if mode == "prf_bo1":
        return rocchio_prf(
            index_ref,
            k1=TUNED_K1,
            b=TUNED_B,
            qe_model="Bo1",
            fb_docs=prf_docs,
            fb_terms=10,
            num_results=top_k,
        )
    if mode == "prf_kl":
        return rocchio_prf(
            index_ref,
            k1=TUNED_K1,
            b=TUNED_B,
            qe_model="KL",
            fb_docs=prf_docs,
            fb_terms=10,
            num_results=top_k,
        )
    if mode == "prf_rm3":
        return rocchio_prf(
            index_ref,
            k1=TUNED_K1,
            b=TUNED_B,
            qe_model="RM3",
            fb_docs=prf_docs,
            fb_terms=10,
            fb_lambda=0.6,
            num_results=top_k,
        )
    if mode == "extended_boolean":
        return extended_boolean(
            remove_stopwords=True,
            mode="or",
            p=ext_p,
            num_results=top_k,
        )
    if mode == "wordnet":
        return wordnet_bm25(
            index_ref,
            k1=TUNED_K1,
            b=TUNED_B,
            num_results=top_k,
        )
    if mode == "word2vec":
        return word2vec_bm25(
            index_ref,
            _word2vec_model(),
            k1=TUNED_K1,
            b=TUNED_B,
            num_results=top_k,
        )
    if mode == "tolerant":
        return tolerant_bm25(
            index_ref,
            _tolerant_index(),
            k1=TUNED_K1,
            b=TUNED_B,
            num_results=top_k,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _rewrite_preview(mode: str, query: str) -> str | None:
    if mode == "wordnet":
        return wordnet_expand_query(query)
    if mode == "word2vec":
        return word2vec_expand_query(_word2vec_model(), query)
    if mode == "tolerant":
        return tolerant_rewrite_query(_tolerant_index(), query)
    return None


def _run_search(
    mode: str,
    query: str,
    *,
    top_k: int,
    prf_docs: int,
    prf_model: QEModel,
    ext_p: float,
) -> pd.DataFrame:
    pipeline = _pipeline_for(
        mode,
        top_k=top_k,
        prf_docs=prf_docs,
        prf_model=prf_model,
        ext_p=ext_p,
    )
    topics = pd.DataFrame([{"qid": "demo", "query": query}])
    results = pipeline.transform(topics)
    if results.empty:
        return results
    return results.sort_values(["rank", "score"], ascending=[True, False]).head(top_k)


st.set_page_config(
    page_title="Total Recall Demo",
    page_icon="IR",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, #f8ead4 0%, transparent 28%),
          radial-gradient(circle at top right, #dcefe6 0%, transparent 25%),
          linear-gradient(180deg, #f7f4ee 0%, #fcfbf8 100%);
      }
      .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(55, 45, 25, 0.12);
        background: rgba(255, 252, 246, 0.82);
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(78, 63, 28, 0.06);
      }
      .doc-card {
        padding: 1rem 1rem 0.8rem;
        border: 1px solid rgba(55, 45, 25, 0.10);
        background: rgba(255, 255, 255, 0.82);
        border-radius: 16px;
        margin-bottom: 0.8rem;
      }
      .doc-head {
        font-size: 1.06rem;
        font-weight: 700;
        color: #17261d;
        margin-bottom: 0.35rem;
      }
      .doc-meta {
        color: #6a6d67;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
      }
      mark {
        background: #ffe18f;
        padding: 0.05rem 0.18rem;
        border-radius: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0 0 0.35rem 0;">Total Recall Demo</h1>
      <div style="font-size:1.02rem; color:#46514a;">
        AP88 üzerinde BM25, PRF, Extended Boolean, WordNet, Word2Vec ve
        tolerant retrieval demoları.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "demo_query" not in st.session_state:
    st.session_state.demo_query = EXAMPLE_QUERIES[0]

with st.sidebar:
    st.header("Demo Controls")
    example = st.selectbox("Example query", EXAMPLE_QUERIES, index=0)
    if st.button("Use example", use_container_width=True):
        st.session_state.demo_query = example

    model_label = st.selectbox("Retrieval mode", list(MODEL_OPTIONS.keys()))
    top_k = st.slider("Top K", min_value=5, max_value=30, value=10, step=5)
    prf_docs = st.selectbox("PRF fb_docs", [3, 5, 10], index=0)
    ext_p = st.select_slider("Extended Boolean p", options=[1.5, 2.0, 3.0], value=1.5)

    st.caption("Tuned BM25 base: k1=1.8, b=0.5")
    st.caption("Wildcard denemek için tolerant mode + örn. `insur* coverage`")

query = st.text_input(
    "Query",
    key="demo_query",
    placeholder="Örn. oil spills",
)

col_a, col_b = st.columns([1, 1])
with col_a:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col_b:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.demo_query = ""
    st.rerun()

if search_clicked:
    if not query.strip():
        st.warning("Önce bir sorgu girmen gerekiyor.")
    else:
        mode = MODEL_OPTIONS[model_label]
        try:
            rewrite = _rewrite_preview(mode, query)
        except Exception as exc:
            rewrite = None
            st.info(f"Rewrite preview alınamadı: {exc}")

        with st.spinner("Searching AP88..."):
            try:
                results = _run_search(
                    mode,
                    query,
                    top_k=top_k,
                    prf_docs=prf_docs,
                    prf_model="Bo1",
                    ext_p=ext_p,
                )
            except Exception as exc:
                st.error(f"Arama çalıştırılamadı: {exc}")
                results = pd.DataFrame()

        if rewrite and rewrite != query:
            st.caption(f"Rewritten query: `{rewrite}`")

        if mode != "tolerant" and "*" in query:
            st.info("Wildcard sorgular pratikte tolerant mode ile daha anlamlı çalışır.")

        if results.empty:
            st.warning("Sonuç bulunamadı.")
        else:
            docs = _doc_store()
            st.success(f"{len(results)} sonuç gösteriliyor.")
            for row in results.itertuples(index=False):
                doc = docs.get(str(row.docno), {"head": "", "text": ""})
                title = doc["head"] or str(row.docno)
                snippet = _make_snippet(f"{doc['head']} {doc['text']}", query)
                st.markdown(
                    f"""
                    <div class="doc-card">
                      <div class="doc-head">{html.escape(title)}</div>
                      <div class="doc-meta">rank #{int(row.rank)} · docno {html.escape(str(row.docno))} · score {float(row.score):.4f}</div>
                      <div>{snippet or "Snippet bulunamadı."}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

