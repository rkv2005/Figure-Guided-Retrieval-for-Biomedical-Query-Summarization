import bm25s
import json
import pickle
import gc
import time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

STORE_ROOT = Path("/content/drive/MyDrive/ffhrag_store")
EMB_ROOT   = STORE_ROOT / "embeddings"
BM25_ROOT  = STORE_ROOT / "BM25"
BM25_ROOT.mkdir(exist_ok=True)

SEPARATOR = "=" * 70


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_pkl(obj, path, label):
    print(f"  Saving {label}...", end=" ", flush=True)
    t0 = time.time()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"done ({time.time()-t0:.1f}s)  {path.stat().st_size/1e6:.1f} MB")


ft_valid_idx  = np.load(EMB_ROOT / "fulltext_valid_idx.npy")
fig_bert_idx  = np.load(EMB_ROOT / "fig_bert_valid_idx.npy")
ft_valid_set  = set(ft_valid_idx.tolist())
fig_valid_set = set(fig_bert_idx.tolist())
del ft_valid_idx, fig_bert_idx; gc.collect()

text_files = [
    (EMB_ROOT / "emb_chunks_meta.jsonl",                            "text", "chunk_id", ft_valid_set),
    (EMB_ROOT / "emb_pmcid_abstract_chunks_spubmedbert_meta.jsonl", "text", "chunk_id", None),
    (EMB_ROOT / "emb_pmid_abstract_chunks_spubmedbert_meta.jsonl",  "text", "chunk_id", None),
]


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 1 — BM25S TEXT INDEX")
print(SEPARATOR)

t0 = time.time()

# ── Stream texts + IDs ────────────────────────────────────────────────────────
print("\n  Streaming texts...")
all_texts     = []
bm25_text_ids = []
count = 0

for path, text_key, id_key, valid_set in text_files:
    for i, m in enumerate(stream_jsonl(path)):
        if valid_set is not None and i not in valid_set:
            continue
        all_texts.append(m.get(text_key, ""))
        cid = m.get(id_key) or f"{m.get('pmid', m.get('pmcid','?'))}__ABS__c{i:04d}"
        bm25_text_ids.append(cid)
        count += 1
        if count % 200000 == 0:
            print(f"    {count:>8} streamed...")

print(f"  Total: {len(all_texts)} docs  |  {len(bm25_text_ids)} IDs")

# ── Tokenize ──────────────────────────────────────────────────────────────────
print("\n  Tokenizing...", end=" ", flush=True)
t1 = time.time()
corpus_tokens = bm25s.tokenize(all_texts, stopwords="en", show_progress=False)
print(f"done ({time.time()-t1:.1f}s)")

# ── CRITICAL: free raw strings before index() ─────────────────────────────────
del all_texts; gc.collect()
print("  Raw strings freed ✅")

# ── Index ─────────────────────────────────────────────────────────────────────
print("  Indexing...", end=" ", flush=True)
t1 = time.time()
retriever = bm25s.BM25(method="bm25+")
retriever.index(corpus_tokens)
print(f"done ({time.time()-t1:.1f}s)")

# ── Free tokens immediately ───────────────────────────────────────────────────
del corpus_tokens; gc.collect()
print("  Corpus tokens freed ✅")

# ── Sanity check ──────────────────────────────────────────────────────────────
test_q          = bm25s.tokenize(["CRISPR gene editing cancer"], stopwords="en",
                                  show_progress=False)
results, scores = retriever.retrieve(test_q, k=3)
print(f"  Sanity — top score: {scores[0][0]:.4f}  top ID: {bm25_text_ids[results[0][0]]}")

# ── Save ──────────────────────────────────────────────────────────────────────
bm25s_text_path = BM25_ROOT / "bm25s_text"
bm25s_text_path.mkdir(exist_ok=True)
retriever.save(str(bm25s_text_path))
print(f"  Saved → bm25s_text/")
save_pkl(bm25_text_ids, BM25_ROOT / "bm25_text_ids.pkl", "bm25_text_ids.pkl")
del retriever, bm25_text_ids; gc.collect()
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 2 — BM25S FIGURES INDEX")
print(SEPARATOR)

t0 = time.time()

fig_texts    = []
bm25_fig_ids = []

for i, m in enumerate(stream_jsonl(EMB_ROOT / "emb_figcaps_meta.jsonl")):
    if i not in fig_valid_set:
        continue
    fig_texts.append(m.get("caption_text", ""))
    bm25_fig_ids.append(f"{m['pmcid']}_{m['fig_id']}")

print(f"  Corpus size: {len(fig_texts)}")

corpus_tokens = bm25s.tokenize(fig_texts, stopwords="en", show_progress=False)
del fig_texts; gc.collect()

fig_retriever = bm25s.BM25(method="bm25+")
fig_retriever.index(corpus_tokens)
del corpus_tokens; gc.collect()

test_q          = bm25s.tokenize(["survival curve patients hazard"], stopwords="en",
                                   show_progress=False)
results, scores = fig_retriever.retrieve(test_q, k=3)
print(f"  Sanity — top score: {scores[0][0]:.4f}  top ID: {bm25_fig_ids[results[0][0]]}")

bm25s_fig_path = BM25_ROOT / "bm25s_figures"
bm25s_fig_path.mkdir(exist_ok=True)
fig_retriever.save(str(bm25s_fig_path))
save_pkl(bm25_fig_ids, BM25_ROOT / "bm25_figures_ids.pkl", "bm25_figures_ids.pkl")
del fig_retriever, bm25_fig_ids, fig_valid_set; gc.collect()
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  STEP 3 — TF-IDF TEXT INDEX")
print(SEPARATOR)

t0 = time.time()

print("\n  Streaming texts for TF-IDF...")
tfidf_texts = []
tfidf_ids   = []
count = 0

for path, text_key, id_key, valid_set in text_files:
    for i, m in enumerate(stream_jsonl(path)):
        if valid_set is not None and i not in valid_set:
            continue
        tfidf_texts.append(m.get(text_key, ""))
        cid = m.get(id_key) or f"{m.get('pmid', m.get('pmcid','?'))}__ABS__c{i:04d}"
        tfidf_ids.append(cid)
        count += 1
        if count % 200000 == 0:
            print(f"    {count:>8} streamed...")

print(f"  Total: {len(tfidf_texts)}")

print("\n  Fitting TfidfVectorizer...", end=" ", flush=True)
t1 = time.time()
tfidf_model = TfidfVectorizer(
    max_features=100000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    ngram_range=(1, 1),
)
tfidf_matrix = tfidf_model.fit_transform(tfidf_texts)
print(f"done ({time.time()-t1:.1f}s)")
del tfidf_texts; gc.collect()

if tfidf_matrix.format != "csr":
    tfidf_matrix = tfidf_matrix.tocsr()

print(f"  Matrix shape : {tfidf_matrix.shape}")
print(f"  NNZ          : {tfidf_matrix.nnz:,}")
print(f"  Vocab size   : {len(tfidf_model.vocabulary_)}")

save_pkl(tfidf_model,  BM25_ROOT / "tfidf_text_model.pkl",  "tfidf_text_model.pkl")
save_pkl(tfidf_ids,    BM25_ROOT / "tfidf_text_ids.pkl",    "tfidf_text_ids.pkl")
save_pkl(tfidf_matrix, BM25_ROOT / "tfidf_text_matrix.pkl", "tfidf_text_matrix.pkl")
del tfidf_model, tfidf_matrix, tfidf_ids; gc.collect()
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  BUILD COMPLETE — FINAL SUMMARY")
print(SEPARATOR)

for p in sorted(BM25_ROOT.glob("**/*")):
    if p.is_file():
        print(f"  ✅ {str(p.relative_to(BM25_ROOT)):<48} {p.stat().st_size/1e6:.1f} MB")

print(f"\n{SEPARATOR}")
print("  ✅  BM25S + TF-IDF BUILD COMPLETE")
print(SEPARATOR)
