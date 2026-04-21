import Embeddings.faiss as faiss
import numpy as np
import json
import gc
from pathlib import Path
import time

STORE_ROOT = Path("/content/drive/MyDrive/ffhrag_store")
EMB_ROOT   = STORE_ROOT / "embeddings"
FAISS_ROOT = STORE_ROOT / "FAISS"
FAISS_ROOT.mkdir(exist_ok=True)

SEPARATOR = "=" * 70


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_faiss_index(vecs, nlist=None):
    n, dim = vecs.shape

    if n < 10000:
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        return index, "Flat"

    nlist = nlist or min(int(np.sqrt(n)), 4096)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"    Training IVFFlat (nlist={nlist}, n={n})...", end=" ", flush=True)
    t0 = time.time()

    if faiss.get_num_gpus() > 0:
        res   = faiss.StandardGpuResources()
        g_idx = faiss.index_cpu_to_gpu(res, 0, index)
        g_idx.train(vecs)
        index = faiss.index_gpu_to_cpu(g_idx)
    else:
        index.train(vecs)

    print(f"done ({time.time()-t0:.1f}s)")
    index.add(vecs)
    index.nprobe = 64
    return index, f"IVFFlat(nlist={nlist}, nprobe=64)"


def verify_index(index, vecs_sample, top_k=5):
    q = vecs_sample[:1]
    D, I = index.search(q, top_k)
    top_score = D[0][0]
    top_idx   = I[0][0]
    self_hit  = top_idx == 0
    print(f"    Verify: top_idx={top_idx}  score={top_score:.6f}  "
          f"self-hit={'✅' if self_hit else '⚠️  not self (normal for IVF)'}")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  LOADING VALID MASKS")
print(SEPARATOR)

ft_valid_idx     = np.load(EMB_ROOT / "fulltext_valid_idx.npy")
fig_bert_idx     = np.load(EMB_ROOT / "fig_bert_valid_idx.npy")
fig_clip_cap_idx = np.load(EMB_ROOT / "fig_clip_cap_valid_idx.npy")
fig_clip_img_idx = np.load(EMB_ROOT / "fig_clip_img_valid_idx.npy")

print(f"  fulltext valid    : {len(ft_valid_idx):>8} rows")
print(f"  fig BERT cap valid: {len(fig_bert_idx):>8} rows")
print(f"  fig CLIP cap valid: {len(fig_clip_cap_idx):>8} rows")
print(f"  fig CLIP img valid: {len(fig_clip_img_idx):>8} rows")

index_mapping = {}


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  INDEX 1 — TEXT CHUNKS (S-PubMedBERT, 768-dim)")
print("  Sources: full-text + PMCID abstracts + PMID abstracts")
print(SEPARATOR)

t0  = time.time()
DIM = 768

# Count rows without loading data
print("\n  Pre-computing total row count...")
n_ft  = len(ft_valid_idx)
n_pca = np.load(EMB_ROOT / "emb_pmcid_abstract_chunks_spubmedbert.npy",
                mmap_mode='r').shape[0]
n_pma = np.load(EMB_ROOT / "emb_pmid_abstract_chunks_spubmedbert.npy",
                mmap_mode='r').shape[0]
n_total = n_ft + n_pca + n_pma
print(f"  Full-text (filtered): {n_ft}")
print(f"  PMCID abstracts     : {n_pca}")
print(f"  PMID abstracts      : {n_pma}")
print(f"  Total               : {n_total}")

# Pre-allocate single contiguous array — no copies
print(f"\n  Pre-allocating ({n_total} × {DIM}) float32...", end=" ", flush=True)
all_text_vecs = np.empty((n_total, DIM), dtype=np.float32)
print(f"done  ({all_text_vecs.nbytes/1e9:.2f} GB)")

# Fill slice by slice via mmap — never more than one source in RAM
print("  Writing full-text vecs (mmap)...", end=" ", flush=True)
ft_mmap = np.load(EMB_ROOT / "emb_chunks_spubmedbert.npy", mmap_mode='r')
all_text_vecs[:n_ft] = ft_mmap[ft_valid_idx]
del ft_mmap; gc.collect()
print(f"done  (rows 0 → {n_ft})")

print("  Writing PMCID abstract vecs (mmap)...", end=" ", flush=True)
pca_mmap = np.load(EMB_ROOT / "emb_pmcid_abstract_chunks_spubmedbert.npy", mmap_mode='r')
all_text_vecs[n_ft:n_ft+n_pca] = pca_mmap[:]
del pca_mmap; gc.collect()
print(f"done  (rows {n_ft} → {n_ft+n_pca})")

print("  Writing PMID abstract vecs (mmap)...", end=" ", flush=True)
pma_mmap = np.load(EMB_ROOT / "emb_pmid_abstract_chunks_spubmedbert.npy", mmap_mode='r')
all_text_vecs[n_ft+n_pca:n_total] = pma_mmap[:]
del pma_mmap; gc.collect()
print(f"done  (rows {n_ft+n_pca} → {n_total})")

# Load metadata (no vecs)
print("\n  Loading metadata...")
ft_meta       = load_jsonl(EMB_ROOT / "emb_chunks_meta.jsonl")
ft_meta_clean = [ft_meta[i] for i in ft_valid_idx]
del ft_meta; gc.collect()
pca_meta      = load_jsonl(EMB_ROOT / "emb_pmcid_abstract_chunks_spubmedbert_meta.jsonl")
pma_meta      = load_jsonl(EMB_ROOT / "emb_pmid_abstract_chunks_spubmedbert_meta.jsonl")
all_text_meta = ft_meta_clean + pca_meta + pma_meta
del ft_meta_clean, pca_meta, pma_meta; gc.collect()
print(f"  Metadata records: {len(all_text_meta)}")

# Norm check
norms = np.linalg.norm(all_text_vecs, axis=1)
print(f"\n  Norm check: min={norms.min():.6f}  max={norms.max():.6f}  "
      f"{'✅' if norms.min()>0.999 and norms.max()<1.001 else '❌ NOT NORMALIZED'}")
del norms; gc.collect()

# Build ID list then free metadata
text_id_list = [
    m.get("chunk_id") or f"{m.get('pmid', m.get('pmcid','?'))}__ABS__c{i:04d}"
    for i, m in enumerate(all_text_meta)
]
index_mapping["text"] = text_id_list
del all_text_meta; gc.collect()

# Build index
print(f"\n  Building IVFFlat index...")
text_index, idx_type = build_faiss_index(all_text_vecs)
print(f"  Index type : {idx_type}")
print(f"  Ntotal     : {text_index.ntotal}")

verify_index(text_index, all_text_vecs)
del all_text_vecs; gc.collect()

faiss.write_index(text_index, str(FAISS_ROOT / "text_chunks_index.faiss"))
print(f"  Saved → text_chunks_index.faiss  "
      f"({(FAISS_ROOT/'text_chunks_index.faiss').stat().st_size/1e6:.1f} MB)")
print(f"  Total time: {time.time()-t0:.1f}s")
del text_index; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  INDEX 2 — FIGURE CAPTIONS BERT (S-PubMedBERT, 768-dim)")
print(SEPARATOR)

t0 = time.time()

print("\n  Loading BERT figcap embeddings...")
fc_bert_vecs       = np.load(EMB_ROOT / "emb_figcaps_spubmedbert.npy", mmap_mode='r')
fc_bert_meta       = load_jsonl(EMB_ROOT / "emb_figcaps_meta.jsonl")
fc_bert_vecs_clean = np.array(fc_bert_vecs[fig_bert_idx], dtype=np.float32)
fc_bert_meta_clean = [fc_bert_meta[i] for i in fig_bert_idx]
del fc_bert_vecs, fc_bert_meta; gc.collect()
print(f"  BERT figcap: {fc_bert_vecs_clean.shape}")

norms = np.linalg.norm(fc_bert_vecs_clean, axis=1)
print(f"  Norm check: min={norms.min():.6f}  max={norms.max():.6f}  "
      f"{'✅' if norms.min()>0.999 and norms.max()<1.001 else '❌'}")
del norms; gc.collect()

fig_bert_id_list = [f"{m['pmcid']}_{m['fig_id']}" for m in fc_bert_meta_clean]
index_mapping["figure_captions_bert"] = fig_bert_id_list
del fc_bert_meta_clean; gc.collect()

print("\n  Building index...")
fc_bert_index, idx_type = build_faiss_index(fc_bert_vecs_clean)
print(f"  Index type : {idx_type}")
print(f"  Ntotal     : {fc_bert_index.ntotal}")

verify_index(fc_bert_index, fc_bert_vecs_clean)
del fc_bert_vecs_clean; gc.collect()

faiss.write_index(fc_bert_index, str(FAISS_ROOT / "figure_captions_bert_index.faiss"))
print(f"  Saved → figure_captions_bert_index.faiss  "
      f"({(FAISS_ROOT/'figure_captions_bert_index.faiss').stat().st_size/1e6:.1f} MB)")
print(f"  Total time: {time.time()-t0:.1f}s")
del fc_bert_index; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  INDEX 3 — FIGURE CAPTIONS CLIP (BiomedCLIP, 512-dim)")
print(SEPARATOR)

t0 = time.time()

print("\n  Loading CLIP figcap embeddings...")
fc_clip_vecs       = np.load(EMB_ROOT / "emb_figcaps_biomedclip.npy", mmap_mode='r')
fc_clip_meta       = load_jsonl(EMB_ROOT / "emb_figcaps_biomedclip_meta.jsonl")
fc_clip_vecs_clean = np.array(fc_clip_vecs[fig_clip_cap_idx], dtype=np.float32)
fc_clip_meta_clean = [fc_clip_meta[i] for i in fig_clip_cap_idx]
del fc_clip_vecs, fc_clip_meta; gc.collect()
print(f"  CLIP figcap: {fc_clip_vecs_clean.shape}")

norms = np.linalg.norm(fc_clip_vecs_clean, axis=1)
print(f"  Norm check: min={norms.min():.6f}  max={norms.max():.6f}  "
      f"{'✅' if norms.min()>0.999 and norms.max()<1.001 else '❌'}")
del norms; gc.collect()

fig_clip_cap_id_list = [f"{m['pmcid']}_{m['fig_id']}" for m in fc_clip_meta_clean]
index_mapping["figure_captions_clip"] = fig_clip_cap_id_list
del fc_clip_meta_clean; gc.collect()

print("\n  Building index...")
fc_clip_index, idx_type = build_faiss_index(fc_clip_vecs_clean)
print(f"  Index type : {idx_type}")
print(f"  Ntotal     : {fc_clip_index.ntotal}")

verify_index(fc_clip_index, fc_clip_vecs_clean)
del fc_clip_vecs_clean; gc.collect()

faiss.write_index(fc_clip_index, str(FAISS_ROOT / "figure_captions_index.faiss"))
print(f"  Saved → figure_captions_index.faiss  "
      f"({(FAISS_ROOT/'figure_captions_index.faiss').stat().st_size/1e6:.1f} MB)")
print(f"  Total time: {time.time()-t0:.1f}s")
del fc_clip_index; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  INDEX 4 — FIGURE IMAGES CLIP (BiomedCLIP, 512-dim)")
print(SEPARATOR)

t0 = time.time()

print("\n  Loading CLIP figimg embeddings...")
fi_clip_vecs       = np.load(EMB_ROOT / "emb_figures_biomedclip_images.npy", mmap_mode='r')
fi_clip_meta       = load_jsonl(EMB_ROOT / "emb_figures_biomedclip_images_meta.jsonl")
fi_clip_vecs_clean = np.array(fi_clip_vecs[fig_clip_img_idx], dtype=np.float32)
fi_clip_meta_clean = [fi_clip_meta[i] for i in fig_clip_img_idx]
del fi_clip_vecs, fi_clip_meta; gc.collect()
print(f"  CLIP figimg: {fi_clip_vecs_clean.shape}")

norms = np.linalg.norm(fi_clip_vecs_clean, axis=1)
print(f"  Norm check: min={norms.min():.6f}  max={norms.max():.6f}  "
      f"{'✅' if norms.min()>0.999 and norms.max()<1.001 else '❌'}")
del norms; gc.collect()

fig_clip_img_id_list = [f"{m['pmcid']}_{m['fig_id']}" for m in fi_clip_meta_clean]
index_mapping["figure_images_clip"] = fig_clip_img_id_list
del fi_clip_meta_clean; gc.collect()

print("\n  Building index...")
fi_clip_index, idx_type = build_faiss_index(fi_clip_vecs_clean)
print(f"  Index type : {idx_type}")
print(f"  Ntotal     : {fi_clip_index.ntotal}")

verify_index(fi_clip_index, fi_clip_vecs_clean)
del fi_clip_vecs_clean; gc.collect()

faiss.write_index(fi_clip_index, str(FAISS_ROOT / "figure_images_index.faiss"))
print(f"  Saved → figure_images_index.faiss  "
      f"({(FAISS_ROOT/'figure_images_index.faiss').stat().st_size/1e6:.1f} MB)")
print(f"  Total time: {time.time()-t0:.1f}s")
del fi_clip_index; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  SAVING INDEX MAPPING")
print(SEPARATOR)

mapping_summary = {
    k: {"count": len(v), "sample": v[:3]}
    for k, v in index_mapping.items()
}
with open(FAISS_ROOT / "index_mapping.json", "w") as f:
    json.dump(mapping_summary, f, indent=2)
print(f"  Saved → index_mapping.json")

for key, id_list in index_mapping.items():
    arr_path = FAISS_ROOT / f"{key}_ids.npy"
    np.save(arr_path, np.array(id_list, dtype=object))
    print(f"  Saved → {arr_path.name}  ({len(id_list)} IDs)")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  BUILD COMPLETE — FINAL SUMMARY")
print(SEPARATOR)

for f in sorted(FAISS_ROOT.glob("*.faiss")):
    print(f"  {f.name:<48} {f.stat().st_size/1e6:.1f} MB")
for f in sorted(FAISS_ROOT.glob("*.npy")):
    print(f"  {f.name:<48} {f.stat().st_size/1e6:.1f} MB")
for f in sorted(FAISS_ROOT.glob("*.json")):
    print(f"  {f.name:<48} {f.stat().st_size/1e6:.2f} MB")

print(f"\n  index_mapping.json:")
print(json.dumps(mapping_summary, indent=4))

print(f"\n{SEPARATOR}")
print("  ✅  ALL 4 FAISS INDICES BUILT SUCCESSFULLY")
print(SEPARATOR)
