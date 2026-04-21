from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import numpy as np

TEXT_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
model = SentenceTransformer(TEXT_MODEL_NAME, device="cuda")

STORE_ROOT = Path("/content/drive/MyDrive/ffhrag_store")
TEXT_ROOT  = STORE_ROOT / "text"
chunks_path  = TEXT_ROOT / "chunks.jsonl"
figures_path = STORE_ROOT / "figures.jsonl"

BATCH_SIZE = 128


def iter_jsonl(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def embed_records(path, text_key, out_vecs_path, out_meta_path):
    texts = []
    metas = []
    for rec in iter_jsonl(path):
        txt = rec[text_key].strip()
        if not txt:
            continue
        texts.append(txt)
        metas.append(rec)

    n = len(texts)
    print(f"Records to embed: {n}")

    all_vecs = np.zeros((n, model.get_sentence_embedding_dimension()), dtype="float32")

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ← THE FIX: L2-normalize before returning
            show_progress_bar=False,
        )
        all_vecs[i : i + len(batch)] = emb
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  {i}/{n}")

    # ── Norm verification (must print ~1.0 / ~1.0 for a clean index) ──
    norms = np.linalg.norm(all_vecs, axis=1)
    print(f"Norm check → min: {norms.min():.6f}  max: {norms.max():.6f}  mean: {norms.mean():.6f}")
    assert norms.min() > 0.999 and norms.max() < 1.001, \
        f"Normalization failed! min={norms.min():.4f} max={norms.max():.4f}"

    np.save(out_vecs_path, all_vecs)
    with out_meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved → {out_vecs_path.name}  ({n} vectors, dim={all_vecs.shape[1]})\n")


# ── Chunks ──
embed_records(
    chunks_path,
    text_key="text",
    out_vecs_path=STORE_ROOT / "emb_chunks_spubmedbert.npy",
    out_meta_path=STORE_ROOT / "emb_chunks_meta.jsonl",
)

# ── Figure captions ──
embed_records(
    figures_path,
    text_key="caption_text",
    out_vecs_path=STORE_ROOT / "emb_figcaps_spubmedbert.npy",
    out_meta_path=STORE_ROOT / "emb_figcaps_meta.jsonl",
)
