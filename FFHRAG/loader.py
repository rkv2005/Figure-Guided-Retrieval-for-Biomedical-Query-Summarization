# ============================================================
# FFHRetrieverLoader — Phase 2 ablation, fully fixed
# ============================================================

import os, json, pickle, csv as _csv, warnings
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')


class FFHRetrieverLoader:
    """
    Loads all retrieval indices, models, metadata, and bridge CSVs.
    All retriever models pinned to CPU — GPU reserved for generator.
    """

    def __init__(self, base_path: str = "/kaggle/input/ffhrag-store"):
        print("⚙️  Initialising FFHRetrieverLoader...")
        self.base_path = base_path

        # ── Bi-encoders — CPU only ────────────────────────────
        print("   Loading Bi-Encoders (CPU)...")
        self.bert_model = SentenceTransformer(
            'pritamdeka/S-PubMedBert-MS-MARCO', device='cuda'
        )
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32', device='cuda'
        )

        # ── FAISS indices ─────────────────────────────────────
        print("   Loading FAISS indices...")
        faiss_path = os.path.join(base_path, "FAISS")
        self.idx_fig_bert      = faiss.read_index(
            os.path.join(faiss_path, "figure_captions_bert_index.faiss"))
        self.idx_fig_clip_text = faiss.read_index(
            os.path.join(faiss_path, "figure_captions_index.faiss"))
        self.idx_fig_clip_img  = faiss.read_index(
            os.path.join(faiss_path, "figure_images_index.faiss"))
        self.idx_text          = faiss.read_index(
            os.path.join(faiss_path, "text_chunks_index.faiss"))

        # Load text_ids for safe chunk_id lookup by FAISS position
        self.text_ids = np.load(
            os.path.join(faiss_path, "text_ids.npy"), allow_pickle=True
        )
        print(f"      Text chunks : {self.idx_text.ntotal:,}")
        print(f"      text_ids    : {len(self.text_ids):,}")
        print(f"      Fig (BERT)  : {self.idx_fig_bert.ntotal:,}")

        # ── Cross-Encoder — CPU only ──────────────────────────
        print("   Loading Cross-Encoder (CPU)...")
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda'
        )

        # ── Sparse indices ────────────────────────────────────
        print("   Loading Sparse indices...")
        bm25_path = os.path.join(base_path, "BM25")
        with open(os.path.join(bm25_path, "bm25_figures.pkl"),     'rb') as f:
            self.bm25_fig     = pickle.load(f)
        with open(os.path.join(bm25_path, "bm25_figures_ids.pkl"), 'rb') as f:
            self.bm25_fig_ids = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_model.pkl"), 'rb') as f:
            self.tfidf_model  = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_matrix.pkl"),'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_ids.pkl"),   'rb') as f:
            self.tfidf_ids    = pickle.load(f)

        # ── Metadata — all three chunk meta files ─────────────
        print("   Loading Metadata...")
        emb_path = os.path.join(base_path, "embeddings")

        chunk_meta_files = [
            os.path.join(emb_path, "emb_chunks_meta.jsonl"),
            os.path.join(emb_path, "emb_pmcid_abstract_chunks_spubmedbert_meta.jsonl"),
            os.path.join(emb_path, "emb_pmid_abstract_chunks_spubmedbert_meta.jsonl"),
        ]
        self.chunk_meta = []
        for path in chunk_meta_files:
            entries = self._load_jsonl(path)
            self.chunk_meta.extend(entries)
            print(f"      {len(entries):>8,}  ← {os.path.basename(path)}")

        print(f"      chunk_meta total : {len(self.chunk_meta):,}")

        # Safe O(1) lookup: chunk_id → meta dict
        self.chunk_id_to_meta = {
            m['chunk_id']: m
            for m in self.chunk_meta
            if m.get('chunk_id')
        }
        print(f"      chunk_id_to_meta : {len(self.chunk_id_to_meta):,}")

        self.fig_meta    = self._load_jsonl(
            os.path.join(emb_path, "emb_figcaps_meta.jsonl"))
        self.section_map = self._load_section_map(base_path)
        self._fix_figure_paths()

        self.fig_id_to_idx = {
            m.get('fig_id'): i
            for i, m in enumerate(self.fig_meta)
            if m.get('fig_id')
        }
        print(f"      fig_meta     : {len(self.fig_meta):,}")
        print(f"      section_map  : {len(self.section_map):,}")

        # ── Bridge CSVs ───────────────────────────────────────
        print("   Loading Bridge CSVs...")
        self.pmcid_to_pmid   = {}
        self.pmid_to_pmcid   = {}
        self.pmcid_to_domain = {}
        self.pmid_to_domain  = {}

        for path in [
            os.path.join(base_path, "Metadata/tgz_available_master.csv"),
            os.path.join(base_path, "Metadata/pmids_pmcid_only_no_tgz.csv"),
        ]:
            if not os.path.exists(path):
                print(f"   ⚠️  Bridge file not found: {path}")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    pc  = str(row.get('pmcid')        or '').strip()
                    pm  = str(row.get('pmid')         or '').strip()
                    dom = str(row.get('domain_final') or '').strip()
                    if pc and pm:
                        self.pmcid_to_pmid[pc] = pm
                        self.pmid_to_pmcid[pm] = pc
                    if pc and dom: self.pmcid_to_domain[pc] = dom
                    if pm and dom: self.pmid_to_domain[pm]  = dom

        print(f"      Bridge: {len(self.pmcid_to_pmid):,} pmcid↔pmid mappings")

        # Expose as bridge namespace
        self.bridge = self._Bridge(
            self.pmcid_to_pmid, self.pmid_to_pmcid,
            self.pmcid_to_domain, self.pmid_to_domain,
        )

        print("✅ FFHRetrieverLoader ready\n")

    # ── Bridge namespace ──────────────────────────────────────

    class _Bridge:
        def __init__(self, p2m, m2p, p2d, m2d):
            self._p2m, self._m2p = p2m, m2p
            self._p2d, self._m2d = p2d, m2d

        def get_pmid(self, pmcid):
            return self._p2m.get(str(pmcid or '').strip())

        def get_pmcid(self, pmid):
            return self._m2p.get(str(pmid or '').strip())

        def get_domain(self, pmcid=None, pmid=None):
            if pmcid:
                d = self._p2d.get(str(pmcid).strip())
                if d: return d
            if pmid:
                d = self._m2d.get(str(pmid).strip())
                if d: return d
            return None

    # ── Loaders ───────────────────────────────────────────────

    def _load_jsonl(self, path: str) -> list:
        data = []
        if not os.path.exists(path):
            print(f"   ⚠️  Not found: {path}")
            return data
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

    def _load_section_map(self, base_path: str) -> dict:
        sec_map    = {}
        candidates = [
            os.path.join(base_path, "ffhrag_store", "text", "sections.jsonl"),
            os.path.join(base_path, "text", "sections.jsonl"),
            os.path.join(base_path, "sections.jsonl"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            print("   ⚠️  sections.jsonl not found — hierarchical retrieval disabled")
            return sec_map
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d   = json.loads(line)
                    key = (d.get('pmcid'), d.get('sec_id'))
                    sec_map[key] = d.get('text', '')
                except json.JSONDecodeError:
                    continue
        return sec_map

    def _fix_figure_paths(self):
        """Remap Google Drive paths → Kaggle input paths."""
        prefix_variants = [
            "/content/drive/mydrive/",
            "/content/drive/MyDrive/",
        ]
        for fig in self.fig_meta:
            path = fig.get('image_path', '')
            if not path:
                continue
            for prefix in prefix_variants:
                if path.lower().startswith(prefix.lower()):
                    relative = path[len(prefix):]   # preserve original casing
                    fig['image_path'] = os.path.join(
                        "/kaggle/input/ffhrag-store", relative
                    )
                    break
