import os, json, pickle, csv as _csv, warnings
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')


class FFHRetrieverLoader:

    def __init__(self, base_path: str = "/kaggle/input/ffhrag-store"):
        print("⚙️  Initialising FFHRetrieverLoader...")
        self.base_path = base_path

        # Dynamically target local GPU space
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ── Bi-encoders — Offloaded to GPU ────────────────────
        print(f"   Loading Text Search Bi-Encoders ({self.device.upper()})...")
        self.bert_model = SentenceTransformer(
            'pritamdeka/S-PubMedBert-MS-MARCO', device=self.device
        )
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32', device=self.device
        )

        # ── NEW: Paired MedCPT Models for FigureBoost Space Alignment ──
        print(f"   Loading Symmetric MedCPT Bi-Encoders ({self.device.upper()})...")
        self.article_encoder = SentenceTransformer("ncbi/MedCPT-Article-Encoder", device=self.device)
        self.query_encoder   = SentenceTransformer("ncbi/MedCPT-Query-Encoder", device=self.device)

        # ── FAISS indices ─────────────────────────────────────
        print("   Loading FAISS indices...")
        faiss_path = os.path.join(base_path, "FAISS")
        self.idx_fig_bert      = faiss.read_index(os.path.join(faiss_path, "figure_captions_bert_index.faiss"))
        self.idx_fig_clip_text = faiss.read_index(os.path.join(faiss_path, "figure_captions_index.faiss"))
        self.idx_fig_clip_img  = faiss.read_index(os.path.join(faiss_path, "figure_images_index.faiss"))
        self.idx_text          = faiss.read_index(os.path.join(faiss_path, "text_chunks_index.faiss"))

        self.text_ids = np.load(os.path.join(faiss_path, "text_ids.npy"), allow_pickle=True)
        print(f"      Text chunks : {self.idx_text.ntotal:,}")
        print(f"      text_ids    : {len(self.text_ids):,}")
        print(f"      Fig (BERT)  : {self.idx_fig_bert.ntotal:,}")

        # ── Cross-Encoder — Upgraded to Biomedical Weights & GPU ──
        print(f"   Loading Biomedical Cross-Encoder ({self.device.upper()})...")
        self.cross_encoder = CrossEncoder('ncbi/MedCPT-Cross-Encoder', device=self.device)

        # ── Sparse indices ────────────────────────────────────
        print("   Loading Sparse indices...")
        bm25_path = os.path.join(base_path, "BM25")
        with open(os.path.join(bm25_path, "bm25_figures.pkl"),      'rb') as f: self.bm25_fig     = pickle.load(f)
        with open(os.path.join(bm25_path, "bm25_figures_ids.pkl"),  'rb') as f: self.bm25_fig_ids = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_model.pkl"),  'rb') as f: self.tfidf_model  = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_matrix.pkl"), 'rb') as f: self.tfidf_matrix = pickle.load(f)
        with open(os.path.join(bm25_path, "tfidf_text_ids.pkl"),    'rb') as f: self.tfidf_ids    = pickle.load(f)

        # ── Metadata loading ──────────────────────────────────
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

        self.chunk_id_to_meta = {m['chunk_id']: m for m in self.chunk_meta if m.get('chunk_id')}
        self.fig_meta    = self._load_jsonl(os.path.join(emb_path, "emb_figcaps_meta.jsonl"))
        self.section_map = self._load_section_map(base_path)
        self._fix_figure_paths()

        self.fig_id_to_idx = {m.get('fig_id'): i for i in range(len(self.fig_meta)) if (m := self.fig_meta[i]).get('fig_id')}
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
        self.bridge = self._Bridge(self.pmcid_to_pmid, self.pmid_to_pmcid, self.pmcid_to_domain, self.pmid_to_domain)
        print("✅ FFHRetrieverLoader ready\n")

    class _Bridge:
        def __init__(self, p2m, m2p, p2d, m2d):
            self._p2m, self._m2p = p2m, m2p
            self._p2d, self._m2d = p2d, m2d
        def get_pmid(self, pmcid): return self._p2m.get(str(pmcid or '').strip())
        def get_pmcid(self, pmid): return self._m2p.get(str(pmid or '').strip())
        def get_domain(self, pmcid=None, pmid=None):
            if pmcid:
                d = self._p2d.get(str(pmcid).strip())
                if d: return d
            if pmid:
                d = self._m2d.get(str(pmid).strip())
                if d: return d
            return None

    def _load_jsonl(self, path: str) -> list:
        data = []
        if not os.path.exists(path): return data
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if (line := line.strip()):
                    try: data.append(json.loads(line))
                    except json.JSONDecodeError: continue
        return data

    def _load_section_map(self, base_path: str) -> dict:
        sec_map = {}
        candidates = [
            os.path.join(base_path, "ffhrag_store", "text", "sections.jsonl"),
            os.path.join(base_path, "text", "sections.jsonl"),
            os.path.join(base_path, "sections.jsonl"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path: return sec_map
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if (line := line.strip()):
                    try:
                        d = json.loads(line)
                        sec_map[(d.get('pmcid'), d.get('sec_id'))] = d.get('text', '')
                    except json.JSONDecodeError: continue
        return sec_map

    def _fix_figure_paths(self):
        prefix_variants = ["/content/drive/mydrive/", "/content/drive/MyDrive/"]
        for fig in self.fig_meta:
            if (path := fig.get('image_path', '')):
                for prefix in prefix_variants:
                    if path.lower().startswith(prefix.lower()):
                        fig['image_path'] = os.path.join("/kaggle/input/ffhrag-store", path[len(prefix):])
                        break
