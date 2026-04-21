# ============================================================
# VANILLA RAG RETRIEVER
# FAISS dense only → top-60 candidates → top-20 by score
# ============================================================

import numpy as np


class VanillaRetriever:
    """
    Pure dense retrieval baseline.
    Query → PubMedBERT FAISS → top-60 candidates → top-20 by score.
    No BM25, no cross-encoder, no MMR, no section expansion, no figures.
    """

    VARIANT = "vanilla"

    def __init__(self, loader):
        self.loader = loader
        # Confirm index type at init — critical for score interpretation
        index_type = type(self.loader.idx_text).__name__
        print(f"   FAISS index type : {index_type}")
        print(f"   Total vectors    : {self.loader.idx_text.ntotal:,}")

    @staticmethod
    def _parse_chunk_id(chunk_id: str) -> tuple[str, str]:
        """
        Parse any chunk ID format into (pmcid, pmid).
        PMC3430043__sec1-1__c0000  → ('PMC3430043', None)
        PMID33337619__ABS__c0000   → (None, '33337619')
        """
        s = str(chunk_id)
        if s.startswith("PMID"):
            return None, s.split("__")[0].replace("PMID", "")
        elif s.startswith("PMC"):
            return s.split("__")[0], None
        return None, None

    def retrieve(self, query: str, top_k: int = 20, n_candidates: int = 60) -> dict:
        """
        Returns standardised retrieval result dict:
            chunks           → used by prompt_builder
            retrieved_papers → used by evaluator
            figures          → empty for vanilla
            sections         → empty for vanilla
            metadata         → diagnostics
        """

        # ── Dense FAISS search ────────────────────────────────
        q_emb = self.loader.bert_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,      # ← critical for IP index
        ).astype("float32")
        
        n_search = min(n_candidates, self.loader.idx_text.ntotal)
        D, I     = self.loader.idx_text.search(q_emb, n_search)

        # ── Collect + deduplicate chunks ──────────────────────
        chunk_pool = []
        seen_ids   = set()

        for raw_dist, i in zip(D[0], I[0]):
            if i == -1 or i >= len(self.loader.chunk_meta):
                continue

            chunk_id = str(self.loader.text_ids[i])
            meta     = self.loader.chunk_id_to_meta.get(chunk_id, {})
            cid      = meta.get('chunk_id') or chunk_id 

            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            # Skip empty text chunks
            text = (meta.get('text') or "").strip()
            if not text:
                continue

            # ── Parse pmcid / pmid from meta, fallback to chunk_id ──
            pmcid = meta.get('pmcid') or None
            pmid  = str(meta.get('pmid')) if meta.get('pmid') else None

            if not pmcid and not pmid:
                pmcid, pmid = self._parse_chunk_id(cid)

            # ── Enrich domain + missing ID side from bridge ──────────
            if pmcid and not pmid:
                pmid = self.loader.bridge.get_pmid(pmcid)
            if pmid and not pmcid:
                pmcid = self.loader.bridge.get_pmcid(pmid)

            domain = meta.get('domain') or self.loader.bridge.get_domain(pmcid, pmid)

            similarity = float(1 / (1 + raw_dist))

            chunk_pool.append({
                "chunk_id"   : cid,
                "text"       : text,
                "pmcid"      : pmcid,
                "pmid"       : pmid,
                "sec_id"     : meta.get('sec_id'),
                "domain"     : domain,
                "faiss_score": float(raw_dist),
                "score"      : similarity,
            })

        # ── Sort by similarity descending, take top_k ─────────
        chunk_pool.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = chunk_pool[:top_k]

        # ── Build retrieved_papers (deduped, order-preserved) ─
        retrieved_papers = []
        seen_papers      = set()

        for chunk in top_chunks:
            pmid  = chunk.get('pmid')
            pmcid = chunk.get('pmcid')
            key   = pmid or pmcid
            if key and key not in seen_papers:
                seen_papers.add(key)
                retrieved_papers.append({
                    "pmid"  : pmid,
                    "pmcid" : pmcid,
                    "domain": chunk.get('domain'),
                })

        return {
            "figures"          : [],
            "sections"         : [],
            "chunks"           : top_chunks,
            "retrieved_papers" : retrieved_papers,
            "metadata"         : {
                "variant"     : self.VARIANT,
                "n_candidates": n_search,
                "pool_size"   : len(chunk_pool),
                "returned"    : len(top_chunks),
            }
        }
