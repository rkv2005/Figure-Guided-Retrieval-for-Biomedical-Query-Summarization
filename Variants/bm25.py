# ============================================================
# BM25 HYBRID RETRIEVER
# FAISS dense + BM25 sparse → RRF fusion → top-20
# ============================================================

import numpy as np
from scipy.sparse import issparse


class BM25Retriever:
    """
    Hybrid retrieval: FAISS dense + BM25 TF-IDF sparse, fused via RRF.
    Query → FAISS top-60 + BM25 top-60 → RRF fusion → top-20.
    No cross-encoder, no MMR, no section expansion, no figures.
    """

    VARIANT    = "bm25"
    RRF_K      = 60      # RRF constant — standard value

    def __init__(self, loader):
        self.loader = loader
        index_type  = type(self.loader.idx_text).__name__
        print(f"   FAISS index type : {index_type}")
        print(f"   Total vectors    : {self.loader.idx_text.ntotal:,}")
        print(f"   TF-IDF matrix    : {self.loader.tfidf_matrix.shape}")
        print(f"   TF-IDF ids       : {len(self.loader.tfidf_ids):,}")

    @staticmethod
    def _parse_chunk_id(chunk_id: str) -> tuple[str, str]:
        s = str(chunk_id)
        if s.startswith("PMID"):
            return None, s.split("__")[0].replace("PMID", "")
        elif s.startswith("PMC"):
            return s.split("__")[0], None
        return None, None

    def _bm25_search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        """
        TF-IDF cosine search.
        Returns list of (chunk_id, score) sorted descending, length top_n.
        """
        q_vec   = self.loader.tfidf_model.transform([query])
        scores  = (self.loader.tfidf_matrix @ q_vec.T)

        # Handle both sparse and dense matrix results
        if issparse(scores):
            scores = scores.toarray().flatten()
        else:
            scores = np.asarray(scores).flatten()

        # Take top_n by score
        top_indices = np.argpartition(scores, -min(top_n, len(scores)))[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            if idx < len(self.loader.tfidf_ids):
                results.append((str(self.loader.tfidf_ids[idx]), score))

        return results[:top_n]

    def _rrf_score(self, rank: int) -> float:
        """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
        return 1.0 / (self.RRF_K + rank)

    def retrieve(self, query: str, top_k: int = 20, n_candidates: int = 60) -> dict:
        """
        Returns standardised retrieval result dict:
            chunks           → used by prompt_builder
            retrieved_papers → used by evaluator
            figures          → empty
            sections         → empty
            metadata         → diagnostics
        """

        # ── 1. FAISS dense search ─────────────────────────────
        q_emb    = self.loader.bert_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        n_search = min(n_candidates, self.loader.idx_text.ntotal)
        D, I     = self.loader.idx_text.search(q_emb, n_search)

        # Map: chunk_id → FAISS rank (1-indexed)
        faiss_ranks  = {}
        faiss_scores = {}
        for rank, (raw_dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx == -1:
                continue
            chunk_id = str(self.loader.text_ids[idx])
            faiss_ranks[chunk_id]  = rank
            faiss_scores[chunk_id] = float(raw_dist)  # raw IP score

        # ── 2. BM25 / TF-IDF sparse search ───────────────────
        bm25_results = self._bm25_search(query, n_candidates)

        # Map: chunk_id → BM25 rank (1-indexed)
        bm25_ranks  = {}
        bm25_scores = {}
        for rank, (chunk_id, score) in enumerate(bm25_results, 1):
            bm25_ranks[chunk_id]  = rank
            bm25_scores[chunk_id] = score

        # ── 3. RRF fusion ─────────────────────────────────────
        all_chunk_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())

        rrf_scores = {}
        for cid in all_chunk_ids:
            score = 0.0
            if cid in faiss_ranks:
                score += self._rrf_score(faiss_ranks[cid])
            if cid in bm25_ranks:
                score += self._rrf_score(bm25_ranks[cid])
            rrf_scores[cid] = score

        # Sort by RRF score descending
        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── 4. Build chunk pool ───────────────────────────────
        chunk_pool = []
        seen_ids   = set()

        for cid in ranked_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = self.loader.chunk_id_to_meta.get(cid, {})
            text = (meta.get('text') or "").strip()
            if not text:
                continue

            pmcid = meta.get('pmcid') or None
            pmid  = str(meta.get('pmid')) if meta.get('pmid') else None

            if not pmcid and not pmid:
                pmcid, pmid = self._parse_chunk_id(cid)

            if pmcid and not pmid:
                pmid  = self.loader.bridge.get_pmid(pmcid)
            if pmid and not pmcid:
                pmcid = self.loader.bridge.get_pmcid(pmid)

            domain = meta.get('domain') or self.loader.bridge.get_domain(pmcid, pmid)

            chunk_pool.append({
                "chunk_id"   : cid,
                "text"       : text,
                "pmcid"      : pmcid,
                "pmid"       : pmid,
                "sec_id"     : meta.get('sec_id'),
                "domain"     : domain,
                "faiss_score": faiss_scores.get(cid, 0.0),
                "bm25_score" : bm25_scores.get(cid, 0.0),
                "score"      : rrf_scores[cid],   # RRF is the ranking score
            })

        top_chunks = chunk_pool[:top_k]

        # ── 5. Build retrieved_papers ─────────────────────────
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

        # ── 6. Metadata ───────────────────────────────────────
        n_faiss_only = sum(
            1 for cid in ranked_ids[:top_k]
            if cid in faiss_ranks and cid not in bm25_ranks
        )
        n_bm25_only  = sum(
            1 for cid in ranked_ids[:top_k]
            if cid in bm25_ranks and cid not in faiss_ranks
        )
        n_both       = sum(
            1 for cid in ranked_ids[:top_k]
            if cid in faiss_ranks and cid in bm25_ranks
        )

        return {
            "figures"          : [],
            "sections"         : [],
            "chunks"           : top_chunks,
            "retrieved_papers" : retrieved_papers,
            "metadata"         : {
                "variant"      : self.VARIANT,
                "n_candidates" : n_search,
                "pool_size"    : len(chunk_pool),
                "returned"     : len(top_chunks),
                "n_faiss_only" : n_faiss_only,
                "n_bm25_only"  : n_bm25_only,
                "n_both"       : n_both,
            }
        }
