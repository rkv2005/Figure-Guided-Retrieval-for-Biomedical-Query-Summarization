# ============================================================
# RERANK RETRIEVER — V3
# FAISS + BM25 → RRF fusion → Cross-Encoder rerank → MMR → top-20
# ============================================================

import numpy as np
import torch
from scipy.sparse import issparse


class RerankRetriever:
    """
    Hybrid retrieval with cross-encoder reranking and MMR diversification.
    Query → FAISS top-100 + BM25 top-100 → RRF → CE rerank top-100
          → drop score ≤ 0.0 → MMR k=20 λ=0.6 → top-20
    """

    VARIANT      = "rerank"
    RRF_K        = 60
    CE_BATCH     = 64       # cross-encoder batch size
    MMR_LAMBDA   = 0.6      # relevance weight in MMR (1-λ = diversity)
    MMR_K        = 20       # final output size

    def __init__(self, loader):
        self.loader = loader

        # Move CE to GPU if available
        self._ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._ce_device == "cuda":
            self.loader.cross_encoder.model.to("cuda")
        print(f"   Cross-Encoder device : {self._ce_device}")
        print(f"   FAISS total vectors  : {self.loader.idx_text.ntotal:,}")
        print(f"   TF-IDF matrix        : {self.loader.tfidf_matrix.shape}")

    @staticmethod
    def _parse_chunk_id(chunk_id: str) -> tuple[str, str]:
        s = str(chunk_id)
        if s.startswith("PMID"):
            return None, s.split("__")[0].replace("PMID", "")
        elif s.startswith("PMC"):
            return s.split("__")[0], None
        return None, None

    def _bm25_search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        q_vec  = self.loader.tfidf_model.transform([query])
        scores = self.loader.tfidf_matrix @ q_vec.T
        if issparse(scores):
            scores = scores.toarray().flatten()
        else:
            scores = np.asarray(scores).flatten()

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
        return 1.0 / (self.RRF_K + rank)

    def _cross_encode(self, query: str, chunks: list[dict]) -> list[float]:
        """
        Score (query, chunk_text) pairs with cross-encoder.
        Returns list of float scores, same length as chunks.
        """
        pairs  = [(query, c['text'][:512]) for c in chunks]
        scores = self.loader.cross_encoder.predict(
            pairs,
            batch_size  = self.CE_BATCH,
            show_progress_bar = False,
        )
        return [float(s) for s in scores]

    def _mmr(
        self,
        query        : str,
        chunks       : list[dict],
        k            : int,
        lam          : float,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance selection.
        Uses CE scores as relevance, SBERT cosine as diversity penalty.
        Returns top-k chunks maximising λ·relevance - (1-λ)·max_sim_to_selected.
        """
        if len(chunks) <= k:
            return chunks

        # Embed all chunk texts for diversity computation
        texts     = [c['text'][:256] for c in chunks]
        embs      = self.loader.bert_model.encode(
            texts,
            convert_to_numpy   = True,
            normalize_embeddings = True,
            show_progress_bar  = False,
            batch_size         = 64,
        )                                   # shape (n, 768)

        ce_scores = np.array([c['ce_score'] for c in chunks])

        # Normalise CE scores to [0, 1] for MMR
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_scores)

        selected_indices = []
        remaining        = list(range(len(chunks)))

        for _ in range(k):
            if not remaining:
                break

            if not selected_indices:
                # First pick: highest CE score
                best = max(remaining, key=lambda i: ce_norm[i])
            else:
                # MMR: λ·relevance - (1-λ)·max_sim_to_selected
                sel_embs   = embs[selected_indices]           # (s, 768)
                rem_embs   = embs[remaining]                  # (r, 768)
                sim_matrix = rem_embs @ sel_embs.T            # (r, s)
                max_sim    = sim_matrix.max(axis=1)           # (r,)

                mmr_scores = (
                    lam * ce_norm[remaining]
                    - (1 - lam) * max_sim
                )
                best = remaining[int(np.argmax(mmr_scores))]

            selected_indices.append(best)
            remaining.remove(best)

        return [chunks[i] for i in selected_indices]

    def retrieve(
        self,
        query        : str,
        top_k        : int = 20,
        n_candidates : int = 100,   # larger pool for reranking
    ) -> dict:

        # ── 1. FAISS dense search ─────────────────────────────
        q_emb    = self.loader.bert_model.encode(
            [query],
            show_progress_bar    = False,
            convert_to_numpy     = True,
            normalize_embeddings = True,
        ).astype("float32")

        n_search = min(n_candidates, self.loader.idx_text.ntotal)
        D, I     = self.loader.idx_text.search(q_emb, n_search)

        faiss_ranks, faiss_scores = {}, {}
        for rank, (raw_dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx == -1:
                continue
            cid = str(self.loader.text_ids[idx])
            faiss_ranks[cid]  = rank
            faiss_scores[cid] = float(raw_dist)

        # ── 2. BM25 sparse search ─────────────────────────────
        bm25_results        = self._bm25_search(query, n_candidates)
        bm25_ranks, bm25_scores = {}, {}
        for rank, (cid, score) in enumerate(bm25_results, 1):
            bm25_ranks[cid]  = rank
            bm25_scores[cid] = score

        # ── 3. RRF fusion ─────────────────────────────────────
        all_ids    = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores = {
            cid: (
                (self._rrf_score(faiss_ranks[cid]) if cid in faiss_ranks else 0.0) +
                (self._rrf_score(bm25_ranks[cid])  if cid in bm25_ranks  else 0.0)
            )
            for cid in all_ids
        }
        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── 4. Build candidate pool (top-100 by RRF) ──────────
        candidate_pool = []
        seen_ids       = set()

        for cid in ranked_ids:
            if len(candidate_pool) >= n_candidates:
                break
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

            candidate_pool.append({
                "chunk_id"   : cid,
                "text"       : text,
                "pmcid"      : pmcid,
                "pmid"       : pmid,
                "sec_id"     : meta.get('sec_id'),
                "domain"     : domain,
                "faiss_score": faiss_scores.get(cid, 0.0),
                "bm25_score" : bm25_scores.get(cid, 0.0),
                "rrf_score"  : rrf_scores[cid],
                "ce_score"   : 0.0,   # filled below
            })

        # ── 5. Cross-encoder reranking ────────────────────────
        ce_scores_list = self._cross_encode(query, candidate_pool)
        for chunk, ce_s in zip(candidate_pool, ce_scores_list):
            chunk['ce_score'] = ce_s
            chunk['score']    = chunk['rrf_score']   # ← RRF drives ordering
        
        candidate_pool = [c for c in candidate_pool if c['ce_score'] > 0.0]
        candidate_pool.sort(key=lambda x: x['rrf_score'], reverse=True)  # ← sort by RRF

        # ── 6. MMR diversification ────────────────────────────
        top_chunks = self._mmr(
            query  = query,
            chunks = candidate_pool,
            k      = self.MMR_K,
            lam    = self.MMR_LAMBDA,
        )

        # ── 7. Build retrieved_papers ─────────────────────────
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
                "variant"          : self.VARIANT,
                "n_candidates"     : n_search,
                "pool_size"        : len(candidate_pool),
                "returned"         : len(top_chunks),
                "ce_device"        : self._ce_device,
                "n_after_ce_filter": len(candidate_pool),
            }
        }
