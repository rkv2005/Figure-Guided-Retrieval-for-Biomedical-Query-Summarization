# ============================================================
# FIGUREBOOST RETRIEVER — V5
# FAISS + BM25 → RRF → FigBoost ×1.2 → CE rerank → MMR → top-20
# Figure retrieval runs first → boost_pmcids → applied to RRF scores
# Figures returned in result but NOT injected into prompt
# ============================================================

import numpy as np
import torch
import faiss
from scipy.sparse import issparse


class FigureBoostRetriever:

    VARIANT      = "figureboost"
    RRF_K        = 60
    CE_BATCH     = 64
    MMR_LAMBDA   = 0.6
    MMR_K        = 20
    FIG_BOOST    = 1.2        # score multiplier for boost_pmcid chunks
    FIG_TOP_K    = 20         # figure candidates from each FAISS channel
    FIG_TOP_N    = 5          # max figures to store in result

    def __init__(self, loader):
        self.loader = loader

        self._ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._ce_device == "cuda":
            self.loader.cross_encoder.model.to("cuda")

        # Build fig_meta composite key index once
        self._fig_meta_index = {
            f"{m.get('pmcid', '')}_{m.get('fig_id', '')}": m
            for m in self.loader.fig_meta
            if m.get('pmcid') and m.get('fig_id')
        }

        # Load figure FAISS ID arrays
        import os
        faiss_path = os.path.join(self.loader.base_path, "FAISS")
        self._fig_bert_ids = np.load(
            os.path.join(faiss_path, "figure_captions_bert_ids.npy"),
            allow_pickle=True
        )
        self._fig_clip_ids = np.load(
            os.path.join(faiss_path, "figure_captions_clip_ids.npy"),
            allow_pickle=True
        )
        self._fig_img_ids  = np.load(
            os.path.join(faiss_path, "figure_images_clip_ids.npy"),
            allow_pickle=True
        )

        print(f"   Cross-Encoder device : {self._ce_device}")
        print(f"   FAISS text vectors   : {self.loader.idx_text.ntotal:,}")
        print(f"   FAISS fig vectors    : {self.loader.idx_fig_bert.ntotal:,}")
        print(f"   fig_meta_index       : {len(self._fig_meta_index):,}")

    # ── Shared helpers (identical to RerankRetriever) ─────────

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

    def _rrf_fuse(self, rank_lists: list[dict]) -> list[tuple[str, float]]:
        fused = {}
        for ranking in rank_lists:
            sorted_items = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_items):
                fused[doc_id] = fused.get(doc_id, 0.0) + self._rrf_score(rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def _cross_encode(self, query: str, chunks: list[dict]) -> list[float]:
        pairs  = [(query, c['text'][:512]) for c in chunks]
        scores = self.loader.cross_encoder.predict(
            pairs,
            batch_size        = self.CE_BATCH,
            show_progress_bar = False,
        )
        return [float(s) for s in scores]

    def _mmr(self, chunks: list[dict], k: int, lam: float) -> list[dict]:
        if len(chunks) <= k:
            return chunks
        texts = [c['text'][:256] for c in chunks]
        embs  = self.loader.bert_model.encode(
            texts,
            convert_to_numpy     = True,
            normalize_embeddings = True,
            show_progress_bar    = False,
            batch_size           = 64,
        )
        ce_scores      = np.array([c['ce_score'] for c in chunks])
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        ce_norm = (
            (ce_scores - ce_min) / (ce_max - ce_min)
            if ce_max > ce_min else np.ones_like(ce_scores)
        )
        selected, remaining = [], list(range(len(chunks)))
        for _ in range(k):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: ce_norm[i])
            else:
                sel_embs   = embs[selected]
                rem_embs   = embs[remaining]
                sim_matrix = rem_embs @ sel_embs.T
                max_sim    = sim_matrix.max(axis=1)
                mmr_scores = lam * ce_norm[remaining] - (1 - lam) * max_sim
                best       = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)
        return [chunks[i] for i in selected]

    # ── Figure retrieval ──────────────────────────────────────
    def _search_figures(self, query: str) -> tuple[list[dict], set[str]]:
        """
        4-channel RRF candidate pool → CE rerank (S2) → top-FIG_TOP_N
        CLIP channels contribute to candidate diversity, CE does final selection.
        """
        top_k = self.FIG_TOP_K

        # PubMedBERT query embedding
        q_bert = self.loader.bert_model.encode(
            [query], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # CLIP query embedding
        q_clip = self.loader.clip_model.encode(
            [query], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        # Channel 1 — PubMedBERT captions
        n    = min(top_k, self.loader.idx_fig_bert.ntotal)
        D, I = self.loader.idx_fig_bert.search(q_bert, n)
        bert_hits = {
            str(self._fig_bert_ids[i]): float(s)
            for s, i in zip(D[0], I[0])
            if i != -1 and i < len(self._fig_bert_ids)
        }

        # Channel 2 — CLIP text captions
        n    = min(top_k, self.loader.idx_fig_clip_text.ntotal)
        D, I = self.loader.idx_fig_clip_text.search(q_clip, n)
        clip_text_hits = {
            str(self._fig_clip_ids[i]): float(s)
            for s, i in zip(D[0], I[0])
            if i != -1 and i < len(self._fig_clip_ids)
        }

        # Channel 3 — CLIP image embeddings
        n    = min(top_k, self.loader.idx_fig_clip_img.ntotal)
        D, I = self.loader.idx_fig_clip_img.search(q_clip, n)
        clip_img_hits = {
            str(self._fig_img_ids[i]): float(s)
            for s, i in zip(D[0], I[0])
            if i != -1 and i < len(self._fig_img_ids)
        }

        # Channel 4 — BM25 captions
        tokenized_query = query.lower().split()
        bm25_raw_scores = self.loader.bm25_fig.get_scores(tokenized_query)
        top_indices     = np.argpartition(
            bm25_raw_scores, -min(top_k, len(bm25_raw_scores))
        )[-top_k:]
        top_indices     = top_indices[np.argsort(bm25_raw_scores[top_indices])[::-1]]
        bm25_hits       = {
            str(self.loader.bm25_fig_ids[idx]): float(bm25_raw_scores[idx])
            for idx in top_indices
            if bm25_raw_scores[idx] > 0 and idx < len(self.loader.bm25_fig_ids)
        }

        # ── 4-channel RRF → candidate pool ───────────────────
        fused = self._rrf_fuse([bert_hits, clip_text_hits, clip_img_hits, bm25_hits])

        # Build enriched candidate list for CE reranking
        candidates = []
        seen_fids  = set()
        for fid, base_rrf in fused[:top_k]:
            if fid in seen_fids:
                continue
            seen_fids.add(fid)
            meta    = self._fig_meta_index.get(fid)
            if not meta:
                continue
            caption = (meta.get('caption_text') or '').strip()
            pmcid   = meta.get('pmcid', '')
            if not pmcid or not caption:
                continue
            candidates.append({
                "fid"     : fid,
                "pmcid"   : pmcid,
                "caption" : caption,
                "base_rrf": base_rrf,
                "meta"    : meta,
            })

        if not candidates:
            return [], set()

        # ── CE rerank (query, caption) pairs ─────────────────
        pairs     = [(query, c['caption'][:512]) for c in candidates]
        ce_scores = self.loader.cross_encoder.predict(
            pairs,
            batch_size        = self.CE_BATCH,
            show_progress_bar = False,
        )
        for c, s in zip(candidates, ce_scores):
            c['ce_score'] = float(s)

        # ── RRF rank fusion: base_rrf rank + CE rank ──────────
        base_ranked = sorted(candidates, key=lambda x: x['base_rrf'], reverse=True)
        ce_ranked   = sorted(candidates, key=lambda x: x['ce_score'],  reverse=True)

        base_rank_map = {c['fid']: r for r, c in enumerate(base_ranked, 1)}
        ce_rank_map   = {c['fid']: r for r, c in enumerate(ce_ranked,   1)}

        for c in candidates:
            c['final_rrf'] = (
                self._rrf_score(base_rank_map[c['fid']]) +
                self._rrf_score(ce_rank_map[c['fid']])
            )

        candidates.sort(key=lambda x: x['final_rrf'], reverse=True)

        # ── Select top-N, max 2 per PMCID ────────────────────
        selected_figs = []
        boost_pmcids  = set()
        pmcid_counts  = {}

        for c in candidates:
            if len(selected_figs) >= self.FIG_TOP_N:
                break
            pmcid = c['pmcid']
            if pmcid_counts.get(pmcid, 0) >= 2:
                continue
            pmcid_counts[pmcid] = pmcid_counts.get(pmcid, 0) + 1
            boost_pmcids.add(pmcid)
            meta = c['meta']
            selected_figs.append({
                "fig_id"    : c['fid'],
                "pmcid"     : pmcid,
                "fig_label" : meta.get('fig_id', ''),
                "caption"   : c['caption'],
                "image_path": meta.get('image_path', ''),
                "has_image" : meta.get('has_image', False),
                "rrf_score" : round(c['final_rrf'], 6),
                "ce_score"  : round(c['ce_score'],  4),
            })

        return selected_figs, boost_pmcids


    # ── Main retrieve ─────────────────────────────────────────

    def retrieve(
        self,
        query        : str,
        top_k        : int = 20,
        n_candidates : int = 100,
    ) -> dict:

        # ── 1. Figure retrieval → boost_pmcids ────────────────
        selected_figs, boost_pmcids = self._search_figures(query)

        # ── 2. FAISS dense search ─────────────────────────────
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

        # ── 3. BM25 sparse search ─────────────────────────────
        bm25_results            = self._bm25_search(query, n_candidates)
        bm25_ranks, bm25_scores = {}, {}
        for rank, (cid, score) in enumerate(bm25_results, 1):
            bm25_ranks[cid]  = rank
            bm25_scores[cid] = score

        # ── 4. RRF fusion ─────────────────────────────────────
        all_ids    = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores = {
            cid: (
                (self._rrf_score(faiss_ranks[cid]) if cid in faiss_ranks else 0.0) +
                (self._rrf_score(bm25_ranks[cid])  if cid in bm25_ranks  else 0.0)
            )
            for cid in all_ids
        }

        # ── 5. Apply FigBoost ×1.2 to RRF scores ─────────────
        # Boosts chunks from figure-retrieved PMCIDs before CE pool cutoff
        for cid in rrf_scores:
            meta  = self.loader.chunk_id_to_meta.get(cid, {})
            pmcid = meta.get('pmcid') or ''
            if not pmcid:
                pmcid, _ = self._parse_chunk_id(cid)
            if pmcid and pmcid in boost_pmcids:
                rrf_scores[cid] *= self.FIG_BOOST

        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── 6. Build candidate pool ───────────────────────────
        candidate_pool = []
        seen_ids       = set()

        for cid in ranked_ids:
            if len(candidate_pool) >= n_candidates:
                break
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = self.loader.chunk_id_to_meta.get(cid, {})
            text = (meta.get('text') or '').strip()
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
                "chunk_id"    : cid,
                "text"        : text,
                "pmcid"       : pmcid,
                "pmid"        : pmid,
                "sec_id"      : meta.get('sec_id'),
                "domain"      : domain,
                "faiss_score" : faiss_scores.get(cid, 0.0),
                "bm25_score"  : bm25_scores.get(cid, 0.0),
                "rrf_score"   : rrf_scores[cid],
                "fig_boosted" : (pmcid in boost_pmcids) if pmcid else False,
                "ce_score"    : 0.0,
            })

        # ── 7. Cross-encoder reranking ────────────────────────
        ce_scores_list = self._cross_encode(query, candidate_pool)
        for chunk, ce_s in zip(candidate_pool, ce_scores_list):
            chunk['ce_score'] = ce_s
            chunk['score']    = ce_s

        candidate_pool = [c for c in candidate_pool if c['ce_score'] > 0.0]
        candidate_pool.sort(key=lambda x: x['ce_score'], reverse=True)

        # ── 8. MMR diversification ────────────────────────────
        top_chunks = self._mmr(
            candidate_pool[:50],
            k   = self.MMR_K,
            lam = self.MMR_LAMBDA,
        )

        # ── 9. Build retrieved_papers ─────────────────────────
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

        n_boosted = sum(1 for c in top_chunks if c.get('fig_boosted'))

        return {
            "figures"          : selected_figs,
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
                "n_figures"        : len(selected_figs),
                "n_boost_pmcids"   : len(boost_pmcids),
                "n_boosted_chunks" : n_boosted,
                "fig_boost_factor" : self.FIG_BOOST,
            }
        }
