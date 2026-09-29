import os
import numpy as np
import torch
from scipy.sparse import issparse
from sentence_transformers import SentenceTransformer


class FFHRAGRetriever:

    VARIANT             = "ffhrag"  # Hook for downstream prompt caption injection
    RRF_K               = 60
    CE_BATCH            = 64
    MMR_LAMBDA          = 0.6
    MMR_K               = 20
    BETA                = 2.0       # Maps to dynamic amplification scaling factor
    FIG_TOP_K           = 20          
    FIG_TOP_N           = 5           
    FIG_PROMPT_N        = 3         # Top-3 captions targeted for final prompt context window
    CE_FILTER_THRESHOLD = 0.0          
    FIG_CE_THRESHOLD    = -2.0

    def __init__(
        self,
        loader,
        # Sweep parameter overrides (Defaults to class constants if None)
        mmr_lambda : float = None,
        fig_boost  : float = None,
        rrf_k      : int   = None,
        mmr_k      : int   = None,
        n_cands    : int   = None,
        fig_top_k  : int   = None,
        fig_top_n  : int   = None,
        ce_batch   : int   = None,
        threshold_margin : int = None
    ):
        self.loader = loader

        # Bind hyperparameter optimization sweep parameters dynamically
        self.MMR_LAMBDA = mmr_lambda if mmr_lambda is not None else self.MMR_LAMBDA
        self.BETA       = fig_boost  if fig_boost  is not None else self.BETA
        self.RRF_K      = rrf_k      if rrf_k      is not None else self.RRF_K
        self.MMR_K      = mmr_k      if mmr_k      is not None else self.MMR_K
        self.FIG_TOP_K  = fig_top_k  if fig_top_k  is not None else self.FIG_TOP_K
        self.FIG_TOP_N  = fig_top_n  if fig_top_n  is not None else self.FIG_TOP_N
        self.CE_BATCH   = ce_batch   if ce_batch   is not None else self.CE_BATCH
        self._n_cands   = n_cands    if n_cands    is not None else 100
        self.SIM_MARGIN_THRESHOLD = threshold_margin if threshold_margin is not None else 0.85

        self._ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._ce_device == "cuda":
            self.loader.cross_encoder.model.to("cuda")

        # Fallback safeguard: pull paired MedCPT models from loader namespace or mount locally
        self.article_encoder = getattr(self.loader, 'article_encoder', None)
        self.query_encoder   = getattr(self.loader, 'query_encoder', None)
        
        if self.article_encoder is None or self.query_encoder is None:
            print("   Loader hooks unlinked. Bootstrapping MedCPT context models locally...")
            self.article_encoder = SentenceTransformer("ncbi/MedCPT-Article-Encoder")
            self.query_encoder   = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
            if self._ce_device == "cuda":
                self.article_encoder.to("cuda")
                self.query_encoder.to("cuda")

        self._fig_meta_index = {
            f"{m.get('pmcid', '')}_{m.get('fig_id', '')}": m
            for m in self.loader.fig_meta
            if m.get('pmcid') and m.get('fig_id')
        }

        faiss_path = os.path.join(self.loader.base_path, "FAISS")
        self._fig_bert_ids = np.load(os.path.join(faiss_path, "figure_captions_bert_ids.npy"), allow_pickle=True)
        self._fig_clip_ids = np.load(os.path.join(faiss_path, "figure_captions_clip_ids.npy"), allow_pickle=True)
        self._fig_img_ids  = np.load(os.path.join(faiss_path, "figure_images_clip_ids.npy"), allow_pickle=True)

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
            if score <= 0.0: continue
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
        scores = self.loader.cross_encoder.predict(pairs, batch_size=self.CE_BATCH, show_progress_bar=False, activation_fct=lambda x: x)
        return [float(s) for s in scores]

    def _mmr(self, query: str, chunks: list[dict], k: int, lam: float) -> list[dict]:
        if len(chunks) <= k: return chunks
        texts = [c['text'][:256] for c in chunks]
        embs  = self.loader.bert_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        ce_scores      = np.array([c['ce_score'] for c in chunks])
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        ce_norm = ((ce_scores - ce_min) / (ce_max - ce_min) if ce_max > ce_min else np.ones_like(ce_scores))
        selected_indices = []
        remaining        = list(range(len(chunks)))

        for _ in range(k):
            if not remaining: break
            if not selected_indices:
                best = remaining[0]
            else:
                sel_embs   = embs[selected_indices]
                rem_embs   = embs[remaining]
                sim_matrix = rem_embs @ sel_embs.T
                max_sim    = sim_matrix.max(axis=1)
                mmr_scores = lam * ce_norm[remaining] - (1 - lam) * max_sim
                best       = remaining[int(np.argmax(mmr_scores))]
            selected_indices.append(best)
            remaining.remove(best)
        return [chunks[i] for i in selected_indices]

    def _search_figures(self, query: str) -> tuple[list[dict], set[str], dict]:
        top_k = self.FIG_TOP_K
        q_bert = self.loader.bert_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        q_clip = self.loader.clip_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        n = min(top_k, self.loader.idx_fig_bert.ntotal)
        D, I = self.loader.idx_fig_bert.search(q_bert, n)
        bert_hits = {str(self._fig_bert_ids[i]): float(s) for s, i in zip(D[0], I[0]) if i != -1 and i < len(self._fig_bert_ids)}

        n = min(top_k, self.loader.idx_fig_clip_text.ntotal)
        D, I = self.loader.idx_fig_clip_text.search(q_clip, n)
        clip_text_hits = {str(self._fig_clip_ids[i]): float(s) for s, i in zip(D[0], I[0]) if i != -1 and i < len(self._fig_clip_ids)}

        n = min(top_k, self.loader.idx_fig_clip_img.ntotal)
        D, I = self.loader.idx_fig_clip_img.search(q_clip, n)
        clip_img_hits = {str(self._fig_img_ids[i]): float(s) for s, i in zip(D[0], I[0]) if i != -1 and i < len(self._fig_img_ids)}

        tokenized_query = query.lower().split()
        bm25_raw_scores = self.loader.bm25_fig.get_scores(tokenized_query)
        top_indices = np.argpartition(bm25_raw_scores, -min(top_k, len(bm25_raw_scores)))[-top_k:]
        top_indices = top_indices[np.argsort(bm25_raw_scores[top_indices])[::-1]]
        bm25_hits = {str(self.loader.bm25_fig_ids[idx]): float(bm25_raw_scores[idx]) for idx in top_indices if bm25_raw_scores[idx] > 0 and idx < len(self.loader.bm25_fig_ids)}

        fused = self._rrf_fuse([bert_hits, clip_text_hits, clip_img_hits, bm25_hits])
        candidates = []
        seen_fids  = set()
        for fid, rrf_score in fused[:top_k]:
            if fid in seen_fids: continue
            seen_fids.add(fid)
            meta = self._fig_meta_index.get(fid)
            if not meta: continue
            caption = (meta.get('caption_text') or '').strip()
            pmcid   = meta.get('pmcid', '')
            if not pmcid or not caption: continue
            candidates.append({"fid": fid, "pmcid": pmcid, "caption": caption, "rrf_score": rrf_score, "meta": meta})

        telemetry = {
            "captions_evaluated": len(candidates),
            "captions_survived": 0,
            "captions_filtered": 0,
            "caption_filtering_rate": 0.0
        }

        if not candidates: return [], set(), telemetry

        pairs     = [(query, c['caption'][:512]) for c in candidates]
        ce_scores = self.loader.cross_encoder.predict(pairs, batch_size=self.CE_BATCH, show_progress_bar=False, activation_fct=lambda x: x)
        for c, s in zip(candidates, ce_scores): c['ce_score'] = float(s)

        surviving_candidates = [c for c in candidates if c['ce_score'] > self.FIG_CE_THRESHOLD]
        telemetry["captions_survived"] = len(surviving_candidates)
        telemetry["captions_filtered"] = telemetry["captions_evaluated"] - len(surviving_candidates)
        if telemetry["captions_evaluated"] > 0:
            telemetry["caption_filtering_rate"] = round(telemetry["captions_filtered"] / telemetry["captions_evaluated"], 4)

        surviving_candidates.sort(key=lambda x: x['ce_score'], reverse=True)
        selected_figs, boost_pmcids, pmcid_counts = [], set(), {}

        for c in surviving_candidates:
            if len(selected_figs) >= self.FIG_TOP_N: break
            pmcid = c['pmcid']
            if pmcid_counts.get(pmcid, 0) >= 2: continue
            pmcid_counts[pmcid] = pmcid_counts.get(pmcid, 0) + 1
            boost_pmcids.add(pmcid)
            meta = c['meta']
            selected_figs.append({
                "fig_id": c['fid'], "pmcid": pmcid, "fig_label": meta.get('fig_id', ''),
                "caption": c['caption'], "image_path": meta.get('image_path', ''), "has_image": meta.get('has_image', False),
                "rrf_score": round(c['rrf_score'], 6), "ce_score": round(c['ce_score'], 4),
            })
        return selected_figs, boost_pmcids, telemetry

    def retrieve(self, query: str, top_k: int = 20, n_candidates: int = 100) -> dict:
        # Step 1: Figure retrieval alignment sequence
        selected_figs, boost_pmcids, cap_telemetry = self._search_figures(query)

        # Step 2: Dense Text space lookup
        q_emb    = self.loader.bert_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        n_search = min(self._n_cands, self.loader.idx_text.ntotal)
        D, I     = self.loader.idx_text.search(q_emb, n_search)

        faiss_ranks, faiss_scores = {}, {}
        for rank, (raw_dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx == -1: continue
            cid = str(self.loader.text_ids[idx])
            faiss_ranks[cid]  = rank
            faiss_scores[cid] = float(raw_dist)

        # Step 3: Sparse Text space lookup
        bm25_results            = self._bm25_search(query, n_candidates)
        bm25_ranks, bm25_scores = {}, {}
        for rank, (cid, score) in enumerate(bm25_results, 1):
            bm25_ranks[cid]  = rank
            bm25_scores[cid] = score

        # Step 4: Merge primary text distributions via RRF
        all_ids    = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores = {cid: ((self._rrf_score(faiss_ranks[cid]) if cid in faiss_ranks else 0.0) + (self._rrf_score(bm25_ranks[cid]) if cid in bm25_ranks else 0.0)) for cid in all_ids}

        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        candidate_pool = []
        seen_ids       = set()

        for cid in ranked_ids:
            if len(candidate_pool) >= n_candidates: break
            if cid in seen_ids: continue
            seen_ids.add(cid)
            meta = self.loader.chunk_id_to_meta.get(cid, {})
            text = (meta.get('text') or '').strip()
            if not text: continue
            pmcid, pmid = meta.get('pmcid') or None, str(meta.get('pmid')) if meta.get('pmid') else None
            if not pmcid and not pmid: pmcid, pmid = self._parse_chunk_id(cid)
            if pmcid and not pmid: pmid  = self.loader.bridge.get_pmid(pmcid)
            if pmid and not pmcid: pmcid = self.loader.bridge.get_pmcid(pmid)
            domain = meta.get('domain') or self.loader.bridge.get_domain(pmcid, pmid)

            candidate_pool.append({
                "chunk_id": cid, "text": text, "pmcid": pmcid, "pmid": pmid, "sec_id": meta.get('sec_id'),
                "domain": domain, "faiss_score": faiss_scores.get(cid, 0.0), "bm25_score": bm25_scores.get(cid, 0.0),
                "rrf_score": rrf_scores[cid], "fig_boosted": False, "ce_score": 0.0,
            })

        total_ce_invocations = len(candidate_pool)

        # Step 5: Cross-Encoder Primary text scoring loop
        ce_scores_list = self._cross_encode(query, candidate_pool)
        for chunk, ce_s in zip(candidate_pool, ce_scores_list):
            chunk['ce_score'] = ce_s
            chunk['score']    = ce_s  

        # Step 6: Late-Stage Continuous Additive Figure Boosting (Upgraded to V8.1 Math)
        eligible_pool_chunks = [c for c in candidate_pool if c.get('pmcid') and c['pmcid'] in boost_pmcids]
        if eligible_pool_chunks and selected_figs:
            fig_ces = [f['ce_score'] for f in selected_figs]
            max_fig_ce, min_fig_ce = max(fig_ces) if fig_ces else 1.0, min(fig_ces) if fig_ces else 0.0
            if max_fig_ce > min_fig_ce:
                w_visual = {f['fig_id']: (f['ce_score'] - min_fig_ce) / (max_fig_ce - min_fig_ce) for f in selected_figs}
            else:
                w_visual = {f['fig_id']: 1.0 for f in selected_figs}

            chunk_texts = list(set([c['text'].strip() for c in eligible_pool_chunks]))
            fig_captions = list(set([f['caption'].strip() for f in selected_figs]))
            
            text_to_emb = {}
            if chunk_texts:
                c_embs = self.article_encoder.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
                for t, emb in zip(chunk_texts, c_embs): text_to_emb[t] = emb
            if fig_captions:
                f_embs = self.query_encoder.encode(fig_captions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
                for t, emb in zip(fig_captions, f_embs): text_to_emb[t] = emb

            for chunk in eligible_pool_chunks:
                pmcid = chunk['pmcid']
                c_text = chunk['text'].strip()
                e_c = text_to_emb.get(c_text)
                if e_c is None: continue
        
                max_boost_term = 0.0
                for f in selected_figs:
                    if f['pmcid'] == pmcid:
                        f_text = f['caption'].strip()
                        e_f = text_to_emb.get(f_text)
                        if e_f is not None:
                            cos_sim = float(np.dot(e_c, e_f))
                            calibrated_sim = max(0.0, cos_sim - self.SIM_MARGIN_THRESHOLD)
                            term = w_visual[f['fig_id']] * calibrated_sim
                            if term > max_boost_term: max_boost_term = term
        
                boost_value = self.BETA * max_boost_term
                chunk['score']    += boost_value
                chunk['ce_score'] += boost_value
                if max_boost_term > 0.0: chunk['fig_boosted'] = True

        # Step 7: Filter by hard-margin gate & sort
        candidate_pool = [c for c in candidate_pool if c['score'] > self.CE_FILTER_THRESHOLD]
        candidate_pool.sort(key=lambda x: x['score'], reverse=True)  

        # Step 8: Context Diversification Pass (MMR)
        top_chunks = self._mmr(query=query, chunks=candidate_pool, k=self.MMR_K, lam=self.MMR_LAMBDA)
        
        # Step 9: Resolve Paper Node targets
        retrieved_papers, seen_papers = [], set()
        for chunk in top_chunks:
            pmid, pmcid = chunk.get('pmid'), chunk.get('pmcid')
            key = pmid or pmcid
            if key and key not in seen_papers:
                seen_papers.add(key)
                retrieved_papers.append({"pmid": pmid, "pmcid": pmcid, "domain": chunk.get('domain')})

        # Step 10: Downstream Prompt Context Target Truncation (Isolates Top-3 for injection)
        prompt_figs = sorted(selected_figs, key=lambda x: x.get('ce_score', -100.0), reverse=True)[:self.FIG_PROMPT_N]

        n_boosted = sum(1 for c in top_chunks if c.get('fig_boosted'))

        return {
            "figures": prompt_figs, # Transmits truncated captions directly to the downstream prompt generator
            "sections": [], 
            "chunks": top_chunks, 
            "retrieved_papers": retrieved_papers,
            "metadata": {
                "variant": self.VARIANT, "n_candidates": n_search, "pool_size": len(candidate_pool), "returned": len(top_chunks),
                "ce_device": self._ce_device, "n_after_ce_filter": len(candidate_pool), "n_figures": len(selected_figs),
                "n_boost_pmcids": len(boost_pmcids), "n_boosted_chunks": n_boosted,
                
                "ce_invocations": total_ce_invocations, 
                
                "captions_evaluated": cap_telemetry["captions_evaluated"], 
                "captions_filtered": cap_telemetry["captions_filtered"],
                "captions_survived": cap_telemetry["captions_survived"], 
                "caption_filtering_rate": cap_telemetry["caption_filtering_rate"]
            }
        }
