import numpy as np
import torch
from scipy.sparse import issparse


class HierarchicalRetriever:
    """
    Biomedically adapted hybrid retrieval with text reranking,
    MMR diversification, and parent section context expansion.
    """

    VARIANT             = "hierarchical"
    RRF_K               = 60
    CE_BATCH            = 64       # Cross-encoder prediction batch size
    MMR_LAMBDA          = 0.6      # Relevance weight in MMR (1-λ = diversity penalty)
    MMR_K               = 20       # Final chunk context count
    SEC_MIN_CHUNKS      = 1        # Minimum chunks required to trigger section expansion
    MAX_RAW_SECTION     = 15000
    CHAR_LIMIT          = 2500     # Window length constraint for smart truncation
    CE_FILTER_THRESHOLD = 0.0      # Hard-margin MedCPT logit filter threshold gate

    def __init__(self, loader):
        self.loader = loader

        self._ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._ce_device == "cuda":
            self.loader.cross_encoder.model.to("cuda")

        print(f"   Cross-Encoder device : {self._ce_device}")
        print(f"   FAISS total vectors  : {self.loader.idx_text.ntotal:,}")
        print(f"   Section map entries  : {len(self.loader.section_map):,}")

        if not self.loader.section_map:
            print("   ⚠️  section_map is empty — expansion fallbacks will be active globally")

    # ── Shared Helpers ────────────────────────────────────────

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

    def _cross_encode(self, query: str, chunks: list[dict]) -> list[float]:
        """Score (query, chunk) pairs with un-saturated raw contractive logits."""
        pairs  = [(query, c['text'][:512]) for c in chunks]
        scores = self.loader.cross_encoder.predict(
            pairs,
            batch_size        = self.CE_BATCH,
            show_progress_bar = False,
            activation_fct    = lambda x: x,  # ← BYPASS AUTO-SIGMOID SATURATION
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
        ce_norm        = (
            (ce_scores - ce_min) / (ce_max - ce_min)
            if ce_max > ce_min else np.ones_like(ce_scores)
        )
        selected, remaining = [], list(range(len(chunks)))
        for _ in range(k):
            if not remaining:
                break
            if not selected:
                # Anchors directly to pre-sorted top semantic cross-modal match
                best = remaining[0]
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

    def _smart_truncate(self, section_text: str, anchor_text: str, char_limit: int = None) -> str:
        limit = char_limit or self.CHAR_LIMIT
        if len(section_text) <= limit:
            return section_text

        search_anchor = anchor_text[:80].lower()
        pos           = section_text.lower().find(search_anchor)

        if pos == -1:
            mid   = len(section_text) // 2
            start = max(0, mid - limit // 2)
        else:
            start = max(0, pos - limit // 4)

        window = section_text[start : start + limit]
        first_period = window.find('. ')
        if 0 < first_period < 200:
            window = window[first_period + 2:]

        return window.strip()

    def _expand_to_sections(self, mmr_chunks: list[dict]) -> tuple[list[dict], list[dict]]:
        sec_hits = {}
        for chunk in mmr_chunks:
            pmcid  = chunk.get('pmcid')
            sec_id = chunk.get('sec_id')
            key    = (pmcid, sec_id)

            if key not in sec_hits:
                sec_hits[key] = {
                    "pmcid"      : pmcid,
                    "pmid"       : chunk.get('pmid'),
                    "sec_id"     : sec_id,
                    "domain"     : chunk.get('domain'),
                    "ce_score"   : chunk.get('ce_score', 0.0),
                    "count"      : 0,
                    "chunk_ids"  : [],
                    "best_chunk" : chunk,
                }

            sec_hits[key]["count"] += 1
            sec_hits[key]["chunk_ids"].append(chunk['chunk_id'])

            if chunk.get('ce_score', 0.0) > sec_hits[key]["ce_score"]:
                sec_hits[key]["ce_score"]   = chunk['ce_score']
                sec_hits[key]["best_chunk"] = chunk

        ordered_secs = sorted(sec_hits.values(), key=lambda x: x['ce_score'], reverse=True)
        sections        = []
        fallback_chunks = []

        for sec_info in ordered_secs:
            if sec_info["count"] < self.SEC_MIN_CHUNKS:
                continue

            pmcid  = sec_info["pmcid"]
            sec_id = sec_info["sec_id"]
            full_text = self.loader.section_map.get((pmcid, sec_id), "")

            if not full_text or not full_text.strip():
                for chunk in mmr_chunks:
                    if chunk.get('pmcid') == pmcid and chunk.get('sec_id') == sec_id:
                        fallback_chunks.append(chunk)
                continue

            full_text = full_text.strip()
            if len(full_text) > self.CHAR_LIMIT:
                anchor    = sec_info["best_chunk"].get('text', '')
                full_text = self._smart_truncate(full_text, anchor)

            sections.append({
                "pmcid"     : pmcid,
                "pmid"      : sec_info["pmid"],
                "sec_id"    : sec_id,
                "text"      : full_text,
                "domain"    : sec_info["domain"],
                "ce_score"  : sec_info["ce_score"],
                "num_chunks": sec_info["count"],
            })

        return sections, fallback_chunks

    # ── Main Retrieve Entrypoint ──────────────────────────────

    def retrieve(self, query: str, top_k: int = 20, n_candidates: int = 100) -> dict:
        # ── 1. FAISS dense search ─────────────────────────────
        q_emb    = self.loader.bert_model.encode([query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        n_search = min(n_candidates, self.loader.idx_text.ntotal)
        D, I     = self.loader.idx_text.search(q_emb, n_search)

        faiss_ranks, faiss_scores = {}, {}
        for rank, (raw_dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx == -1: continue
            cid = str(self.loader.text_ids[idx])
            faiss_ranks[cid]  = rank
            faiss_scores[cid] = float(raw_dist)

        # ── 2. BM25 sparse search ─────────────────────────────
        bm25_results            = self._bm25_search(query, n_candidates)
        bm25_ranks, bm25_scores = {}, {}
        for rank, (cid, score) in enumerate(bm25_results, 1):
            bm25_ranks[cid]  = rank
            bm25_scores[cid] = score

        # ── 3. RRF fusion ─────────────────────────────────────
        all_ids    = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores = {cid: ((self._rrf_score(faiss_ranks[cid]) if cid in faiss_ranks else 0.0) + (self._rrf_score(bm25_ranks[cid]) if cid in bm25_ranks else 0.0)) for cid in all_ids}
        ranked_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── 4. Build candidate pool ───────────────────────────
        candidate_pool = []
        seen_ids       = set()

        for cid in ranked_ids:
            if len(candidate_pool) >= n_candidates: break
            if cid in seen_ids: continue
            seen_ids.add(cid)

            meta = self.loader.chunk_id_to_meta.get(cid, {})
            text = (meta.get('text') or "").strip()
            if not text: continue

            pmcid, pmid = meta.get('pmcid') or None, str(meta.get('pmid')) if meta.get('pmid') else None
            if not pmcid and not pmid: pmcid, pmid = self._parse_chunk_id(cid)
            if pmcid and not pmid: pmid  = self.loader.bridge.get_pmid(pmcid)
            if pmid and not pmcid: pmcid = self.loader.bridge.get_pmcid(pmid)
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
                "ce_score"   : 0.0,
                "score"      : 0.0,
            })

        total_ce_invocations = len(candidate_pool)

        # ── 5. Cross-encoder reranking ────────────────────────
        ce_scores_list = self._cross_encode(query, candidate_pool)
        for chunk, ce_s in zip(candidate_pool, ce_scores_list):
            chunk['ce_score'] = ce_s
            chunk['score']    = ce_s

        # Apply hard-margin binary filter constraint gate
        candidate_pool = [c for c in candidate_pool if c['ce_score'] > self.CE_FILTER_THRESHOLD]
        candidate_pool.sort(key=lambda x: x['ce_score'], reverse=True)

        # ── 6. MMR on top-50 CE-ranked chunks ─────────────────
        mmr_chunks = self._mmr(candidate_pool[:50], k=self.MMR_K, lam=self.MMR_LAMBDA)

        # ── 7. Section expansion ──────────────────────────────
        sections, fallback_chunks = self._expand_to_sections(mmr_chunks)

        # ── 8. retrieved_papers from MMR chunks ───────────────
        retrieved_papers, seen_papers = [], set()
        for chunk in mmr_chunks:
            pmid, pmcid = chunk.get('pmid'), chunk.get('pmcid')
            key = pmid or pmcid
            if key and key not in seen_papers:
                seen_papers.add(key)
                retrieved_papers.append({"pmid": pmid, "pmcid": pmcid, "domain": chunk.get('domain')})

        # ── 9. chunks field — fallback if no sections ─────────
        effective_chunks = fallback_chunks if not sections else mmr_chunks

        return {
            "figures"          : [],
            "sections"         : sections,         # ✅ FIXED: Now passes the expanded 2500-char windows!
            "chunks"           : effective_chunks, # ✅ Kept for metric evaluation backend parity
            "retrieved_papers" : retrieved_papers,
            "metadata"         : {
                "variant"          : self.VARIANT,
                "n_candidates"     : n_search,
                "pool_size"        : len(candidate_pool),
                "returned"         : len(mmr_chunks),
                "n_sections"       : len(sections),
                "n_fallback_chunks": len(fallback_chunks),
                "ce_device"        : self._ce_device,
                "ce_invocations"   : total_ce_invocations,
            }
        }
