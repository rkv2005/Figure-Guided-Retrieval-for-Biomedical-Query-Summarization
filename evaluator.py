import json, csv, re, os
import numpy as np
from rouge_score import rouge_scorer
from bert_score  import score as bert_score
from sentence_transformers import SentenceTransformer, util

PRECISION_K        = 5
RECALL_K           = 10

BASE = "/kaggle/input/ffhrag-store"


class AblationEvaluator:

    def __init__(
        self,
        eval_json_path : str,
        sbert_model    : SentenceTransformer,
        bridge_csvs    : list[str] = None,
    ):
        with open(eval_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.questions     = {q['id']: q for q in data['questions']}
        self.sbert         = sbert_model
        self.rouge         = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL'], use_stemmer=True
        )

        # ── Build PMCID↔PMID bridge ───────────────────────────
        self.pmcid_to_pmid   = {}
        self.pmid_to_pmcid   = {}
        self.pmcid_to_domain = {}
        self.pmid_to_domain  = {}

        csvs = bridge_csvs or [
            os.path.join(BASE, "Metadata/tgz_available_master.csv"),
            os.path.join(BASE, "Metadata/pmids_pmcid_only_no_tgz.csv"),
        ]
        for path in csvs:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    pc  = str(row.get('pmcid')        or '').strip()
                    pm  = str(row.get('pmid')         or '').strip()
                    dom = str(row.get('domain_final') or '').strip()
                    if pc and pm:
                        self.pmcid_to_pmid[pc] = pm
                        self.pmid_to_pmcid[pm] = pc
                    if pc and dom:
                        self.pmcid_to_domain[pc] = dom
                    if pm and dom:
                        self.pmid_to_domain[pm] = dom

        print(f"✅ AblationEvaluator ready — {len(self.questions)} questions loaded")
        print(f"   Bridge Map: {len(self.pmcid_to_pmid):,} elements indexed")

    # ============================================================
    # BRIDGE HELPERS
    # ============================================================

    def _canonical(self, pmcid=None, pmid=None):
        pmcid = str(pmcid or '').strip() or None
        pmid  = str(pmid  or '').strip() or None
        if pmcid and not pmid:
            pmid  = self.pmcid_to_pmid.get(pmcid)
        if pmid  and not pmcid:
            pmcid = self.pmid_to_pmcid.get(pmid)
        return pmcid, pmid

    def _parse_chunk_id(self, chunk_id):
        s = str(chunk_id)
        if s.startswith('PMC'):
            m = re.match(r'(PMC\d+)', s)
            return self._canonical(pmcid=m.group(1)) if m else (None, None)
        else:
            m = re.match(r'(?:PMID)?(\d{4,10})(?:__|$)', s)
            return self._canonical(pmid=m.group(1)) if m else (None, None)

    def _resolve_gold_paper(self, g):
        return self._canonical(pmcid=g.get('pmcid'), pmid=g.get('pmid'))

    def _get_domain(self, pmcid=None, pmid=None):
        if pmcid:
            d = self.pmcid_to_domain.get(pmcid)
            if d: return d
        if pmid:
            d = self.pmid_to_domain.get(pmid)
            if d: return d
        return None

    # ============================================================
    # UNIFIED NDCG@K METRIC ENGINE
    # ============================================================

    def _compute_ndcg(self, q: dict, ret_result: dict, k: int) -> float:
        """Computes binary NDCG@K for text chunk ranking sequence."""
        gold_papers = q.get('gold_papers', [])
        chunks      = ret_result.get('chunks', [])
        if not gold_papers or not chunks:
            return 0.0

        gold_canonical = [self._resolve_gold_paper(g) for g in gold_papers]
        gold_pmids  = {pm for _, pm in gold_canonical if pm}
        gold_pmcids = {pc for pc, _ in gold_canonical if pc}
        total_unique_gold = len(gold_pmids | gold_pmcids)

        # Build binary relevance mapping array
        relevance = []
        for chunk in chunks[:k]:
            cid = chunk.get('chunk_id')
            pc, pm = self._parse_chunk_id(cid)
            
            # Match against explicit dictionary attributes or structural identifiers
            if (pm and pm in gold_pmids) or (pc and pc in gold_pmcids):
                relevance.append(1.0)
            elif (chunk.get('pmid') and str(chunk.get('pmid')) in gold_pmids) or \
                 (chunk.get('pmcid') and str(chunk.get('pmcid')) in gold_pmcids):
                relevance.append(1.0)
            else:
                relevance.append(0.0)

        # Calculate Discounted Cumulative Gain (DCG)
        dcg = 0.0
        for rank_idx, rel in enumerate(relevance, 1):
            if rel > 0.0:
                dcg += rel / np.log2(rank_idx + 1)

        # Calculate Ideal DCG (IDCG)
        idcg = 0.0
        for rank_idx in range(1, min(total_unique_gold, k) + 1):
            idcg += 1.0 / np.log2(rank_idx + 1)

        return float(dcg / idcg) if idcg > 0.0 else 0.0

    # ============================================================
    # PUBLIC ENTRY PARSERS
    # ============================================================
    def evaluate_variant(self, variant: str, results: list[dict]) -> dict:
        retrieval_scores  = []
        ndcg_5_scores     = []
        ndcg_10_scores    = []
        domain_scores     = []
        generated_answers = []
        gold_answers      = []

        latency_ret_list  = []
        latency_gen_list  = []
        latency_tot_list  = []
        peak_vram_list    = []
        ce_calls_list     = []

        cap_eval_list     = []
        cap_filt_list     = []
        cap_surv_list     = []
        cap_rate_list     = []

        fig_scores_true  = []   
        fig_scores_false = []   

        for r in results:
            qid = r['question_id']
            q   = self.questions.get(qid)
            if not q: continue

            ret_result = r['retrieval_result']
            gen_answer = r.get('generated_answer', '')

            retrieval_scores.append(self._retrieval_metrics(q, ret_result))
            ndcg_5_scores.append(self._compute_ndcg(q, ret_result, k=5))
            ndcg_10_scores.append(self._compute_ndcg(q, ret_result, k=10))

            dom = self._domain_metrics(q, ret_result)
            if dom is not None: domain_scores.append(dom)

            generated_answers.append(gen_answer)
            gold_answers.append(q['ideal_answer'])

            fig = self._figure_metrics(q, ret_result)
            if q.get('needs_figure') is True:
                fig_scores_true.append(fig)
            else:
                fig_scores_false.append(fig)

            # FIXED: Avoid destructive short-circuiting. Merge all source telemetry blocks.
            combined_telemetry = {}
            if isinstance(ret_result, dict) and 'metadata' in ret_result:
                combined_telemetry.update(ret_result['metadata'])
            if 'metrics' in r and isinstance(r['metrics'], dict):
                combined_telemetry.update(r['metrics'])
            if 'telemetry' in r and isinstance(r['telemetry'], dict):
                combined_telemetry.update(r['telemetry'])

            if combined_telemetry:
                if 'latency_retrieval_ms' in combined_telemetry: latency_ret_list.append(combined_telemetry['latency_retrieval_ms'])
                if 'latency_generation_ms' in combined_telemetry: latency_gen_list.append(combined_telemetry['latency_generation_ms'])
                if 'latency_total_ms' in combined_telemetry: latency_tot_list.append(combined_telemetry['latency_total_ms'])
                if 'peak_gpu_vram_mb' in combined_telemetry: peak_vram_list.append(combined_telemetry['peak_gpu_vram_mb'])
                if 'peak_local_vram_mb' in combined_telemetry: peak_vram_list.append(combined_telemetry['peak_local_vram_mb'])
                if 'ce_invocations' in combined_telemetry: ce_calls_list.append(combined_telemetry['ce_invocations'])
                
                if 'captions_evaluated' in combined_telemetry: cap_eval_list.append(combined_telemetry['captions_evaluated'])
                if 'captions_filtered' in combined_telemetry: cap_filt_list.append(combined_telemetry['captions_filtered'])
                if 'captions_survived' in combined_telemetry: cap_surv_list.append(combined_telemetry['captions_survived'])
                if 'caption_filtering_rate' in combined_telemetry: cap_rate_list.append(combined_telemetry['caption_filtering_rate'])

        bert_p, bert_r, bert_f1 = bert_score(
            generated_answers, gold_answers,
            lang       = 'en',
            model_type = 'distilbert-base-uncased',
            verbose    = False,
            batch_size = 32,
        )

        gen_embs   = self.sbert.encode(generated_answers, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        gold_embs  = self.sbert.encode(gold_answers,      convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        sbert_sims = util.cos_sim(gen_embs, gold_embs).diagonal().cpu().numpy()

        rouge1_scores, rougeL_scores = [], []
        for gen, gold in zip(generated_answers, gold_answers):
            sc = self.rouge.score(gold, gen)
            rouge1_scores.append(sc['rouge1'].fmeasure)
            rougeL_scores.append(sc['rougeL'].fmeasure)

        def mean(lst): return float(np.mean(lst)) if lst else 0.0
        def median(lst): return float(np.median(lst)) if lst else 0.0
        def p95(lst): return float(np.percentile(lst, 95)) if lst else 0.0

        return {
            "variant"          : variant,
            "n_questions"      : len(results),
            "precision_at_5"   : mean([s['precision_at_5'] for s in retrieval_scores]),
            "recall_at_10"     : mean([s['recall_at_10']    for s in retrieval_scores]),
            "mrr"              : mean([s['mrr']             for s in retrieval_scores]),
            "r_precision"      : mean([s['r_precision']     for s in retrieval_scores]),
            "ndcg_at_5"        : mean(ndcg_5_scores),
            "ndcg_at_10"       : mean(ndcg_10_scores),
            "domain_precision" : mean([s['precision'] for s in domain_scores]),
            "domain_recall"    : mean([s['recall']    for s in domain_scores]),
            "domain_f1"        : mean([s['f1']        for s in domain_scores]),
            "rouge1"           : mean(rouge1_scores),
            "rougeL"           : mean(rougeL_scores),
            "bertscore_f1"     : float(bert_f1.mean()),
            "sbert_cosine"     : float(sbert_sims.mean()),

            "computational_telemetry": {
                "latency_retrieval_ms_median" : median(latency_ret_list),
                "latency_retrieval_ms_p95"    : p95(latency_ret_list),
                "latency_generation_ms_median": median(latency_gen_list),
                "latency_generation_ms_p95"   : p95(latency_gen_list),
                "latency_total_ms_median"     : median(latency_tot_list),
                "latency_total_ms_p95"        : p95(latency_tot_list),
                "peak_gpu_vram_mb_max"        : float(np.max(peak_vram_list)) if peak_vram_list else 0.0,
                "avg_cross_encoder_invocations"   : mean(ce_calls_list),
                
                "macro_captions_evaluated"    : mean(cap_eval_list),
                "macro_captions_filtered"     : mean(cap_filt_list),
                "macro_captions_survived"     : mean(cap_surv_list),
                "macro_caption_filtering_rate": mean(cap_rate_list)
            },

            "figure_metrics"   : {
                "needs_fig_true": {
                    "n"              : len(fig_scores_true),
                    "fig_precision"  : mean([s['fig_precision']   for s in fig_scores_true]),
                    "fig_recall"     : mean([s['fig_recall']      for s in fig_scores_true]),
                    "fig_f1"         : mean([s['fig_f1']          for s in fig_scores_true]), 
                    "fig_hit_rate"   : mean([s['fig_hit_rate']    for s in fig_scores_true]),
                    "fig_cap_sim"    : mean([s['fig_cap_sim']     for s in fig_scores_true if s['fig_cap_sim'] is not None]),
                },
                "needs_fig_false": {
                    "n"                  : len(fig_scores_false),
                    "fig_false_pos_rate" : mean([s['fig_hit_rate'] for s in fig_scores_false]),
                    "specificity"        : 1.0 - mean([s['fig_hit_rate'] for s in fig_scores_false]),
                },
            },
        }
    
    # ============================================================
    # INDIVIDUAL RETRIEVED HIT MATCHERS
    # ============================================================

    def _retrieval_metrics(self, q: dict, ret_result: dict) -> dict:
        gold_papers      = q.get('gold_papers', [])
        retrieved_papers = ret_result.get('retrieved_papers', [])

        gold_canonical = [self._resolve_gold_paper(g) for g in gold_papers]
        gold_pmids     = {pm for _, pm in gold_canonical if pm}
        gold_pmcids    = {pc for pc, _ in gold_canonical if pc}
        R = len(gold_papers)

        def ret_paper_is_gold(p):
            pc, pm = self._canonical(pmcid=p.get('pmcid'), pmid=p.get('pmid'))
            if pm and pm in gold_pmids:  return True
            if pc and pc in gold_pmcids: return True
            return False

        hits_ordered = [ret_paper_is_gold(p) for p in retrieved_papers]

        top5   = hits_ordered[:PRECISION_K]
        p_at5  = sum(top5) / PRECISION_K if retrieved_papers else 0.0
        top10  = hits_ordered[:RECALL_K]
        r_at10 = sum(top10) / R if R else 0.0

        mrr = 0.0
        for rank, hit in enumerate(hits_ordered, 1):
            if hit:
                mrr = 1.0 / rank
                break

        top_r  = hits_ordered[:R] if R else []
        r_prec = sum(top_r) / R   if R else 0.0

        return {
            "precision_at_5": p_at5,
            "recall_at_10"  : r_at10,
            "mrr"           : mrr,
            "r_precision"   : r_prec,
        }

    def _figure_metrics(self, q: dict, ret_result: dict) -> dict:
        gold_figures  = q.get('gold_figures', [])
        ret_figures   = ret_result.get('figures', [])
        needs_figure  = q.get('needs_figure', False)

        gold_fig_pmcids = {str(gf.get('pmcid', '')).strip() for gf in gold_figures if gf.get('pmcid')}
        ret_fig_pmcids  = {str(f.get('pmcid', '')).strip() for f in ret_figures if f.get('pmcid')}

        if needs_figure:
            fig_hit = 1.0 if (ret_fig_pmcids & gold_fig_pmcids) else 0.0
        else:
            fig_hit = 1.0 if ret_fig_pmcids else 0.0

        if needs_figure and gold_fig_pmcids and ret_fig_pmcids:
            tp = len(ret_fig_pmcids & gold_fig_pmcids)
            p  = tp / len(ret_fig_pmcids)
            r  = tp / len(gold_fig_pmcids)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        elif needs_figure and gold_fig_pmcids and not ret_fig_pmcids:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            p, r, f1 = 0.0, 0.0, 0.0

        fig_cap_sim = None
        if needs_figure and ret_figures:
            captions = [f.get('caption', '')[:300] for f in ret_figures if f.get('caption', '').strip()]
            if captions:
                query = q.get('body', '')
                q_emb = self.sbert.encode([query], convert_to_tensor=True, show_progress_bar=False)
                cap_embs = self.sbert.encode(captions, convert_to_tensor=True, show_progress_bar=False)
                sims = util.cos_sim(q_emb, cap_embs)[0].cpu().numpy()
                fig_cap_sim = float(np.mean(sims))

        return {
            "fig_precision" : p,
            "fig_recall"    : r,
            "fig_f1"        : f1,
            "fig_hit_rate"  : fig_hit,
            "fig_cap_sim"   : fig_cap_sim,
        }

    def _domain_metrics(self, q: dict, ret_result: dict) -> dict | None:
        gold_domains = {
            str(gp.get('domain', '') or '').strip()
            for gp in q.get('gold_papers', [])
            if gp.get('domain') and str(gp.get('domain', '')).lower() not in ('unmapped', 'null', '')
        }
        if not gold_domains:
            return None

        ret_domains = set()
        for p in ret_result.get('retrieved_papers', []):
            domain = p.get('domain') or self._get_domain(pmcid=p.get('pmcid'), pmid=p.get('pmid'))
            if domain and str(domain).lower() not in ('unmapped', 'null', ''):
                ret_domains.add(str(domain).strip())

        if not ret_domains:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(ret_domains & gold_domains)
        p  = tp / len(ret_domains)
        r  = tp / len(gold_domains)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}        pmid  = str(pmid  or '').strip() or None
        if pmcid and not pmid:
            pmid  = self.pmcid_to_pmid.get(pmcid)
        if pmid  and not pmcid:
            pmcid = self.pmid_to_pmcid.get(pmid)
        return pmcid, pmid

    def _parse_chunk_id(self, chunk_id):
        s = str(chunk_id)
        if s.startswith('PMC'):
            m = re.match(r'(PMC\d+)', s)
            return self._canonical(pmcid=m.group(1)) if m else (None, None)
        else:
            m = re.match(r'(?:PMID)?(\d{4,10})(?:__|$)', s)
            return self._canonical(pmid=m.group(1)) if m else (None, None)

    def _resolve_gold_paper(self, g):
        return self._canonical(pmcid=g.get('pmcid'), pmid=g.get('pmid'))

    def _get_domain(self, pmcid=None, pmid=None):
        if pmcid:
            d = self.pmcid_to_domain.get(pmcid)
            if d: return d
        if pmid:
            d = self.pmid_to_domain.get(pmid)
            if d: return d
        return None

    # ============================================================
    # PUBLIC — main entry point
    # ============================================================

    def evaluate_variant(self, variant: str, results: list[dict]) -> dict:
        retrieval_scores  = []
        passage_scores    = []
        domain_scores     = []
        generated_answers = []
        gold_answers      = []

        # Figure metric accumulators — split by needs_figure
        fig_scores_true  = []   # needs_figure=True  (84 q)
        fig_scores_false = []   # needs_figure=False (206 q)

        for r in results:
            qid = r['question_id']
            q   = self.questions.get(qid)
            if not q:
                continue

            ret_result = r['retrieval_result']
            gen_answer = r.get('generated_answer', '')

            retrieval_scores.append(self._retrieval_metrics(q, ret_result))
            passage_scores.append(self._hit_rate(q, ret_result))

            dom = self._domain_metrics(q, ret_result)
            if dom is not None:
                domain_scores.append(dom)

            generated_answers.append(gen_answer)
            gold_answers.append(q['ideal_answer'])

            # Figure metrics — computed for ALL questions, split later
            fig = self._figure_metrics(q, ret_result)
            if q.get('needs_figure') is True:
                fig_scores_true.append(fig)
            else:
                fig_scores_false.append(fig)

        # ── BERTScore batch ───────────────────────────────────
        bert_p, bert_r, bert_f1 = bert_score(
            generated_answers, gold_answers,
            lang       = 'en',
            model_type = 'distilbert-base-uncased',
            verbose    = False,
            batch_size = 32,
        )

        # ── SBERT generation cosine batch ─────────────────────
        gen_embs   = self.sbert.encode(generated_answers, convert_to_tensor=True,
                                        show_progress_bar=False, batch_size=32)
        gold_embs  = self.sbert.encode(gold_answers,      convert_to_tensor=True,
                                        show_progress_bar=False, batch_size=32)
        sbert_sims = util.cos_sim(gen_embs, gold_embs).diagonal().cpu().numpy()

        # ── ROUGE per question ────────────────────────────────
        rouge1_scores, rougeL_scores = [], []
        for gen, gold in zip(generated_answers, gold_answers):
            sc = self.rouge.score(gold, gen)
            rouge1_scores.append(sc['rouge1'].fmeasure)
            rougeL_scores.append(sc['rougeL'].fmeasure)

        def mean(lst): return float(np.mean(lst)) if lst else 0.0

        return {
            "variant"          : variant,
            "n_questions"      : len(results),
            "precision_at_5"   : mean([s['precision_at_5'] for s in retrieval_scores]),
            "recall_at_10"     : mean([s['recall_at_10']    for s in retrieval_scores]),
            "mrr"              : mean([s['mrr']             for s in retrieval_scores]),
            "r_precision"      : mean([s['r_precision']     for s in retrieval_scores]),
            "hit_rate_at_k"    : mean(passage_scores),
            "domain_precision" : mean([s['precision'] for s in domain_scores]),
            "domain_recall"    : mean([s['recall']    for s in domain_scores]),
            "domain_f1"        : mean([s['f1']        for s in domain_scores]),
            "domain_n_scored"  : len(domain_scores),
            "rouge1"           : mean(rouge1_scores),
            "rougeL"           : mean(rougeL_scores),
            "bertscore_f1"     : float(bert_f1.mean()),
            "sbert_cosine"     : float(sbert_sims.mean()),

            # ── Figure metrics — split by needs_figure ────────
            "figure_metrics"   : {
                # needs_figure=True subset (84 q) — precision/recall/f1
                "needs_fig_true": {
                    "n"              : len(fig_scores_true),
                    "fig_precision"  : mean([s['fig_precision']   for s in fig_scores_true]),
                    "fig_recall"     : mean([s['fig_recall']      for s in fig_scores_true]),
                    "fig_f1"         : mean([s['fig_f1']          for s in fig_scores_true]),
                    "fig_hit_rate"   : mean([s['fig_hit_rate']    for s in fig_scores_true]),
                    "fig_cap_sim"    : mean([s['fig_cap_sim']     for s in fig_scores_true
                                            if s['fig_cap_sim'] is not None]),
                },
                # needs_figure=False subset (206 q) — specificity only
                # fig_precision/recall meaningless here, only report:
                # fig_hit_rate = % of questions where ≥1 figure was retrieved (false positive rate)
                "needs_fig_false": {
                    "n"                  : len(fig_scores_false),
                    "fig_false_pos_rate" : mean([s['fig_hit_rate'] for s in fig_scores_false]),
                    "specificity"        : 1.0 - mean([s['fig_hit_rate'] for s in fig_scores_false]),
                },
            },
        }

    # ============================================================
    # FIGURE METRICS
    # ============================================================

    def _figure_metrics(self, q: dict, ret_result: dict) -> dict:
        """
        Computes per-question figure metrics.

        For needs_figure=True:
          - fig_precision : retrieved_fig_pmcids ∩ gold_fig_pmcids / retrieved_fig_pmcids
          - fig_recall    : retrieved_fig_pmcids ∩ gold_fig_pmcids / gold_fig_pmcids
          - fig_f1        : harmonic mean
          - fig_hit_rate  : 1 if any retrieved figure matches a gold PMCID
          - fig_cap_sim   : avg SBERT cosine(query, retrieved caption)

        For needs_figure=False:
          - fig_precision/recall/f1 → 0.0  (no gold figures exist)
          - fig_hit_rate → 1 if retriever returned ANY figure (false positive)
          - fig_cap_sim  → None
        """
        query         = q.get('body', '')
        gold_figures  = q.get('gold_figures', [])
        ret_figures   = ret_result.get('figures', [])
        needs_figure  = q.get('needs_figure', False)

        # Gold PMCIDs from gold_figures field (stage6 construction)
        gold_fig_pmcids = {
            str(gf.get('pmcid', '')).strip()
            for gf in gold_figures
            if gf.get('pmcid')
        }

        # Retrieved figure PMCIDs
        ret_fig_pmcids = {
            str(f.get('pmcid', '')).strip()
            for f in ret_figures
            if f.get('pmcid')
        }

        # fig_hit_rate — any retrieved fig from a gold PMCID (True qs)
        #              — any retrieved fig at all           (False qs, = false positive)
        if needs_figure:
            fig_hit = 1.0 if (ret_fig_pmcids & gold_fig_pmcids) else 0.0
        else:
            fig_hit = 1.0 if ret_fig_pmcids else 0.0

        # Precision / Recall / F1
        if needs_figure and gold_fig_pmcids and ret_fig_pmcids:
            tp  = len(ret_fig_pmcids & gold_fig_pmcids)
            p   = tp / len(ret_fig_pmcids)
            r   = tp / len(gold_fig_pmcids)
            f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        elif needs_figure and gold_fig_pmcids and not ret_fig_pmcids:
            # Gold exists but nothing retrieved
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            # needs_figure=False — no gold, metrics undefined → 0
            p, r, f1 = 0.0, 0.0, 0.0

        # Caption similarity — SBERT cosine(query, retrieved captions)
        # Only meaningful for needs_figure=True with actual retrieved figs
        fig_cap_sim = None
        if needs_figure and ret_figures:
            captions = [
                f.get('caption', '')[:300]
                for f in ret_figures
                if f.get('caption', '').strip()
            ]
            if captions:
                q_emb   = self.sbert.encode(
                    [query], convert_to_tensor=True,
                    show_progress_bar=False
                )
                cap_embs = self.sbert.encode(
                    captions, convert_to_tensor=True,
                    show_progress_bar=False
                )
                sims        = util.cos_sim(q_emb, cap_embs)[0].cpu().numpy()
                fig_cap_sim = float(np.mean(sims))

        return {
            "fig_precision" : p,
            "fig_recall"    : r,
            "fig_f1"        : f1,
            "fig_hit_rate"  : fig_hit,
            "fig_cap_sim"   : fig_cap_sim,
        }

    # ============================================================
    # RETRIEVAL METRICS — bridge-aware (unchanged)
    # ============================================================

    def _retrieval_metrics(self, q: dict, ret_result: dict) -> dict:
        gold_papers      = q.get('gold_papers', [])
        retrieved_papers = ret_result.get('retrieved_papers', [])

        gold_canonical = [self._resolve_gold_paper(g) for g in gold_papers]
        gold_pmids     = {pm for _, pm in gold_canonical if pm}
        gold_pmcids    = {pc for pc, _ in gold_canonical if pc}
        R = len(gold_papers)

        def ret_paper_is_gold(p):
            pc, pm = self._canonical(pmcid=p.get('pmcid'), pmid=p.get('pmid'))
            if pm and pm in gold_pmids:  return True
            if pc and pc in gold_pmcids: return True
            return False

        hits_ordered = [ret_paper_is_gold(p) for p in retrieved_papers]

        top5   = hits_ordered[:PRECISION_K]
        p_at5  = sum(top5) / PRECISION_K if retrieved_papers else 0.0
        top10  = hits_ordered[:RECALL_K]
        r_at10 = sum(top10) / R if R else 0.0

        mrr = 0.0
        for rank, hit in enumerate(hits_ordered, 1):
            if hit:
                mrr = 1.0 / rank
                break

        top_r  = hits_ordered[:R] if R else []
        r_prec = sum(top_r) / R   if R else 0.0

        return {
            "precision_at_5": p_at5,
            "recall_at_10"  : r_at10,
            "mrr"           : mrr,
            "r_precision"   : r_prec,
        }

    # ============================================================
    # HIT RATE — SBERT cosine, chunk vs gold snippet (unchanged)
    # ============================================================

    def _hit_rate(self, q: dict, ret_result: dict) -> float:
        snippets = q.get('bioasq_snippets', [])
        chunks   = ret_result.get('chunks', [])
        if not snippets or not chunks:
            return 0.0

        snippet_texts = [s['text'] for s in snippets if s.get('text')]
        chunk_texts   = [c['text'] for c in chunks   if c.get('text')]
        if not snippet_texts or not chunk_texts:
            return 0.0

        snip_embs  = self.sbert.encode(snippet_texts, convert_to_tensor=True,
                                        show_progress_bar=False)
        chunk_embs = self.sbert.encode(chunk_texts,   convert_to_tensor=True,
                                        show_progress_bar=False)
        max_sim = float(util.cos_sim(chunk_embs, snip_embs).max())
        return 1.0 if max_sim >= self.hit_threshold else 0.0

    # ============================================================
    # DOMAIN METRICS (unchanged)
    # ============================================================

    def _domain_metrics(self, q: dict, ret_result: dict) -> dict | None:
        gold_domains = {
            str(gp.get('domain', '') or '').strip()
            for gp in q.get('gold_papers', [])
            if gp.get('domain') and
               str(gp.get('domain', '')).lower() not in ('unmapped', 'null', '')
        }
        if not gold_domains:
            return None

        ret_domains = set()
        for p in ret_result.get('retrieved_papers', []):
            domain = p.get('domain') or self._get_domain(
                pmcid=p.get('pmcid'), pmid=p.get('pmid')
            )
            if domain and str(domain).lower() not in ('unmapped', 'null', ''):
                ret_domains.add(str(domain).strip())

        if not ret_domains:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = len(ret_domains & gold_domains)
        p  = tp / len(ret_domains)
        r  = tp / len(gold_domains)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1}
