import json
import os
import csv
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Paths ───────────────────────────────────────────────────
BASE_PATH   = "/kaggle/input/ffhrag-store"
EVAL_PATH   = "/kaggle/input/datasets/raghavkishorev/stage-2-eval/stage6_final_eval.json"
OUTPUT_PATH = "/kaggle/working/sensitivity_results.json"
SBERT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# ── Sweep Configuration Matrix ──────────────────────────────
SWEEPS = {
    "MMR_LAMBDA": [0.4, 0.5, 0.6, 0.7, 0.8],
    "FIG_BOOST" : [1.0, 1.5, 2.0, 2.5, 3.0],
    "RRF_K"     : [20,  40,  60,  80],
    "THRESHOLD_MARGIN": [0.7, 0.75, 0.8, 0.85, 0.9]  # Added Tau margin sweep
}

DEFAULTS = {
    "MMR_LAMBDA"      : 0.6,
    "FIG_BOOST"       : 1.2,  # Equation Beta
    "RRF_K"           : 60,
    "THRESHOLD_MARGIN": 0.8,  # Baseline equation Tau margin floor
    "MMR_K"           : 20,
    "N_CANDS"         : 100,
    "MMR_POOL"        : 50,
    "FIG_TOP_K"       : 20,
    "FIG_TOP_N"       : 5,
    "CE_BATCH"        : 64,
}

# ============================================================
# SINGLE PARAMETER SWEEP EXECUTION WORKER
# ============================================================

def run_sweep(param_name, param_value, questions, loader, evaluator):
    # Override only the targeted active parameter parameter row
    cfg = {**DEFAULTS, param_name: param_value}

    retriever = FFHRAGRetriever(
        loader,
        mmr_lambda       = cfg["MMR_LAMBDA"],
        fig_boost        = cfg["FIG_BOOST"],
        rrf_k            = cfg["RRF_K"],
        threshold_margin = cfg["THRESHOLD_MARGIN"],  # Injected into retriever model
        mmr_k            = cfg["MMR_K"],
        n_cands          = cfg["N_CANDS"],
        fig_top_k        = cfg["FIG_TOP_K"],
        fig_top_n        = cfg["FIG_TOP_N"],
        ce_batch         = cfg["CE_BATCH"],
    )

    scores = []
    for q in tqdm(questions, desc=f"🔍 Sweeping {param_name}={param_value}", leave=False):
        try:
            result = retriever.retrieve(q.get('body', ''))
            scores.append(evaluator._retrieval_metrics(q, result))
        except Exception as e:
            tqdm.write(f"   ❌ Execution fault on QID [{q.get('id', 'UNKNOWN')}]: {e}")

    def mean(lst): 
        return float(np.mean(lst)) if lst else 0.0
        
    return {
        "param"          : param_name,
        "value"          : param_value,
        "n"              : len(scores),
        "precision_at_5" : mean([s.get('precision_at_5', 0.0) for s in scores]),
        "recall_at_10"   : mean([s.get('recall_at_10', 0.0) for s in scores]),
        "mrr"            : mean([s.get('mrr', 0.0) for s in scores]),
        "r_precision"    : mean([s.get('r_precision', 0.0) for s in scores]),
        "ndcg_at_5"      : mean([s.get('ndcg_at_5', 0.0) for s in scores]),    # Added IR Compliance tracking
        "ndcg_at_10"     : mean([s.get('ndcg_at_10', 0.0) for s in scores]),  # Added IR Compliance tracking
    }

# ============================================================
# MAIN MATRIX COORDINATOR ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("📂 Ingesting validation ground truth records...")
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = data['questions']
    print(f"   Total verification anchors: {len(questions)} queries.")

    print("🚀 Mounting production disk file loaders...")
    loader = FFHRetrieverLoader(base_path=BASE_PATH)

    print("🔤 Spawning context embedding scoring layers...")
    sbert_model = SentenceTransformer(SBERT_MODEL, device='cuda')
    evaluator = AblationEvaluator(EVAL_PATH, sbert_model)

    all_results = {}

    # Run the sweeps sequentially
    for param_name, values in SWEEPS.items():
        print(f"\n{'='*75}")
        print(f"🚀 RUNNING PARAMETER SENSITIVITY SWEEP: {param_name}")
        print(f"   Held Defaults: { {k:v for k,v in DEFAULTS.items() if k != param_name} }")
        print(f"{'='*75}")

        sweep_results = []
        for val in values:
            result = run_sweep(param_name, val, questions, loader, evaluator)
            sweep_results.append(result)

        all_results[param_name] = sweep_results

        # Print detailed local dashboard column view
        print(f"\n📊 Sweep Performance Matrix [{param_name}]:")
        print(f"  {'Value':<12} | {'P@5':^8} | {'R@10':^8} | {'MRR':^8} | {'NDCG@5':^8} | {'NDCG@10':^8}")
        print(f"  {'─'*71}")
        for r in sweep_results:
            marker = " ◀ default option" if r['value'] == DEFAULTS[param_name] else ""
            print(f"  {str(r['value']):<12} | "
                  f"{r['precision_at_5']:^8.4f} | "
                  f"{r['recall_at_10']:^8.4f} | "
                  f"{r['mrr']:^8.4f} | "
                  f"{r['ndcg_at_5']:^8.4f} | "
                  f"{r['ndcg_at_10']:^8.4f}"
                  f"{marker}")

    # ── Exporting Run Matrices ────────────────────────────────
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Execution matrix written to output layer → {OUTPUT_PATH}")

    # ── Final Macro Cross-Sweep Summary Panel ─────────────────
    print(f"\n{'='*75}")
    print(f"🏆 SENSITIVITY OPTIMIZATION SUMMARY (ANCHOR METRIC: MRR)")
    print(f"{'='*75}")
    for param_name, sweep_results in all_results.items():
        best        = max(sweep_results, key=lambda x: x['mrr'])
        default_val = DEFAULTS[param_name]
        default_r   = next(r for r in sweep_results if r['value'] == default_val)
        delta       = best['mrr'] - default_r['mrr']
        
        print(f"  {param_name:<18}: Best={str(best['value']):<5} (MRR={best['mrr']:.4f}) | "
              f"Default={str(default_val):<5} (MRR={default_r['mrr']:.4f}) | "
              f"Δ={delta:+.4f}")
    print(f"{'='*75}")
    print("✅ Sensitivity sweep execution cleanly decoupled.")
