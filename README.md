Beyond Text: Quantifying the Impact of Figure-Guided Retrieval in Biomedical RAG Systems
This repository contains the official implementation of Figure-First Hybrid RAG (FFHRAG), a multi-stage retrieval architecture designed to leverage visual evidence (figures and charts) to guide text selection in the biomedical domain. Unlike traditional systems, FFHRAG employs a decoupled approach that avoids joint embedding fusion, effectively mitigating alignment noise.
+2

📂 Repository Structure
🧠 Embeddings

Scripts responsible for generating dense and sparse representations of the corpus.

bm25.py: Implements sparse indexing for exact lexical matches using the BM25+ variant.

clip_embeddings.py: Generates multimodal image and caption representations using the BiomedCLIP vision-language model.
+1

faiss.py: Handles the construction and inner-product search of IVFFlat indices for sub-linear retrieval time.

pubmedbert_embeddings.py: Encodes text chunks and captions into 768-dimensional vectors using a domain-adapted PubMedBERT model.

🧪 Variants

The core implementations for the six-variant ablation study used to quantify the impact of figure-guided retrieval.
+2

vanilla.py: Baseline single-stage dense retrieval.

bm25.py: Hybrid retrieval combining dense and sparse (TF-IDF) lexical signals via Reciprocal Rank Fusion (RRF).
+1

ce_filter.py: Integrates Cross-Encoder re-ranking and MMR diversification for high-precision text-only retrieval.

hierarchical.py: Implements section-level expansion anchored on relevant chunks.

figboost.py: Introduces the Figureboost mechanism, applying a multiplicative score multiplier to text chunks associated with relevant figures.

ffhrag.py: The complete pipeline, combining figure-guided retrieval with prompt-level caption injection.
+1

🛠️ Core Utilities

chunking.py: Logic for parsing NXML full-text archives into fixed-length 1,500-character chunks.
+1

evaluator.py: The evaluation suite for computing retrieval (MRR, R-Precision) and generation (ROUGE, BERTScore) metrics.
+2

loader.py: Handles memory-mapped loading of large-scale embedding arrays and metadata.

prompt.py: A variant-aware template builder that assembles visual context, text evidence, and system instructions for the LLM generator.
+1

📊 Data

phase1_eval.json: Benchmark results for the zero-shot generator selection study across six candidate LLMs.
+1

phase2_eval.json: The curated test set of 290 BioASQ summary-type questions used for end-to-end pipeline evaluation.

pmid_domain_labels.csv: Metadata mapping 29,532 PMC documents across 22 standard biomedical domains.
+1

📝 Status
Current Status: Under review at Frontiers in Artificial Intelligence — Natural Language Processing.
