# Beyond Text: Quantifying the Impact of Figure-Guided Retrieval in Biomedical RAG Systems

This repository contains the official implementation of **Figure-First Hybrid RAG (FFHRAG)**, a multi-stage retrieval architecture designed to leverage visual evidence (figures, charts, and diagrams) to guide textual evidence selection in the biomedical domain. 

The architecture implements a decoupled discovery phase that identifies relevant visual evidence to generate a cross-modal guiding signal. This approach provides a lightweight, noise-free alternative to traditional embedding fusion, effectively mitigating the "alignment noise" prevalent in joint multimodal RAG frameworks.

## 📂 Repository Structure

### 🧠 Embeddings
Scripts responsible for generating dense and sparse representations of the knowledge corpus.
* **`bm25.py`**: Implements sparse indexing for exact lexical matches using the BM25+ variant to prioritize low-frequency clinical terms.
* **`clip_embeddings.py`**: Generates multimodal image and caption representations using the **BiomedCLIP** vision-language model.
* **`faiss.py`**: Handles the construction of **IVFFlat** indices, enabling sub-linear search time over the 1.1 million chunk corpus.
* **`pubmedbert_embeddings.py`**: Encodes text chunks and captions into 768-dimensional vectors using domain-adapted **S-PubMedBERT**.

### 🧪 Variants
Implementations for the six-variant progressive ablation study used to quantify the impact of each architectural component.
* **`vanilla.py`**: Baseline single-stage dense retrieval using PubMedBERT.
* **`bm25.py`**: Hybrid retrieval combining dense semantic and sparse lexical signals via Reciprocal Rank Fusion (RRF).
* **`ce_filter.py`**: Integrates Cross-Encoder re-ranking and MMR diversification for high-precision text-only retrieval.
* **`hierarchical.py`**: Implements section-level expansion (up to 2,500 characters) anchored on retrieved chunks.
* **`figboost.py`**: Introduces the **Figureboost** mechanism, applying a multiplicative score multiplier ($\alpha=1.2$) to text chunks associated with relevant figures.
* **`ffhrag.py`**: The complete pipeline, combining figure-guided retrieval with prompt-level caption injection for grounded generation.

### 🛠️ Core Utilities
* **`chunking.py`**: Logic for parsing PMC NXML archives into fixed-length 1,500-character chunks while maintaining hierarchical metadata.
* **`evaluator.py`**: Comprehensive evaluation suite computing retrieval (MRR, R-Precision, HitRate@K) and generation (ROUGE, BLEU, BERTScore) metrics.
* **`loader.py`**: Handles memory-mapped loading of large-scale FAISS indices and metadata to minimize local RAM overhead.
* **`prompt.py`**: A variant-aware template builder that assembles the visual context block, text evidence, and system instructions for the LLM generator.

### 📊 Data & Evaluation
* **`phase1_eval.json`**: Results of the zero-shot generator benchmarking study across six candidate LLMs (Llama, Gemini, Zephyr, etc.).
* **`phase2_eval.json`**: The curated test set of 290 BioASQ-13b summary-type questions, stratified into **Fig-Yes** and **Fig-No** subsets.
* **`pmid_domain_labels.csv`**: Metadata mapping the corpus of 30,000 papers across 22 standard biomedical domains (e.g., Genetics, Oncology). Kindly download the papers from the PMC OA subset if you want to replicate the corpus. 
Link: https://pmc.ncbi.nlm.nih.gov/tools/ftp/

---

## 📝 Status
**Current Status:** Under review at *Frontiers in Artificial Intelligence — Natural Language Processing*.

