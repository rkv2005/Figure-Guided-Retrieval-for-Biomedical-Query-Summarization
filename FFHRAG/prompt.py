import re
import warnings

VARIANTS = {
    "vanilla",
    "bm25",
    "rerank",
    "filtering",
    "hierarchical",
    "figureboost",
    "ffhrag",
}

# Per-variant chunk character limits — longer for richer variants
CHUNK_CHAR_LIMITS = {
    "vanilla"      : 600,
    "bm25"         : 600,
    "rerank"       : 600,
    "filtering"    : 700,
    "hierarchical" : 2500,
    "figureboost"  : 1500,
    "ffhrag"       : 1500,
}

# Per-variant how many chunks/sections to include in prompt
N_CHUNKS = {
    "vanilla"      : 10,
    "bm25"         : 10,
    "rerank"       : 8,
    "filtering"    : 8,
    "hierarchical" : 4,   # sections are longer, fewer needed
    "figureboost"  : 8,
    "ffhrag"       : 8,
}

BIOASQ_PROMPT = """\
You are a biomedical expert. Answer the question using the provided evidence only.

# Question
{query}

# Evidence
{figure_context}{text_context}
# Instructions
- Write 3-4 clear sentences (60-80 words)
- Include key mechanisms, genes, or pathways mentioned in the evidence
- Use precise biomedical terminology
- Be comprehensive but concise
- Answer in English only
- Do NOT include citations, references, or passage numbers

Answer:"""


def build_prompt(
    query           : str,
    retrieval_result: dict,
    variant         : str,
) -> str:
    """
    Build a variant-aware prompt for the generator.

    retrieval_result keys:
        figures  : list[dict]  — caption, figure_id  (ffhrag only)
        sections : list[dict]  — text                (hierarchical)
        chunks   : list[dict]  — text                (all others)
    """
    variant = variant.lower().strip()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Must be one of: {VARIANTS}")

    char_limit = CHUNK_CHAR_LIMITS[variant]
    top_n      = N_CHUNKS[variant]

    figure_context = ""
    if variant == "ffhrag":
        fig_source = (
            retrieval_result.get("prompt_figures")
            or retrieval_result.get("figures", [])
        )[:3]
        fig_lines = []
        for i, fig in enumerate(fig_source, 1):
            caption = (fig.get("caption") or "").strip()
            if caption:
                caption = caption[:180] + ("..." if len(caption) > 180 else "")
                fig_lines.append(f"[Figure {i}]: {caption}")
        if fig_lines:
            figure_context = "\n".join(fig_lines) + "\n\n"

    # ── Text context ───────────────────────────────────────────
    text_context = ""
    sections = retrieval_result.get("sections", [])
    chunks   = retrieval_result.get("chunks",   [])

    if sections:
        passages = []
        for sec in sections[:top_n]:
            text = (sec.get("text") or "").strip()
            if text:
                text = text[:char_limit] + ("..." if len(text) > char_limit else "")
                passages.append(text)
        text_context = "\n\n".join(
            f"[Passage {i}]: {p}" for i, p in enumerate(passages, 1)
        ) + "\n\n"

    elif chunks:
        passages = []
        for chunk in chunks[:top_n]:
            text = (chunk.get("text") or "").strip()
            if text:
                text = text[:char_limit] + ("..." if len(text) > char_limit else "")
                passages.append(text)
        text_context = "\n\n".join(
            f"[Passage {i}]: {p}" for i, p in enumerate(passages, 1)
        ) + "\n\n"

    else:
        warnings.warn(
            f"No evidence found for variant='{variant}' — check retrieval pipeline",
            RuntimeWarning
        )
        text_context = "[No relevant evidence retrieved.]\n\n"

    return BIOASQ_PROMPT.format(
        query          = query.strip(),
        figure_context = figure_context,
        text_context   = text_context,
    )


def clean_answer(answer: str) -> str:
    answer = re.sub(r'\[Figure \d+\]',  '', answer)
    answer = re.sub(r'\[Fig\.? \d+\]',  '', answer)
    answer = re.sub(r'\[Passage \d+\]', '', answer)
    answer = re.sub(r'\[\d+\]',         '', answer)
    answer = re.sub(r'^(Answer:|Your Answer:)\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(
        r'^Based on (the )?(provided|retrieved) evidence[,:\s]+',
        '', answer, flags=re.IGNORECASE
    )
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()


def truncate_to_target(answer: str, target: int = 75, max_words: int = 100) -> str:
    words = answer.split()
    if len(words) <= target:
        return answer
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    result, count = [], 0
    for sent in sentences:
        sw = len(sent.split())
        if count + sw <= max_words:
            result.append(sent)
            count += sw
        else:
            break
    return (' '.join(result) if result else ' '.join(words[:max_words]) + '.')
