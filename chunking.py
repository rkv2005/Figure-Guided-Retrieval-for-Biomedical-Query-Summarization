from pathlib import Path
import xml.etree.ElementTree as ET
import json

NXML_ROOT = Path("/content/drive/MyDrive/ffhrag_store/nxml")
OUT_ROOT  = Path("/content/drive/MyDrive/ffhrag_store/text")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SECTIONS_PATH = OUT_ROOT / "sections.jsonl"
CHUNKS_PATH   = OUT_ROOT / "chunks.jsonl"

MAX_CHARS_PER_CHUNK = 1500   # tune later if needed


def iter_nxml_files(root: Path):
    for p in root.glob("PMC*.nxml"):
        if p.is_file():
            yield p


def clean_text(t: str) -> str:
    if not t:
        return ""
    return " ".join(t.split())


def extract_sections_from_nxml(nxml_path: Path):
    """
    Returns list of sections:
    each = {
        'sec_id', 'parent_sec_id', 'sec_path', 'title', 'text'
    }
    """
    tree = ET.parse(nxml_path)
    root = tree.getroot()

    # JATS article body sections under <body> and sometimes <front>/<back>
    # We’ll handle <body> main sections only for now.
    body = root.find(".//body")
    if body is None:
        return []

    sections = []

    def walk_secs(elem, parent_path="", parent_sec_id=None, idx=1):
        # Only treat <sec> as sections
        for s in elem.findall("./sec"):
            sec_id = s.get("id") or f"s{len(sections)+1}"
            title_el = s.find("./title")
            title = clean_text("".join(title_el.itertext())) if title_el is not None else ""
            this_path = (parent_path + "/" + title) if parent_path and title else (title or parent_path or "BODY")

            # collect all paragraph-like text under this <sec>
            texts = []
            for p in s.findall(".//p"):
                t = clean_text("".join(p.itertext()))
                if t:
                    texts.append(t)
            full_txt = "\n\n".join(texts)

            sections.append({
                "sec_id": sec_id,
                "parent_sec_id": parent_sec_id,
                "sec_path": this_path,
                "title": title,
                "text": full_txt,
            })

            # recurse into nested sec
            walk_secs(s, parent_path=this_path, parent_sec_id=sec_id)

    walk_secs(body)
    return sections


def chunk_section(pmcid: str, sec):
    """
    sec: one element from extract_sections_from_nxml
    yields chunk dicts
    """
    text = sec["text"]
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    c_idx = 0

    while start < n:
        end = min(start + MAX_CHARS_PER_CHUNK, n)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            break

        chunk_id = f"{pmcid}__{sec['sec_id']}__c{c_idx:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "pmcid": pmcid,
            "sec_id": sec["sec_id"],
            "sec_path": sec["sec_path"],
            "text": chunk_text,
            "char_start": int(start),
            "char_end": int(end),
        })

        c_idx += 1
        start = end

    return chunks


def build_corpus():
    # open output files once and append line by line
    with SECTIONS_PATH.open("w", encoding="utf-8") as f_sec, \
         CHUNKS_PATH.open("w", encoding="utf-8") as f_chnk:

        for nxml_path in iter_nxml_files(NXML_ROOT):
            pmcid = nxml_path.stem   # e.g., PMC7618588
            sections = extract_sections_from_nxml(nxml_path)

            for sec in sections:
                sec_rec = {
                    "pmcid": pmcid,
                    "sec_id": sec["sec_id"],
                    "parent_sec_id": sec["parent_sec_id"],
                    "sec_path": sec["sec_path"],
                    "title": sec["title"],
                    "text": sec["text"],
                }
                f_sec.write(json.dumps(sec_rec, ensure_ascii=False) + "\n")

                chunks = chunk_section(pmcid, sec)
                for ch in chunks:
                    f_chnk.write(json.dumps(ch, ensure_ascii=False) + "\n")

            print("Processed", pmcid, "sections:", len(sections))

# Run once
if __name__ == '__main__':
    build_corpus()
