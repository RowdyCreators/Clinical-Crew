"""
Chunk cleaned PubMed paper JSON files for a RAG pipeline.

Usage (from project root):
    python Backend/src/chunk_papers.py                       # every disease folder
    python Backend/src/chunk_papers.py lyme_disease asthma   # just these diseases
    python Backend/src/chunk_papers.py --processed-dir data/processed --output-dir data/chunks

Input : data/processed/<disease>/*.json, one JSON per paper (schema: paper_id,
        title, authors, year, journal, doi, pmcid, url, license, abstract,
        sections[{heading, parent, text}])
Output: data/chunks/<disease>/<paper_id>.json          (chunks for that paper)
        data/chunks/<disease>/<disease>_chunks.jsonl   (every chunk for that disease)
        data/chunks/all_chunks.jsonl                   (every chunk, all diseases)

Each chunk's "disease" field is the folder name (e.g. "lyme_disease"), so the
vector store can filter by disease. all_chunks.jsonl is rebuilt from every
disease's .jsonl at the end, so chunking one disease never drops the others.
"""

import argparse
import json
import logging
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# Chunk size settings (measured in WORDS; ~1.3-1.5 tokens/word for biomedical text)
# ----------------------------------------------------------------------------
TARGET_WORDS = 180    # aim for ~180 words  (~250-270 tokens)
MAX_WORDS = 250       # hard ceiling        (~350-400 tokens, safe under 512)
MIN_WORDS = 40        # don't emit a chunk smaller than this if it can be avoided
OVERLAP_WORDS = 30    # carry up to ~1-2 trailing sentences into the next chunk
MIN_KEEP_WORDS = 5    # discard fragments shorter than this (e.g. stray "Figure 1")

# Sections that are not useful for answering medical questions
SKIP_HEADING_RE = re.compile(
    r"^\s*(\d+[\.\)]?\s*)?"
    r"(references?|bibliography|literature cited|acknowledge?ments?|funding|"
    r"conflicts? of interest|competing interests?|author contributions?|"
    r"data availability|supplementary|abbreviations|ethics statement)\b",
    re.IGNORECASE,
)

# Words ending in "." that do NOT end a sentence
ABBREVIATIONS = {
    "al", "fig", "figs", "e.g", "i.e", "vs", "approx", "ca", "no", "nos", "dr",
    "mr", "ms", "prof", "inc", "ltd", "cf", "eq", "ref", "refs", "resp", "spp",
    "sp", "st", "vol", "pp", "ed", "eds", "min", "max", "conc", "sec",
}

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"'])")

log = logging.getLogger("chunker")


# ----------------------------------------------------------------------------
# Sentence splitting + chunking
# ----------------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that is aware of common biomedical abbreviations."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    pieces = SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            last_token = sentences[-1].split()[-1].rstrip(".").lower()
            # previous piece ended with an abbreviation -> glue back together
            if sentences[-1].endswith(".") and last_token in ABBREVIATIONS:
                sentences[-1] += " " + piece
                continue
        sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]


def hard_split(sentence: str, size: int) -> list[str]:
    """Fallback for a single sentence longer than MAX_WORDS: split on word boundaries."""
    words = sentence.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def chunk_text(text: str) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_words = 0
    new_in_cur = 0  # sentences in `cur` that are not just overlap

    def flush():
        nonlocal cur, cur_words, new_in_cur
        chunks.append(cur)
        # build overlap from the tail of the chunk just emitted
        overlap: list[str] = []
        ow = 0
        for s in reversed(cur):
            w = word_count(s)
            if ow + w > OVERLAP_WORDS:
                break
            overlap.insert(0, s)
            ow += w
        cur, cur_words, new_in_cur = overlap, ow, 0

    for sent in sentences:
        w = word_count(sent)

        if w > MAX_WORDS:  # pathological long "sentence"
            if new_in_cur:
                flush()
            for part in hard_split(sent, TARGET_WORDS):
                chunks.append([part])
            cur, cur_words, new_in_cur = [], 0, 0
            continue

        too_big = cur_words + w > MAX_WORDS
        past_target = cur_words + w > TARGET_WORDS and cur_words >= MIN_WORDS
        if cur and new_in_cur and (too_big or past_target):
            flush()

        cur.append(sent)
        cur_words += w
        new_in_cur += 1

    if new_in_cur:
        tail_new = cur[len(cur) - new_in_cur:]
        tail_words = sum(word_count(s) for s in tail_new)
        prev_words = sum(word_count(s) for s in chunks[-1]) if chunks else 0
        # tiny trailing runt -> append to previous chunk instead of its own chunk
        if chunks and tail_words < MIN_WORDS and prev_words + tail_words <= MAX_WORDS + MIN_WORDS:
            chunks[-1].extend(tail_new)
        else:
            chunks.append(cur)

    return [" ".join(c).strip() for c in chunks if c]


# ----------------------------------------------------------------------------
# Paper -> chunks
# ----------------------------------------------------------------------------
def should_skip(section: dict) -> bool:
    heading = section.get("heading") or ""
    parent = section.get("parent") or ""
    return bool(SKIP_HEADING_RE.match(heading) or SKIP_HEADING_RE.match(parent))


def build_chunks(paper: dict, disease: str | None = None) -> list[dict]:
    paper_id = paper["paper_id"]
    meta = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "journal": paper.get("journal", ""),
        "doi": paper.get("doi", ""),
        "url": paper.get("url", ""),
        "license": paper.get("license", ""),
        "disease": disease or paper.get("disease", ""),
    }

    # (heading, parent, text) units in reading order
    units: list[tuple[str, str, str]] = []
    if (paper.get("abstract") or "").strip():
        units.append(("Abstract", "", paper["abstract"]))
    for sec in paper.get("sections", []):
        if should_skip(sec):
            continue
        units.append((sec.get("heading", ""), sec.get("parent", ""), sec.get("text", "")))

    chunks: list[dict] = []
    for heading, parent, text in units:
        for piece in chunk_text(text):
            if word_count(piece) < MIN_KEEP_WORDS:
                continue
            chunks.append({
                "chunk_id": f"{paper_id}_c{len(chunks) + 1}",
                "text": piece,
                **meta,
                "heading": heading,
                "parent": parent,
            })
    return chunks


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def chunk_disease(input_dir: Path, output_dir: Path, disease: str) -> tuple[int, int, list[str]]:
    """Chunk every paper JSON in input_dir into output_dir.
    Returns (papers seen, chunks written, paper_ids that produced no chunks)."""
    files = sorted(input_dir.glob("*.json"))
    if not files:
        log.warning("[skip] %s: no JSON files in %s", disease, input_dir)
        return 0, 0, []

    output_dir.mkdir(parents=True, exist_ok=True)
    total, empty_papers = 0, []
    with open(output_dir / f"{disease}_chunks.jsonl", "w", encoding="utf-8") as combined:
        for f in files:
            try:
                paper = json.loads(f.read_text(encoding="utf-8"))
                chunks = build_chunks(paper, disease=disease)
            except Exception as e:  # keep going on a bad file
                log.error("Failed on %s/%s: %s", disease, f.name, e)
                continue

            if not chunks:
                empty_papers.append(paper.get("paper_id", f.stem))
                log.warning("No chunks produced for %s/%s", disease, f.name)
                continue

            assert all(c["text"].strip() for c in chunks), f"empty chunk in {f.name}"
            out = output_dir / f"{paper['paper_id']}.json"
            out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            for c in chunks:
                combined.write(json.dumps(c, ensure_ascii=False) + "\n")
            total += len(chunks)

    log.info("%s: papers %d | chunks %d | papers with no chunks %d",
             disease, len(files), total, len(empty_papers))
    return len(files), total, empty_papers


def rebuild_combined(output_root: Path) -> int:
    """Concatenate every <disease>/<disease>_chunks.jsonl into all_chunks.jsonl."""
    count = 0
    with open(output_root / "all_chunks.jsonl", "w", encoding="utf-8") as out:
        for jsonl in sorted(output_root.glob("*/*_chunks.jsonl")):
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    count += 1
    return count


def main() -> None:
    root = Path(__file__).resolve().parents[2]  # project root (Backend/src/..)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("diseases", nargs="*",
                    help="disease folder names to chunk (default: every folder in --processed-dir)")
    ap.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    ap.add_argument("--output-dir", type=Path, default=root / "data" / "chunks")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.processed_dir.is_dir():
        raise SystemExit(f"Processed folder not found: {args.processed_dir}")
    folders = sorted(p for p in args.processed_dir.iterdir() if p.is_dir())
    if args.diseases:
        names = {f.name for f in folders}
        for missing in sorted(set(args.diseases) - names):
            log.warning("[skip] %s: no such folder in %s", missing, args.processed_dir)
        folders = [f for f in folders if f.name in set(args.diseases)]
    if not folders:
        raise SystemExit(f"No disease folders to chunk in {args.processed_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    papers = chunks = 0
    all_empty: list[str] = []
    for folder in folders:
        p, c, empty = chunk_disease(folder, args.output_dir / folder.name, folder.name)
        papers, chunks, all_empty = papers + p, chunks + c, all_empty + empty

    combined = rebuild_combined(args.output_dir)
    log.info("This run: %d disease(s) | papers %d | chunks %d",
             len(folders), papers, chunks)
    log.info("all_chunks.jsonl now holds %d chunks across all diseases", combined)
    if all_empty:
        log.warning("Papers with zero chunks: %s", all_empty)


if __name__ == "__main__":
    main()