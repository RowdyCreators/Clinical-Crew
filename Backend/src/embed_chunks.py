"""
Embed chunk files with Gemini Embedding 2 and store them in ChromaDB.

Usage (from project root):
    pip install google-genai chromadb
    set GEMINI_API_KEY=...            (PowerShell: $env:GEMINI_API_KEY="...")

    python Backend/src/embed_chunks.py                   # every chunk in data/chunks/all_chunks.jsonl
    python Backend/src/embed_chunks.py asthma            # only these diseases
    python Backend/src/embed_chunks.py --rebuild         # wipe the collection and re-embed everything
    python Backend/src/embed_chunks.py asthma --rebuild  # re-embed only asthma, keep the rest

Re-running is safe: chunks whose chunk_id is already in the collection are skipped,
so an interrupted run (rate limit, closed laptop) picks up where it stopped.

Gemini Embedding 2 notes:
  * It does NOT accept task_type. Retrieval is asymmetric and signalled by prefixes:
        documents -> "title: {title} | text: {content}"
        queries   -> "task: search result | query: {content}"
    Documents and queries MUST use these formats consistently, so use embed_query()
    below at search time instead of embedding the raw question.
  * Vectors at 768/1536 dims come back already normalized. Keep DIMS identical
    for documents and queries; changing it means --rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # project root (Backend/src/..)
CHUNKS_FILE = ROOT / "data" / "chunks" / "all_chunks.jsonl"
DB_DIR = ROOT / "data" / "chroma"
COLLECTION = "medical_papers"

MODEL = "gemini-embedding-2"
DIMS = 768            # 768 / 1536 / 3072 recommended; 768 is plenty for ~250-word chunks
BATCH_SIZE = 100      # texts per embed request
MAX_RETRIES = 6

log = logging.getLogger("embedder")


# ----------------------------------------------------------- text formatting ---

def document_text(chunk: dict) -> str:
    """Gemini Embedding 2 document format. The section heading is folded into the
    text so 'Treatment' vs 'Symptoms' context survives in the vector."""
    title = (chunk.get("title") or "").strip() or "none"
    heading = (chunk.get("heading") or "").strip()
    body = f"{heading}: {chunk['text']}" if heading else chunk["text"]
    return f"title: {title} | text: {body}"


def query_text(question: str) -> str:
    return f"task: search result | query: {question}"


def chroma_metadata(chunk: dict) -> dict:
    """Chroma metadata only accepts str/int/float/bool, so lists are joined and
    missing values are dropped."""
    meta = {
        "paper_id": chunk.get("paper_id", ""),
        "disease": chunk.get("disease", ""),
        "title": chunk.get("title", ""),
        "authors": ", ".join(chunk.get("authors") or []),
        "year": chunk.get("year"),
        "journal": chunk.get("journal", ""),
        "doi": chunk.get("doi", ""),
        "url": chunk.get("url", ""),
        "license": chunk.get("license", ""),
        "heading": chunk.get("heading", ""),
        "parent": chunk.get("parent", ""),
    }
    return {k: v for k, v in meta.items() if v not in (None, "")}


# --------------------------------------------------------------- embedding ---

def make_client():
    from google import genai
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey).")
    return genai.Client()


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, backing off on rate limits / transient errors."""
    from google.genai import types
    config = types.EmbedContentConfig(output_dimensionality=DIMS)
    for attempt in range(MAX_RETRIES):
        try:
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
            result = client.models.embed_content(model=MODEL, contents=contents, config=config)
            vectors = [e.values for e in result.embeddings]
            if len(vectors) != len(texts):
                raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
            return vectors
        except Exception as e:  # 429 / 5xx -> wait and retry
            msg = str(e)
            retryable = any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE"))
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            wait = min(60, 2 ** attempt * 5)
            log.warning("Embedding call failed (%s); retrying in %ds", msg.splitlines()[0][:120], wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def embed_query(client, question: str) -> list[float]:
    """Use this at search time so queries match the document format."""
    return embed_texts(client, [query_text(question)])[0]


# ------------------------------------------------------------------ chroma ---

def get_collection(rebuild: bool = False):
    import chromadb
    db = chromadb.PersistentClient(path=str(DB_DIR))
    if rebuild:
        try:
            db.delete_collection(COLLECTION)
            log.info("Deleted existing collection %r", COLLECTION)
        except Exception:
            pass
    # cosine distance; we always pass our own vectors, so no embedding_function
    return db.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": MODEL, "dims": DIMS},
        embedding_function=None,
    )


def existing_ids(collection, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), 500):
        found.update(collection.get(ids=ids[i:i + 500], include=[])["ids"])
    return found


def search(collection, client, question: str, k: int = 5, disease: str | None = None) -> list[dict]:
    """Example retrieval: top-k chunks, optionally restricted to one disease."""
    res = collection.query(
        query_embeddings=[embed_query(client, question)],
        n_results=k,
        where={"disease": disease} if disease else None,
    )
    return [{"chunk_id": i, "distance": d, "text": doc, **meta}
            for i, d, doc, meta in zip(res["ids"][0], res["distances"][0],
                                       res["documents"][0], res["metadatas"][0])]


# -------------------------------------------------------------------- main ---

def load_chunks(path: Path, diseases: set[str] | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run chunk_papers.py first.")
    chunks, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if diseases and c.get("disease") not in diseases:
            continue
        if c["chunk_id"] in seen:          # guard against duplicate ids in the file
            continue
        seen.add(c["chunk_id"])
        chunks.append(c)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("diseases", nargs="*", help="only embed these diseases (default: all)")
    ap.add_argument("--chunks-file", type=Path, default=CHUNKS_FILE)
    ap.add_argument("--rebuild", action="store_true", help="delete the collection and re-embed everything")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chunks = load_chunks(args.chunks_file, set(args.diseases) or None)
    if not chunks:
        raise SystemExit("No chunks matched.")

    if args.rebuild and args.diseases:
        # only clear the named diseases; everything else stays embedded
        collection = get_collection()
        collection.delete(where={"disease": {"$in": list(args.diseases)}})
        log.info("Cleared existing chunks for: %s", ", ".join(args.diseases))
    else:
        collection = get_collection(rebuild=args.rebuild)
    done = existing_ids(collection, [c["chunk_id"] for c in chunks])
    todo = [c for c in chunks if c["chunk_id"] not in done]
    log.info("%d chunks total | %d already embedded | %d to embed", len(chunks), len(done), len(todo))
    if not todo:
        return

    client = make_client()
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        vectors = embed_texts(client, [document_text(c) for c in batch])
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],      # raw text, for showing / prompting
            metadatas=[chroma_metadata(c) for c in batch],
        )
        log.info("Embedded %d / %d", min(i + BATCH_SIZE, len(todo)), len(todo))

    log.info("Done. Collection %r now holds %d chunks (%s)", COLLECTION, collection.count(), DB_DIR)


if __name__ == "__main__":
    main()