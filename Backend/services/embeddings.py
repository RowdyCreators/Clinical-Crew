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
import math
from pathlib import Path

from Backend.config import CHROMA_DIR, DATA_DIR, EMBEDDING_DIMS, EMBEDDING_MODEL, embedding_api_key

ROOT = Path(__file__).resolve().parents[2]          # project root (Backend/src/..)
CHUNKS_FILE = DATA_DIR / 'chunks' / 'all_chunks.jsonl'
DB_DIR = CHROMA_DIR
COLLECTION = "medical_papers"

MODEL = EMBEDDING_MODEL
DIMS = EMBEDDING_DIMS
BATCH_SIZE = 100      # texts per embed request
MAX_RETRIES = 6

log = logging.getLogger("embedder")
_clients = {}


class EmbeddingError(ValueError):
    """Safe provider failure message; never includes request URLs or credentials."""

    def __init__(self, message, status=None, reason=None):
        super().__init__(message)
        self.status = status
        self.reason = reason


def provider_error(error):
    code = getattr(error, 'code', None)
    payload = getattr(error, 'details', {})
    payload = payload.get('error', payload) if isinstance(payload, dict) else {}
    allowed = {'ACCESS_TOKEN_TYPE_UNSUPPORTED', 'CREDENTIALS_MISSING', 'API_KEY_INVALID',
               'API_KEY_EXPIRED', 'API_KEY_SERVICE_BLOCKED', 'SERVICE_DISABLED',
               'API_KEY_HTTP_REFERRER_BLOCKED', 'API_KEY_IP_ADDRESS_BLOCKED'}
    details = payload.get('details', [])
    reason = next((d.get('reason') for d in details if isinstance(d, dict) and d.get('reason') in allowed), None)
    label = f'HTTP {code}' if code in (400, 401, 403, 404, 429, 500, 503) else 'provider error'
    if reason:
        label += f', {reason}'
    if code in (400, 401, 403) and (code != 400 or reason):
        message = (f'Google rejected the embedding request ({label}). '
                   'Check the Gemini API key and its project/API restrictions in Google AI Studio. '
                   'Save the key in GEMINI_API_KEY in the project .env; new requests reload it automatically.')
    else:
        message = f'Embedding request failed ({label}). Check Gemini model access, quota, and network connectivity, then retry.'
    return EmbeddingError(message, status=code, reason=reason)


def close_clients():
    for client in _clients.values():
        client.close()
    _clients.clear()


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
        "source_sha256": chunk.get('source_sha256'),
        "pipeline_version": chunk.get('pipeline_version'),
        "embedding_model": chunk.get('embedding_model'),
        "embedding_dims": chunk.get('embedding_dims'),
        "content_sha256": chunk.get('content_sha256'),
    }
    return {k: v for k, v in meta.items() if v not in (None, "")}


# --------------------------------------------------------------- embedding ---

def make_client():
    from google import genai
    from google.genai import types
    key = embedding_api_key()
    if not key:
        raise EmbeddingError('Set GEMINI_APIKEY or GEMINI_API_KEY in the backend .env file.')
    # Both standard and newer auth keys use this header-based Developer API client.
    return genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
        base_url='https://generativelanguage.googleapis.com', api_version='v1beta', timeout=60000))


def check_connection():
    """Test the actual embedding method with a tiny probe; store nothing."""
    client = make_client()
    try:
        embed_texts(client, ['title: Connection check | text: Clinical Crew embedding connection check.'], attempts=1)
        return {'status': 'ready', 'model': MODEL, 'dimensions': DIMS}
    finally:
        client.close()


def validate_vectors(vectors, expected):
    if len(vectors) != expected:
        raise ValueError('The embedding provider returned an unexpected number of vectors.')
    for vector in vectors:
        if len(vector) != DIMS or not all(math.isfinite(x) for x in vector):
            raise ValueError('The embedding provider returned an invalid vector.')
        if sum(x * x for x in vector) <= 0:
            raise ValueError('The embedding provider returned an empty vector.')


def embed_texts(client, texts: list[str], attempts=MAX_RETRIES) -> list[list[float]]:
    """Embed a batch of texts, backing off on rate limits / transient errors."""
    from google.genai import types
    config = types.EmbedContentConfig(output_dimensionality=DIMS)
    for attempt in range(attempts):
        try:
            contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
            result = client.models.embed_content(model=MODEL, contents=contents, config=config)
            vectors = [e.values for e in result.embeddings]
            validate_vectors(vectors, len(texts))
            return vectors
        except Exception as e:  # 429 / 5xx -> wait and retry
            code = getattr(e, 'code', None)
            if code in (401, 403):
                raise provider_error(e) from None
            msg = str(e)
            retryable = any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE"))
            if not retryable or attempt == attempts - 1:
                raise provider_error(e) from None
            wait = min(60, 2 ** attempt * 5)
            log.warning('Temporary embedding failure; retrying in %ds', wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def embed_query(client, question: str) -> list[float]:
    """Use this at search time so queries match the document format."""
    return embed_texts(client, [query_text(question)])[0]


# ------------------------------------------------------------------ chroma ---

def get_collection(rebuild: bool = False):
    import chromadb
    path = str(DB_DIR)
    if path not in _clients:
        _clients[path] = chromadb.PersistentClient(path=path)
    db = _clients[path]
    if rebuild:
        try:
            db.delete_collection(COLLECTION)
            log.info("Deleted existing collection %r", COLLECTION)
        except Exception:
            pass
    # cosine distance; we always pass our own vectors, so no embedding_function
    collection = db.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": MODEL, "dims": DIMS},
        embedding_function=None,
    )
    metadata = collection.metadata or {}
    if metadata.get('embedding_model') != MODEL or metadata.get('dims') != DIMS:
        raise ValueError('This Chroma collection uses a different embedding model or dimension. Use a separate DATA_DIR.')
    sample = collection.get(limit=1, include=['embeddings'])
    if sample['ids'] and len(sample['embeddings'][0]) != DIMS:
        raise ValueError('Stored vector dimensions do not match the configured embedding model.')
    return collection


def existing_ids(collection, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), 500):
        found.update(collection.get(ids=ids[i:i + 500], include=[])["ids"])
    return found


def search(collection, client, question: str, k: int = 5, disease: str | None = None,
           where: dict | None = None) -> list[dict]:
    """Top-k chunks, optionally restricted to one disease or by an explicit Chroma `where` filter."""
    res = collection.query(
        query_embeddings=[embed_query(client, question)],
        n_results=k,
        where=where if where is not None else ({"disease": disease} if disease else None),
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
    # Refuse stale IDs instead of silently treating changed text as already embedded.
    if done:
        stored = collection.get(ids=list(done), include=['documents', 'metadatas'])
        originals = {c['chunk_id']: c for c in chunks}
        for chunk_id, text, meta in zip(stored['ids'], stored['documents'], stored['metadatas']):
            chunk = originals[chunk_id]
            if text != chunk['text'] or any(meta.get(k) != v for k, v in chroma_metadata(chunk).items()):
                raise SystemExit('Existing chunk content changed. Review the source and rebuild the affected disease explicitly.')
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
