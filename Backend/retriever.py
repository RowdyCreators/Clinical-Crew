"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion, then cross-encoder rerank."""
import re
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .schemas import Chunk

_tok = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _tok.findall(text.lower())


class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = client.get_or_create_collection("medical", metadata={"hnsw:space": "cosine"})
        self._build_bm25()

    def _build_bm25(self):
        data = self.col.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if self.docs else None

    def _to_chunk(self, id_: str, doc: str, meta: dict, score: float = 0.0) -> Chunk:
        return Chunk(id=id_, text=doc, score=score, **meta)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        res = self.col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return self._to_chunk(res["ids"][0], res["documents"][0], res["metadatas"][0])

    def search(self, query: str, sections: Optional[list[str]] = None,
               k: int = config.TOP_K_RETRIEVE, final: int = config.TOP_K_FINAL) -> list[Chunk]:
        if not self.docs:
            return []

        # 1) dense retrieval (with optional metadata filter)
        qvec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        where = {"section": {"$in": sections}} if sections else None
        dense = self.col.query(query_embeddings=qvec, n_results=min(k, len(self.docs)), where=where)
        dense_ids = dense["ids"][0]

        # 2) sparse retrieval
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse_ids = []
        for i in order:
            if scores[i] <= 0:
                break
            if sections and self.metas[i]["section"] not in sections:
                continue
            sparse_ids.append(self.ids[i])
            if len(sparse_ids) >= k:
                break

        # 3) reciprocal rank fusion
        rrf: dict[str, float] = {}
        for ranking in (dense_ids, sparse_ids):
            for rank, id_ in enumerate(ranking):
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (60 + rank)
        candidates = sorted(rrf, key=rrf.get, reverse=True)[:k]
        if not candidates:
            return []

        # 4) cross-encoder rerank
        index = {id_: i for i, id_ in enumerate(self.ids)}
        pairs = [(query, self.docs[index[c]]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)[:final]
        return [
            self._to_chunk(c, self.docs[index[c]], self.metas[index[c]], float(s))
            for c, s in ranked
        ]
