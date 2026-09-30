"""
Symptom differential-support RAG backend (simple version).

Setup (Python 3.12 recommended):
    pip install fastapi uvicorn chromadb groq pydantic python-dotenv
    set GROQ_API_KEY=your_key        (Windows cmd; use `export` on macOS/Linux)
    uvicorn main:app --reload

Flow: red-flag check -> retrieve from Chroma -> grounded LLM call -> structured JSON.
"""

import json
import os
from typing import Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = 5

DISCLAIMER = (
    "This tool provides educational differential support only. "
    "It is not a medical diagnosis. Consult a licensed clinician."
)

# ---------------------------------------------------------------------------
# App + clients
# ---------------------------------------------------------------------------
app = FastAPI(title="Symptom RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("medical_docs")  # uses Chroma's default embedder
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Free-text symptom description")
    age: int | None = Field(None, ge=0, le=120)
    sex: Literal["male", "female", "other"] | None = None
    duration: str | None = Field(None, description="e.g. '3 days'")


class Condition(BaseModel):
    name: str
    likelihood: Literal["high", "medium", "low"]
    supporting_symptoms: list[str]
    citations: list[str]  # chunk IDs


class ChatResponse(BaseModel):
    urgency: Literal["emergency", "see_doctor", "self_care", "unknown"]
    conditions: list[Condition]
    missing_information: list[str]
    red_flag_triggered: bool
    disclaimer: str = DISCLAIMER


class Doc(BaseModel):
    id: str
    text: str
    condition: str
    section: str = "symptoms"
    source: str = ""
    url: str = ""


# ---------------------------------------------------------------------------
# Deterministic red-flag layer (runs BEFORE the LLM)
# ---------------------------------------------------------------------------
RED_FLAGS = [
    "chest pain", "crushing chest", "can't breathe", "cannot breathe",
    "difficulty breathing", "face drooping", "slurred speech", "sudden weakness",
    "worst headache", "coughing blood", "severe bleeding", "unconscious",
    "suicid", "kill myself", "overdose", "anaphylaxis", "throat closing",
]


def check_red_flags(text: str) -> bool:
    t = text.lower()
    return any(flag in t for flag in RED_FLAGS)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    if collection.count() == 0:
        return []
    res = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        where={"section": "symptoms"},  # metadata filter: symptom chunks only
    )
    return [
        {"id": i, "text": d, "meta": m}
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


SYSTEM_PROMPT = """You are a clinical decision-SUPPORT assistant, not a doctor.
Use ONLY the provided context chunks. If the context is insufficient, return an empty
"conditions" list and explain what is missing in "missing_information".
Never state a definitive diagnosis. Every condition must cite chunk IDs.
Respond with ONLY valid JSON matching this shape:
{
  "urgency": "emergency" | "see_doctor" | "self_care" | "unknown",
  "conditions": [
    {"name": str, "likelihood": "high"|"medium"|"low",
     "supporting_symptoms": [str], "citations": [chunk_id]}
  ],
  "missing_information": [str]
}"""


def generate(req: ChatRequest, chunks: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{c['id']}] ({c['meta'].get('condition')}) {c['text']}" for c in chunks
    )
    patient = (
        f"Symptoms: {req.symptoms}\nAge: {req.age}\nSex: {req.sex}\nDuration: {req.duration}"
    )
    completion = llm.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nPATIENT:\n{patient}"},
        ],
    )
    return json.loads(completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": collection.count()}


@app.post("/ingest")
def ingest(docs: list[Doc]):
    """Add pre-chunked documents (one chunk per condition section)."""
    collection.upsert(
        ids=[d.id for d in docs],
        documents=[d.text for d in docs],
        metadatas=[
            {"condition": d.condition, "section": d.section, "source": d.source, "url": d.url}
            for d in docs
        ],
    )
    return {"ingested": len(docs), "total": collection.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Red-flag short-circuit
    if check_red_flags(req.symptoms):
        return ChatResponse(
            urgency="emergency",
            conditions=[],
            missing_information=[],
            red_flag_triggered=True,
        )

    # 2. Retrieve
    chunks = retrieve(req.symptoms)
    if not chunks:
        return ChatResponse(
            urgency="unknown",
            conditions=[],
            missing_information=["No knowledge base content found. Ingest documents first."],
            red_flag_triggered=False,
        )

    # 3. Generate grounded, structured output
    try:
        data = generate(req, chunks)
        return ChatResponse(**data, red_flag_triggered=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")


@app.get("/sources/{chunk_id}")
def get_source(chunk_id: str):
    res = collection.get(ids=[chunk_id])
    if not res["ids"]:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"id": chunk_id, "text": res["documents"][0], "metadata": res["metadatas"][0]}