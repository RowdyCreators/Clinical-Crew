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
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App + clients
# ---------------------------------------------------------------------------
app = FastAPI(title="Symptom RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
