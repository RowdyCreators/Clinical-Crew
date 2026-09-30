import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .pipeline import run_pipeline
from .retriever import HybridRetriever
from .schemas import Chunk, ChatRequest, ChatResponse, FeedbackRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medrag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = await run_in_threadpool(HybridRetriever)  # load models once
    yield


app = FastAPI(title="Medical RAG - Differential Support API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "chunks_indexed": len(request.app.state.retriever.ids)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    result = None
    async for ev in run_pipeline(req, request.app.state.retriever):
        if ev["event"] == "result":
            result = ev["data"]
    if result is None:
        raise HTTPException(500, "Pipeline produced no result")
    # Log metadata only: never the raw symptom text (may be identifying).
    log.info("chat type=%s urgency=%s sources=%s", result.type, result.urgency,
             [s.id for s in result.sources])
    return result


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Server-Sent Events: status updates, retrieved sources, then the final result."""
    async def gen():
        try:
            async for ev in run_pipeline(req, request.app.state.retriever):
                data = ev["data"]
                payload = data.model_dump() if hasattr(data, "model_dump") else data
                yield f"event: {ev['event']}\ndata: {json.dumps(payload)}\n\n"
        except Exception:
            log.exception("stream failed")
            yield 'event: error\ndata: {"detail": "Something went wrong."}\n\n'
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/sources/{chunk_id}", response_model=Chunk)
async def get_source(chunk_id: str, request: Request):
    chunk = await run_in_threadpool(request.app.state.retriever.get_chunk, chunk_id)
    if not chunk:
        raise HTTPException(404, "Source not found")
    return chunk


@app.post("/feedback", status_code=204)
async def feedback(fb: FeedbackRequest):
    config.FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fb.model_dump()) + "\n")