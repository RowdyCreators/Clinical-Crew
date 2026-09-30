"""Orchestration: gather -> red flags -> normalize -> retrieve -> generate -> confidence check -> ask more or answer.
Implemented as an async generator so the same code backs both /chat and /chat/stream."""
from typing import AsyncIterator

from fastapi.concurrency import run_in_threadpool

from . import config
from .generator import expand_query, generate_differential
from .red_flags import EMERGENCY_MESSAGE, check_red_flags
from .retriever import HybridRetriever
from .schemas import ChatRequest, ChatResponse


def build_patient_context(req: ChatRequest) -> tuple[str, str]:
    """Returns (free text of everything the user said, full structured context for the LLM)."""
    user_msgs = [t.content for t in req.history if t.role == "user"] + [req.message]
    free_text = "\n".join(user_msgs)
    facts = []
    if req.age is not None: facts.append(f"Age: {req.age}")
    if req.sex: facts.append(f"Sex: {req.sex}")
    if req.duration: facts.append(f"Duration: {req.duration}")
    if req.severity: facts.append(f"Severity (1-10): {req.severity}")
    return free_text, "\n".join(facts + [f"Reported symptoms:\n{free_text}"])


def _event(name: str, data) -> dict:
    return {"event": name, "data": data}


async def run_pipeline(req: ChatRequest, retriever: HybridRetriever) -> AsyncIterator[dict]:
    free_text, patient_context = build_patient_context(req)

    # 1) Deterministic red-flag layer: runs first, LLM cannot override it.
    flags = check_red_flags(free_text)
    if flags:
        yield _event("result", ChatResponse(
            type="emergency", urgency="emergency", message=EMERGENCY_MESSAGE,
            red_flags=flags, disclaimer=config.DISCLAIMER))
        return

    # 2) Normalize lay language into clinical terms
    yield _event("status", "Understanding your symptoms...")
    query = await expand_query(free_text)

    # 3) Retrieve: prefer symptom sections, fall back to everything
    yield _event("status", "Searching medical sources...")
    chunks = await run_in_threadpool(retriever.search, query, ["symptoms"])
    if len(chunks) < 3:
        chunks = await run_in_threadpool(retriever.search, query, None)

    turn = sum(1 for t in req.history if t.role == "user") + 1
    can_ask_more = turn <= config.MAX_FOLLOWUP_TURNS

    # 4) Retrieval-confidence gate
    if not chunks or chunks[0].score < config.MIN_RERANK_SCORE:
        if can_ask_more:
            yield _event("result", ChatResponse(
                type="follow_up",
                message="I couldn't find enough information to go on. Could you tell me more?",
                follow_up_questions=["What other symptoms are you experiencing?",
                                     "How long have the symptoms lasted, and how severe are they?"],
                disclaimer=config.DISCLAIMER))
        else:
            yield _event("result", ChatResponse(
                type="insufficient", urgency="see_doctor",
                message="There isn't enough reliable information here to suggest possible conditions. "
                        "Please consult a healthcare professional.",
                disclaimer=config.DISCLAIMER))
        return

    yield _event("sources", [c.model_dump() for c in chunks])

    # 5) Grounded, structured generation
    yield _event("status", "Preparing possible conditions...")
    out = await generate_differential(patient_context, chunks)

    # 6) Ask follow-ups if the model says the differential is under-determined
    if out.needs_more_info and out.follow_up_questions and can_ask_more:
        yield _event("result", ChatResponse(
            type="follow_up", urgency=out.urgency,
            message="A few more details would help narrow this down.",
            follow_up_questions=out.follow_up_questions[:3],
            sources=chunks, disclaimer=config.DISCLAIMER))
        return

    if not out.conditions:
        yield _event("result", ChatResponse(
            type="insufficient", urgency="see_doctor",
            message="I couldn't ground any possible conditions in the sources. "
                    "Please consult a healthcare professional.",
            sources=chunks, disclaimer=config.DISCLAIMER))
        return

    cited = {cid for c in out.conditions for cid in c.citations}
    yield _event("result", ChatResponse(
        type="differential", urgency=out.urgency,
        message="These are possibilities to discuss with a clinician, not a diagnosis.",
        conditions=out.conditions,
        sources=[c for c in chunks if c.id in cited],
        disclaimer=config.DISCLAIMER))
