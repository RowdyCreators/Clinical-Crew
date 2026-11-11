"""Chat endpoint used by the React frontend."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Backend.services import chat
from Backend.services.chat import AnswerError, ChatError, EmbeddingError

router = APIRouter(prefix='/api/v1/chat', tags=['Chat'])


class Turn(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    disease: str | None = Field(default=None, max_length=64)
    top_k: int = Field(default=6, ge=1, le=12)
    # Differential clarification state, echoed back from the previous clarification response.
    candidates: list[str] = Field(default_factory=list, max_length=20)
    clarify_round: int = Field(default=0, ge=0, le=10)
    clarify: bool = True  # False answers right away ("Skip questions")


@router.post('')
def send_message(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, 'Describe your symptoms before sending.')
    try:
        return chat.answer(message, [t.model_dump() for t in body.history], k=body.top_k,
                           disease=body.disease or None, candidates=body.candidates,
                           clarify_round=body.clarify_round, clarify=body.clarify)
    except ChatError as error:
        raise HTTPException(400, str(error)) from None
    except (EmbeddingError, AnswerError) as error:
        raise HTTPException(503, str(error)) from None
    except ValueError as error:  # collection/model mismatch from get_collection
        raise HTTPException(500, str(error)) from None


@router.get('/categories')
def list_categories():
    try:
        return {'categories': chat.categories()}
    except ValueError as error:
        raise HTTPException(500, str(error)) from None
