"""A small credential/model diagnostic; never returns the configured key."""
from fastapi import APIRouter, HTTPException

from Backend.services.embeddings import EmbeddingError, check_connection

router = APIRouter(prefix='/api/v1/embedding', tags=['Embedding connection'])


@router.post('/check')
def embedding_connection():
    try:
        return check_connection()
    except EmbeddingError as error:
        raise HTTPException(503, str(error)) from None
