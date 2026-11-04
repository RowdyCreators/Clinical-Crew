"""Run from the project root: python -m uvicorn Backend.main:app --reload."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from Backend.api.ingestion import router
from Backend.api.embedding import router as embedding_router
from Backend.services.jobs import Jobs
from Backend.services.embeddings import close_clients

STATIC_DIR = Path(__file__).parent / 'static'

@asynccontextmanager
async def lifespan(app):
    app.state.jobs = Jobs()
    yield
    app.state.jobs.close()
    close_clients()

app = FastAPI(title='Clinical Crew XML Ingestion', version='0.2.0', lifespan=lifespan)
app.include_router(router)
app.include_router(embedding_router)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

@app.get('/', include_in_schema=False)
def upload_page():
    return FileResponse(STATIC_DIR / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ok'}
