"""Upload and progress endpoints used by the browser."""
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from Backend.config import MAX_XML_BYTES
from Backend.services.ingestion import IngestionError, disease_name

router = APIRouter(prefix='/api/v1/ingestions', tags=['XML ingestion'])


@router.post('', status_code=202)
async def upload_xml(request: Request, disease: str = Form(...), file: UploadFile = File(...)):
    try:
        disease = disease_name(disease)
    except IngestionError as error:
        raise HTTPException(400, str(error)) from error
    filename = (file.filename or '').replace('\\', '/').split('/')[-1]
    if Path(filename).suffix.lower() != '.xml':
        raise HTTPException(400, 'Choose a PMC/JATS .xml file.')
    try:
        content = await file.read(MAX_XML_BYTES + 1)
    finally:
        await file.close()
    if not content:
        raise HTTPException(400, 'The XML file is empty.')
    if len(content) > MAX_XML_BYTES:
        raise HTTPException(413, 'The XML file exceeds the 10 MB limit.')
    return request.app.state.jobs.submit(content, filename, disease)


@router.get('/{job_id}')
def ingestion_status(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, 'Ingestion job not found.')
    return job
