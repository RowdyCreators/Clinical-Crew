"""Small local job tracker: JSON status files and one ingestion worker."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from threading import Lock

from Backend.config import DATA_DIR
from Backend.services.ingestion import IngestionError, ingest_xml, write_atomic
from Backend.services.embeddings import EmbeddingError


class Jobs:
    def __init__(self, data_dir=DATA_DIR):
        self.root = Path(data_dir)
        self.lock = Lock()
        self.folder = self.root / 'jobs'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='xml-ingestion')
        for path in self.folder.glob('*.json'):
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['status'] not in ('completed', 'failed'):
                self.update(job, status='failed', error='Processing was interrupted. Upload the XML again to retry.')

    def update(self, job, **fields):
        with self.lock:
            job.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
            write_atomic(self.folder / f"{job['id']}.json", json.dumps(job, ensure_ascii=False).encode())

    def submit(self, content, filename, disease):
        job = {'id': uuid4().hex, 'filename': filename, 'disease': disease, 'status': 'queued',
               'total_chunks': 0, 'completed_chunks': 0, 'error': None, 'result': None,
               'created_at': datetime.now(timezone.utc).isoformat()}
        self.update(job)
        self.executor.submit(self.run, job, content)
        return self.get(job['id'])

    def get(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            return None
        path = self.folder / f'{job_id}.json'
        with self.lock:
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def run(self, job, content):
        try:
            result = ingest_xml(content, job['filename'], job['disease'],
                                progress=lambda **fields: self.update(job, **fields), data_dir=self.root)
            self.update(job, status='completed', result=result, completed_chunks=result['chunks'])
        except (IngestionError, EmbeddingError) as error:
            self.update(job, status='failed', error=str(error))
        except Exception as error:
            logging.getLogger(__name__).warning('Ingestion job %s failed during %s (%s).', job['id'], job['status'], type(error).__name__)
            self.update(job, status='failed', error='Processing failed. Check backend API-key configuration, provider access/quota, and storage, then retry.')

    def close(self):
        self.executor.shutdown(wait=True)
