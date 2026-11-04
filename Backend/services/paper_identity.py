"""Paper identity and duplicate checks for ingestion."""
import hashlib
import re

from Backend.services import embeddings
from Backend.services.ingestion_errors import IngestionError


def stable_paper_id(paper, source_hash):
    identity = paper.get('pmcid') or paper.get('doi') or source_hash
    stable_id = paper.get('pmcid') or ('paper_' + hashlib.sha256(identity.encode()).hexdigest()[:24])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stable_id):
        raise IngestionError('The article contains an invalid paper identifier.')
    return stable_id


def find_existing(collection, paper, stable_id, disease):
    filters = [{'paper_id': stable_id}]
    if paper.get('url'):
        filters.append({'url': paper['url']})
    if paper.get('doi'):
        filters.append({'doi': paper['doi']})
    existing = collection.get(where={'$or': filters} if len(filters) > 1 else filters[0],
                              include=['documents', 'metadatas'])
    if not existing['ids']:
        return existing, stable_id
    paper_ids = {meta['paper_id'] for meta in existing['metadatas']}
    if len(paper_ids) != 1:
        raise IngestionError('Multiple indexed versions exist for this paper. Review the library before uploading.')
    stable_id = next(iter(paper_ids))  # Preserve existing legacy paper IDs.
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stable_id):
        raise IngestionError('An existing paper identifier is unsafe. Review the library first.')
    if any(meta.get('disease') != disease for meta in existing['metadatas']):
        raise IngestionError('This paper is already indexed under another category. No records were changed.')
    return existing, stable_id


def is_same_paper(existing, chunks):
    previous = {i: (text, meta) for i, text, meta in zip(
        existing['ids'], existing['documents'], existing['metadatas'])}
    if len(previous) != len(chunks):
        return False
    for chunk in chunks:
        old = previous.get(chunk['chunk_id'])
        if not old or old[0] != chunk['text'] or any(
                old[1].get(key) != value for key, value in embeddings.chroma_metadata(chunk).items()):
            return False
    return True


def reject_existing_changes(existing, chunks):
    if not is_same_paper(existing, chunks):
        raise IngestionError('This paper is already indexed with different content or metadata. No data was replaced; review the existing version first.')


def reject_chunk_id_conflicts(collection, ids):
    if collection.get(ids=ids, include=[])['ids']:
        raise IngestionError('Chunk IDs conflict with existing records. No data was changed.')
