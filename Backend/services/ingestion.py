"""One XML file -> parsed paper -> chunks -> validated vectors -> Chroma."""
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError

from Backend.config import DATA_DIR, MAX_CHUNKS, MAX_XML_BYTES, PIPELINE_VERSION
from Backend.services.parser import Parser
from Backend.services.chunker import build_chunks
from Backend.services import embeddings

INGEST_LOCK = Lock()


class IngestionError(ValueError):
    pass


def disease_name(value):
    value = value.strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise IngestionError('Use a category with letters, numbers, underscores, or hyphens.')
    return value


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temp.write_bytes(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def combined_files(root):
    """Keep the existing standalone JSONL format current after uploads."""
    all_lines, output = [], {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        lines = []
        for paper in sorted(folder.glob('*.json')):
            lines.extend(json.dumps(c, ensure_ascii=False) for c in json.loads(paper.read_text(encoding='utf-8')))
        output[folder / f'{folder.name}_chunks.jsonl'] = ('\n'.join(lines) + ('\n' if lines else '')).encode()
        all_lines.extend(lines)
    output[root / 'all_chunks.jsonl'] = ('\n'.join(all_lines) + ('\n' if all_lines else '')).encode()
    return output


def ingest_xml(content, filename, disease, progress=None, collection=None, client=None, data_dir=None):
    """Shared API/CLI pipeline. Never silently overwrite an indexed paper."""
    with INGEST_LOCK:
        return _ingest(content, filename, disease, progress or (lambda **_: None), collection, client,
                       Path(data_dir) if data_dir else DATA_DIR)


def _ingest(content, filename, disease, progress, collection, client, root):
    disease = disease_name(disease)
    if Path(filename).suffix.lower() != '.xml':
        raise IngestionError('Upload a PMC/JATS .xml file.')
    if not content or len(content) > MAX_XML_BYTES:
        raise IngestionError('XML files must be nonempty and no larger than 10 MB.')
    root.mkdir(parents=True, exist_ok=True)
    progress(status='parsing')
    with TemporaryDirectory(prefix='parse-', dir=root) as temp:
        path = Path(temp) / 'article.xml'
        path.write_bytes(content)
        parser = Parser(xml_subfolder=path.parent.name, project_dir=root)
        try:
            paper = parser.parse_file(path.name)
        except (ParseError, DefusedXmlException, ValueError) as error:
            raise IngestionError('Invalid or unsafe PMC/JATS XML. Entity declarations and external entity resolution are not allowed.') from error
    if not paper.get('title'):
        raise IngestionError('The article must have a title.')
    source_hash = hashlib.sha256(content).hexdigest()
    identity = paper.get('pmcid') or paper.get('doi') or source_hash
    stable_id = paper.get('pmcid') or ('paper_' + hashlib.sha256(identity.encode()).hexdigest()[:24])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stable_id):
        raise IngestionError('The article contains an invalid paper identifier.')
    collection = collection if collection is not None else embeddings.get_collection()
    filters = [{'paper_id': stable_id}]
    if paper.get('url'):
        filters.append({'url': paper['url']})
    if paper.get('doi'):
        filters.append({'doi': paper['doi']})
    existing = collection.get(where={'$or': filters} if len(filters) > 1 else filters[0], include=['documents', 'metadatas'])
    if existing['ids']:
        paper_ids = {m['paper_id'] for m in existing['metadatas']}
        if len(paper_ids) != 1:
            raise IngestionError('Multiple indexed versions exist for this paper. Review the library before uploading.')
        stable_id = next(iter(paper_ids))  # Preserve the existing legacy paper IDs.
        if not re.fullmatch(r'[A-Za-z0-9_-]+', stable_id):
            raise IngestionError('An existing paper identifier is unsafe. Review the library first.')
        if any(m.get('disease') != disease for m in existing['metadatas']):
            raise IngestionError('This paper is already indexed under another category. No records were changed.')
    paper['paper_id'] = stable_id
    progress(status='chunking')
    chunks = build_chunks(paper, disease=disease)
    if not chunks:
        raise IngestionError('The article has no usable abstract or body text to embed.')
    if len(chunks) > MAX_CHUNKS:
        raise IngestionError(f'This article exceeds the {MAX_CHUNKS}-chunk limit.')
    progress(total_chunks=len(chunks), completed_chunks=0)
    if existing['ids']:
        previous = {i: (text, meta) for i, text, meta in zip(existing['ids'], existing['documents'], existing['metadatas'])}
        same = len(previous) == len(chunks)
        for chunk in chunks:
            old = previous.get(chunk['chunk_id'])
            if not old or old[0] != chunk['text'] or any(old[1].get(k) != v for k, v in embeddings.chroma_metadata(chunk).items()):
                same = False
        if not same:
            raise IngestionError('This paper is already indexed with different content or metadata. No data was replaced; review the existing version first.')
        return {'outcome': 'already_indexed', 'paper_id': stable_id, 'title': paper['title'], 'disease': disease,
                'chunks': len(chunks), 'collection': embeddings.COLLECTION}
    for chunk in chunks:
        chunk.update(source_sha256=source_hash, pipeline_version=PIPELINE_VERSION,
                     embedding_model=embeddings.MODEL, embedding_dims=embeddings.DIMS,
                     content_sha256=hashlib.sha256(embeddings.document_text(chunk).encode()).hexdigest())
    ids = [c['chunk_id'] for c in chunks]
    if collection.get(ids=ids, include=[])['ids']:
        raise IngestionError('Chunk IDs conflict with existing records. No data was changed.')
    artifacts = {
        root / 'raw' / disease / f'{stable_id}.xml': content,
        root / 'processed' / disease / f'{stable_id}.json': json.dumps(paper, ensure_ascii=False, indent=2).encode(),
        root / 'chunks' / disease / f'{stable_id}.json': json.dumps(chunks, ensure_ascii=False, indent=2).encode(),
    }
    if any(p.exists() for p in artifacts):
        raise IngestionError('A local artifact already uses this paper ID. Review it or index it with the standalone scripts first.')
    progress(status='embedding')
    owns_client = client is None
    client = client if client is not None else embeddings.make_client()
    vectors = []
    try:
        for offset in range(0, len(chunks), embeddings.BATCH_SIZE):
            batch = chunks[offset:offset + embeddings.BATCH_SIZE]
            vectors.extend(embeddings.embed_texts(client, [embeddings.document_text(c) for c in batch]))
            progress(completed_chunks=len(vectors))
    finally:
        if owns_client and hasattr(client, 'close'):
            client.close()
    embeddings.validate_vectors(vectors, len(chunks))
    progress(status='storing')
    saved = {}
    try:
        collection.add(ids=ids, embeddings=vectors, documents=[c['text'] for c in chunks],
                       metadatas=[embeddings.chroma_metadata(c) for c in chunks])
        stored = collection.get(ids=ids, include=['documents'])
        if dict(zip(stored['ids'], stored['documents'])) != {c['chunk_id']: c['text'] for c in chunks}:
            raise RuntimeError('Chroma verification failed.')
        for path, value in artifacts.items():
            saved[path] = None
            write_atomic(path, value)
        for path, value in combined_files(root / 'chunks').items():
            saved[path] = path.read_bytes() if path.exists() else None
            write_atomic(path, value)
    except Exception:
        collection.delete(ids=ids)
        for path, value in saved.items():
            if value is None:
                path.unlink(missing_ok=True)
            else:
                write_atomic(path, value)
        raise
    return {'outcome': 'indexed', 'paper_id': stable_id, 'title': paper['title'], 'disease': disease,
            'chunks': len(chunks), 'collection': embeddings.COLLECTION}
