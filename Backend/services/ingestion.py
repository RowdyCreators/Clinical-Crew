"""Coordinate the XML ingestion flow from parsing through publication."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError

from Backend.config import DATA_DIR, MAX_CHUNKS, MAX_XML_BYTES, PIPELINE_VERSION
from Backend.services import embeddings
from Backend.services.artifact_store import artifact_paths, publish_artifacts, write_atomic
from Backend.services.artifact_store import combined_files  # Kept as a public helper for compatibility.
from Backend.services.chunker import build_chunks
from Backend.services.ingestion_errors import IngestionError, disease_name
from Backend.services.paper_identity import (
    find_existing,
    reject_chunk_id_conflicts,
    reject_existing_changes,
    stable_paper_id,
)
from Backend.services.parser import Parser

INGEST_LOCK = Lock()


def ingest_xml(content, filename, disease, progress=None, collection=None, client=None, data_dir=None):
    """Run the shared API/CLI pipeline; never silently overwrite an indexed paper."""
    with INGEST_LOCK:
        return _ingest(content, filename, disease, progress or (lambda **_: None), collection, client,
                       Path(data_dir) if data_dir else DATA_DIR)


def _parse_paper(content, root, progress):
    """Parse one validated file and report an actionable error on unsafe XML."""
    progress(status='parsing')
    with TemporaryDirectory(prefix='parse-', dir=root) as temp:
        path = Path(temp) / 'article.xml'
        path.write_bytes(content)
        parser = Parser(xml_subfolder=path.parent.name, project_dir=root)
        try:
            paper = parser.parse_file(path.name)
        except (ParseError, DefusedXmlException, ValueError) as error:
            raise IngestionError(
                'Invalid or unsafe PMC/JATS XML. Entity declarations and external entity resolution are not allowed.'
            ) from error
    if not paper.get('title'):
        raise IngestionError('The article must have a title.')
    return paper


def _embed_chunks(chunks, client, progress):
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
    return vectors


def _ingest(content, filename, disease, progress, collection, client, root):
    """Keep the pipeline's major stages in order; helpers own stage details."""
    disease = disease_name(disease)
    if Path(filename).suffix.lower() != '.xml':
        raise IngestionError('Upload a PMC/JATS .xml file.')
    if not content or len(content) > MAX_XML_BYTES:
        raise IngestionError('XML files must be nonempty and no larger than 10 MB.')
    root.mkdir(parents=True, exist_ok=True)

    # Parse the source and establish its stable identity.
    paper = _parse_paper(content, root, progress)
    source_hash = hashlib.sha256(content).hexdigest()
    stable_id = stable_paper_id(paper, source_hash)
    collection = collection if collection is not None else embeddings.get_collection()
    existing, stable_id = find_existing(collection, paper, stable_id, disease)
    paper['paper_id'] = stable_id

    # Build and validate section-aware chunks before any embedding calls.
    progress(status='chunking')
    chunks = build_chunks(paper, disease=disease)
    if not chunks:
        raise IngestionError('The article has no usable abstract or body text to embed.')
    if len(chunks) > MAX_CHUNKS:
        raise IngestionError(f'This article exceeds the {MAX_CHUNKS}-chunk limit.')
    progress(total_chunks=len(chunks), completed_chunks=0)
    if existing['ids']:
        reject_existing_changes(existing, chunks)
        return {'outcome': 'already_indexed', 'paper_id': stable_id, 'title': paper['title'], 'disease': disease,
                'chunks': len(chunks), 'collection': embeddings.COLLECTION}

    for chunk in chunks:
        chunk.update(source_sha256=source_hash, pipeline_version=PIPELINE_VERSION,
                     embedding_model=embeddings.MODEL, embedding_dims=embeddings.DIMS,
                     content_sha256=hashlib.sha256(embeddings.document_text(chunk).encode()).hexdigest())
    ids = [chunk['chunk_id'] for chunk in chunks]
    reject_chunk_id_conflicts(collection, ids)

    files = artifact_paths(root, disease, stable_id)
    if any(path.exists() for path in files):
        raise IngestionError('A local artifact already uses this paper ID. Review it or index it with the standalone scripts first.')
    artifact_bytes = {
        files[0]: content,
        files[1]: json.dumps(paper, ensure_ascii=False, indent=2).encode(),
        files[2]: json.dumps(chunks, ensure_ascii=False, indent=2).encode(),
    }

    # Generate and validate every vector before publishing any records.
    progress(status='embedding')
    vectors = _embed_chunks(chunks, client, progress)

    # Publish vectors and files together, with rollback for ordinary failures.
    progress(status='storing')
    publish_artifacts(
        collection, ids, chunks,
        {'vectors': vectors, 'metadatas': [embeddings.chroma_metadata(c) for c in chunks], 'files': artifact_bytes},
        root / 'chunks', writer=write_atomic,
    )
    return {'outcome': 'indexed', 'paper_id': stable_id, 'title': paper['title'], 'disease': disease,
            'chunks': len(chunks), 'collection': embeddings.COLLECTION}
