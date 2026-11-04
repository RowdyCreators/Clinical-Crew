"""Filesystem artifacts produced by ingestion."""
import json
from pathlib import Path
from uuid import uuid4


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temp.write_bytes(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def combined_files(root):
    """Build category and global JSONL indexes from per-paper chunk files."""
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


def artifact_paths(root, disease, paper_id):
    """Return the raw, parsed, and per-paper chunk output paths."""
    return (
        root / 'raw' / disease / f'{paper_id}.xml',
        root / 'processed' / disease / f'{paper_id}.json',
        root / 'chunks' / disease / f'{paper_id}.json',
    )


def publish_artifacts(collection, ids, chunks, artifacts, chunks_root, writer=write_atomic):
    """Store vectors and filesystem outputs, restoring both on a normal failure."""
    saved = {}
    try:
        collection.add(ids=ids, embeddings=artifacts['vectors'], documents=[c['text'] for c in chunks],
                       metadatas=artifacts['metadatas'])
        stored = collection.get(ids=ids, include=['documents'])
        if dict(zip(stored['ids'], stored['documents'])) != {c['chunk_id']: c['text'] for c in chunks}:
            raise RuntimeError('Chroma verification failed.')
        for path, value in artifacts['files'].items():
            saved[path] = None
            writer(path, value)
        for path, value in combined_files(chunks_root).items():
            saved[path] = path.read_bytes() if path.exists() else None
            writer(path, value)
    except Exception:
        collection.delete(ids=ids)
        for path, value in saved.items():
            if value is None:
                path.unlink(missing_ok=True)
            else:
                writer(path, value)
        raise
