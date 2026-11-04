"""Isolated pipeline/API tests. Embedding calls are mocked; no real corpus is changed."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient

from Backend import main
from Backend.services import embeddings, ingestion
from Backend.services.jobs import Jobs


def article(pmc='99990001', text='Research evidence describes symptoms and diagnostic methods for this study population.'):
    return f'''<article><front><journal-meta><journal-title>Test Journal</journal-title></journal-meta>
      <article-meta><article-id pub-id-type="pmc">{pmc}</article-id>
      <article-id pub-id-type="doi">10.1234/{pmc}</article-id>
      <title-group><article-title>Research paper {pmc}</article-title></title-group>
      <pub-date pub-type="epub"><year>2026</year></pub-date>
      <abstract><p>{text}</p></abstract></article-meta></front>
      <body><sec><title>Findings</title><p>{text}</p></sec></body></article>'''.encode()


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = chromadb.PersistentClient(path=str(self.root / 'chroma'))
        self.collection = self.db.get_or_create_collection('medical_papers', embedding_function=None,
            metadata={'hnsw:space': 'cosine', 'embedding_model': embeddings.MODEL, 'dims': embeddings.DIMS})
        self.client_patch = patch.object(embeddings, 'make_client', return_value=object())
        self.client_patch.start()
        self.embed_patch = patch.object(embeddings, 'embed_texts', side_effect=lambda client, texts: [[1.0] + [0.0] * (embeddings.DIMS - 1) for _ in texts])
        self.embed = self.embed_patch.start()

    def tearDown(self):
        self.embed_patch.stop()
        self.client_patch.stop()
        self.db.close()
        embeddings.close_clients()
        self.temp.cleanup()

    def ingest(self, xml=None, filename='article.xml', disease='diabetes', **kwargs):
        return ingestion.ingest_xml(xml if xml is not None else article(), filename, disease,
                                    collection=self.collection, data_dir=self.root, **kwargs)

    def test_complete_pipeline_persists_vectors_metadata_and_artifacts_and_skips_duplicate(self):
        stages = []
        result = self.ingest(progress=lambda **fields: stages.append(fields.get('status')))
        self.assertEqual(result['outcome'], 'indexed')
        self.assertEqual(result['paper_id'], 'PMC99990001')
        self.assertEqual(self.collection.count(), result['chunks'])
        stored = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        self.assertTrue(all(len(v) == 768 for v in stored['embeddings']))
        self.assertTrue(all(m['pipeline_version'] == 'xml-v1' and len(m['source_sha256']) == 64 for m in stored['metadatas']))
        self.assertTrue((self.root / 'raw/diabetes/PMC99990001.xml').exists())
        self.assertTrue((self.root / 'processed/diabetes/PMC99990001.json').exists())
        chunks = json.loads((self.root / 'chunks/diabetes/PMC99990001.json').read_text())
        self.assertEqual({c['chunk_id']: c['text'] for c in chunks}, dict(zip(stored['ids'], stored['documents'])))
        self.assertEqual(len((self.root / 'chunks/all_chunks.jsonl').read_text().splitlines()), result['chunks'])
        self.assertEqual([s for s in stages if s], ['parsing', 'chunking', 'embedding', 'storing'])
        calls = self.embed.call_count
        self.assertEqual(self.ingest(filename='renamed.xml')['outcome'], 'already_indexed')
        self.assertEqual(self.embed.call_count, calls)
        self.assertEqual(self.collection.count(), result['chunks'])
        # The stored vectors are queryable without involving an external embedding API.
        found = self.collection.query(query_embeddings=[[1.0] + [0.0] * 767], n_results=1)
        self.assertEqual(len(found['ids'][0]), 1)

    def test_changed_paper_or_category_is_rejected_without_modifying_original(self):
        self.ingest()
        before = self.collection.get(include=['documents', 'metadatas'])
        for kwargs in ({'xml': article(text='A different revised article has changed the clinical evidence considerably.')}, {'disease': 'asthma'}):
            with self.assertRaises(ingestion.IngestionError):
                self.ingest(**kwargs)
        self.assertEqual(self.collection.get(include=['documents', 'metadatas']), before)

    def test_legacy_paper_is_recognized_without_reembedding(self):
        self.ingest()
        data = self.collection.get(include=['documents', 'metadatas', 'embeddings'])
        self.collection.delete(ids=data['ids'])
        ids = []
        for i, meta in enumerate(data['metadatas']):
            meta['paper_id'] = 'diabetes_001'
            ids.append(f'diabetes_001_c{i+1}')
        self.collection.add(ids=ids, embeddings=data['embeddings'], documents=data['documents'], metadatas=data['metadatas'])
        self.embed.reset_mock()
        self.assertEqual(self.ingest()['paper_id'], 'diabetes_001')
        self.embed.assert_not_called()

    def test_rejects_malformed_unsafe_empty_and_unsupported_xml(self):
        samples = [b'', b'<broken>', b'<root/>',
            b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///secret">]><article>&x;</article>',
            article(text='x')]
        for xml in samples:
            with self.subTest(xml=xml[:20]), self.assertRaises(ingestion.IngestionError):
                self.ingest(xml)
        for kwargs in ({'filename': 'article.pdf'}, {'disease': '../outside'}):
            with self.assertRaises(ingestion.IngestionError):
                self.ingest(**kwargs)
        self.embed.assert_not_called()
        self.assertEqual(self.collection.count(), 0)

    def test_standard_jats_doctype_is_accepted_without_fetching_dtd(self):
        xml = b'<!DOCTYPE article SYSTEM "https://example.invalid/JATS.dtd">' + article()
        self.assertEqual(self.ingest(xml)['outcome'], 'indexed')

    def test_provider_failure_or_invalid_vector_leaves_no_partial_paper(self):
        for outcome in (RuntimeError('provider failure'), [[float('nan')] * 768], [[1.0, 0.0]]):
            self.embed.side_effect = outcome if isinstance(outcome, Exception) else lambda *args: outcome
            with self.assertRaises((RuntimeError, ValueError)):
                self.ingest()
            self.assertEqual(self.collection.count(), 0)
            self.assertFalse((self.root / 'processed/diabetes/PMC99990001.json').exists())

    def test_artifact_failure_rolls_back_only_new_records(self):
        self.ingest()
        before_count = self.collection.count()
        previous = (self.root / 'chunks/all_chunks.jsonl').read_bytes()
        real_write = ingestion.write_atomic
        failed = False
        def write(path, content):
            nonlocal failed
            if path.name == 'all_chunks.jsonl' and not failed:
                failed = True
                raise OSError('simulated disk failure')
            real_write(path, content)
        with patch.object(ingestion, 'write_atomic', side_effect=write), self.assertRaises(OSError):
            self.ingest(article('99990002'))
        self.assertEqual(self.collection.count(), before_count)
        self.assertEqual((self.root / 'chunks/all_chunks.jsonl').read_bytes(), previous)
        self.assertFalse((self.root / 'processed/diabetes/PMC99990002.json').exists())

    def test_collection_model_mismatch_is_rejected(self):
        self.collection.modify(metadata={'embedding_model': 'different', 'dims': 768})
        with patch.object(embeddings, 'DB_DIR', self.root / 'chroma'), self.assertRaises(ValueError):
            embeddings.get_collection()

    def test_embedding_authentication_error_is_clear_and_does_not_expose_key(self):
        class AuthError(Exception):
            code = 401
        from types import SimpleNamespace
        from unittest.mock import Mock
        provider = SimpleNamespace(models=SimpleNamespace(embed_content=Mock(side_effect=AuthError('secret-key-in-provider-error'))))
        # Exercise the real wrapper rather than the pipeline's mocked embedding function.
        self.embed_patch.stop()
        try:
            with self.assertRaises(embeddings.EmbeddingError) as caught:
                embeddings.embed_texts(provider, ['document'])
            self.assertIn('HTTP 401', str(caught.exception))
            self.assertNotIn('secret-key', str(caught.exception))
        finally:
            self.embed = self.embed_patch.start()

    def test_api_upload_progress_and_validation(self):
        with patch.object(main, 'Jobs', side_effect=lambda: Jobs(self.root)), patch.object(embeddings, 'get_collection', return_value=self.collection):
            with TestClient(main.app) as api:
                self.assertEqual(api.get('/').status_code, 200)
                self.assertEqual(api.get('/static/upload.js').status_code, 200)
                response = api.post('/api/v1/ingestions', data={'disease': 'diabetes'}, files={'file': ('paper.xml', article(), 'application/xml')})
                self.assertEqual(response.status_code, 202)
                job_id = response.json()['id']
                for _ in range(1000):
                    job = api.get('/api/v1/ingestions/' + job_id).json()
                    if job['status'] in ('completed', 'failed'):
                        break
                    time.sleep(.01)
                self.assertEqual(job['status'], 'completed', job)
                self.assertEqual(job['result']['chunks'], self.collection.count())
                self.assertEqual(api.post('/api/v1/ingestions', data={'disease': '../escape'}, files={'file': ('x.xml', article())}).status_code, 400)
                self.assertEqual(api.post('/api/v1/ingestions', data={'disease': 'test'}, files={'file': ('x.pdf', article())}).status_code, 400)
                self.assertEqual(api.get('/api/v1/ingestions/' + 'f' * 32).status_code, 404)

    def test_interrupted_job_is_recoverable_after_restart(self):
        jobs = Jobs(self.root)
        job = {'id': 'a' * 32, 'status': 'embedding'}
        jobs.update(job)
        jobs.close()
        restarted = Jobs(self.root)
        self.assertEqual(restarted.get(job['id'])['status'], 'failed')
        restarted.close()


if __name__ == '__main__':
    unittest.main()
