"""Chat endpoint tests. Retrieval and generation are mocked; no provider calls are made."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from Backend import main
from Backend.services import chat

SOURCE = {'chunk_id': 'PMC1_0', 'distance': 0.25, 'text': 'Polyuria and polydipsia are classic diabetes symptoms.',
          'paper_id': 'PMC1', 'disease': 'diabetes', 'title': 'Diabetes overview', 'year': 2024, 'heading': 'Symptoms'}


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_answer_returns_grounded_text_and_public_sources(self):
        with patch.object(chat, 'retrieve', return_value=[SOURCE]) as retrieve, \
             patch.object(chat, 'generate', return_value='Likely diabetes [1].') as generate:
            response = self.client.post('/api/v1/chat', json={
                'message': 'Always thirsty', 'disease': 'diabetes', 'top_k': 3,
                'history': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['answer'], 'Likely diabetes [1].')
        self.assertEqual(body['sources'][0]['relevance'], 0.75)
        self.assertNotIn('distance', body['sources'][0])
        retrieve.assert_called_once_with('Always thirsty', k=3, disease='diabetes')
        prompt = generate.call_args.args[0]
        self.assertIn('[1] Diabetes overview (2024) - Symptoms', prompt)
        self.assertIn('ASSISTANT: hello', prompt)
        self.assertTrue(prompt.endswith('USER: Always thirsty'))

    def test_blank_message_is_rejected(self):
        self.assertEqual(self.client.post('/api/v1/chat', json={'message': '   '}).status_code, 400)
        self.assertEqual(self.client.post('/api/v1/chat', json={'message': ''}).status_code, 422)

    def test_no_sources_returns_safe_error(self):
        with patch.object(chat, 'retrieve', return_value=[]):
            response = self.client.post('/api/v1/chat', json={'message': 'headache'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('No matching research excerpts', response.json()['detail'])

    def test_provider_failure_maps_to_503(self):
        with patch.object(chat, 'retrieve', side_effect=chat.EmbeddingError('Embedding request failed.')):
            response = self.client.post('/api/v1/chat', json={'message': 'headache'})
        self.assertEqual(response.status_code, 503)

    def test_answer_model_rate_limit_retries_then_reports_answer_model(self):
        class RateLimited(Exception):
            code = 429

        class Models:
            calls = 0

            def generate_content(self, **kwargs):
                Models.calls += 1
                raise RateLimited()

        client = type('Client', (), {'models': Models()})()
        with patch.object(chat.time, 'sleep') as sleep:
            with self.assertRaises(chat.AnswerError) as caught:
                chat.generate('prompt', client=client)
        self.assertEqual(Models.calls, chat.GENERATE_ATTEMPTS)
        self.assertEqual(sleep.call_count, chat.GENERATE_ATTEMPTS - 1)
        self.assertIn('answering model', str(caught.exception))
        self.assertNotIn('Embedding', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
