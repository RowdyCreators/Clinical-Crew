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


def source(chunk_id, disease, distance):
    return {**SOURCE, 'chunk_id': chunk_id, 'disease': disease, 'distance': distance}


POOL = [source('D1', 'diabetes', 0.20), source('D2', 'diabetes', 0.22), source('L1', 'lyme_disease', 0.24),
        source('A1', 'Alzheimer', 0.40)]


def triage_reply(matching, questions=(), urgent=False):
    import json
    return json.dumps({'matching': [{'condition': c, 'reason': f'{c} fits [1].'} for c in matching],
                       'questions': [{'question': q, 'options': ['Yes', 'No']} for q in questions],
                       'urgent': urgent})


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_first_message_matching_several_diseases_asks_follow_up_questions(self):
        with patch.object(chat, 'retrieve', return_value=POOL) as retrieve, \
             patch.object(chat, 'generate', return_value=triage_reply(
                 ['diabetes', 'lyme_disease', 'not_indexed'], ['Did you notice a rash?'])) as generate:
            response = self.client.post('/api/v1/chat', json={'message': 'Tired and achy'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['kind'], 'clarification')
        self.assertEqual(body['candidates'], ['diabetes', 'lyme_disease'])  # unindexed disease dropped
        self.assertEqual(body['questions'], [{'question': 'Did you notice a rash?', 'options': ['Yes', 'No']}])
        self.assertEqual(body['clarify_round'], 1)
        self.assertIn('Lyme disease', body['answer'])
        retrieve.assert_called_once_with('Tired and achy', k=chat.POOL_SIZE, diseases=None)
        self.assertEqual(generate.call_args.kwargs['system'], chat.TRIAGE_PROMPT)
        self.assertEqual(generate.call_args.kwargs['schema']['properties']['matching']['items']['properties']
                         ['condition']['enum'], ['diabetes', 'lyme_disease', 'Alzheimer'])

    def test_answers_narrowing_to_one_disease_return_final_answer_for_it(self):
        history = [{'role': 'user', 'content': 'Tired and achy'},
                   {'role': 'assistant', 'content': 'Which fits?\n1. Did you notice a rash?'}]
        lyme = [source('L1', 'lyme_disease', 0.2)]
        with patch.object(chat, 'retrieve', side_effect=[POOL[:3], lyme]) as retrieve, \
             patch.object(chat, 'generate', side_effect=[triage_reply(['lyme_disease']), 'Likely Lyme [1].']):
            response = self.client.post('/api/v1/chat', json={
                'message': 'Yes, a bullseye rash', 'history': history,
                'candidates': ['diabetes', 'lyme_disease'], 'clarify_round': 1})
        body = response.json()
        self.assertEqual(body['kind'], 'answer')
        self.assertEqual(body['answer'], 'Likely Lyme [1].')
        self.assertEqual(body['candidates'], ['lyme_disease'])
        first, final = retrieve.call_args_list
        self.assertEqual(first.args[0], 'Tired and achy Yes, a bullseye rash')
        self.assertEqual(first.kwargs['diseases'], ['diabetes', 'lyme_disease'])
        self.assertEqual(final.kwargs['disease'], 'lyme_disease')

    def test_round_limit_answers_without_more_questions(self):
        with patch.object(chat, 'retrieve', return_value=POOL), \
             patch.object(chat, 'generate', return_value='Both remain possible.') as generate:
            response = self.client.post('/api/v1/chat', json={
                'message': 'Not sure', 'history': [{'role': 'user', 'content': 'Tired'}],
                'candidates': ['diabetes', 'lyme_disease'], 'clarify_round': chat.MAX_CLARIFY_ROUNDS})
        self.assertEqual(response.json()['kind'], 'answer')
        generate.assert_called_once()  # no triage call

    def test_skip_and_follow_up_after_an_answer_do_not_triage(self):
        for payload in ({'message': 'Tired', 'clarify': False},
                        {'message': 'What tests?', 'history': [{'role': 'user', 'content': 'Tired'}]}):
            with patch.object(chat, 'retrieve', return_value=POOL), \
                 patch.object(chat, 'generate', return_value='Answer.') as generate:
                self.assertEqual(self.client.post('/api/v1/chat', json=payload).json()['kind'], 'answer')
            generate.assert_called_once()

    def test_urgent_symptoms_skip_questions(self):
        with patch.object(chat, 'retrieve', return_value=POOL), \
             patch.object(chat, 'generate', side_effect=[
                 triage_reply(['diabetes', 'lyme_disease'], ['Rash?'], urgent=True), 'Seek care now.']):
            body = self.client.post('/api/v1/chat', json={'message': 'Confused and fainting'}).json()
        self.assertEqual(body['kind'], 'answer')
        self.assertEqual(body['answer'], 'Seek care now.')

    def test_balance_keeps_every_disease_represented(self):
        picked = chat.balance(POOL, per_disease=1, limit=6)
        self.assertEqual([s['chunk_id'] for s in picked], ['D1', 'L1', 'A1'])


if __name__ == '__main__':
    unittest.main()
