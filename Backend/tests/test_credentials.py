"""No network calls. Verify credential reloads and redacted provider diagnostics."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from Backend import config
from Backend.services import embeddings


class CredentialTests(unittest.TestCase):
    def test_env_file_edit_is_used_without_restart_and_shadows_stale_environment(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, 'ROOT', Path(temp)):
            path = Path(temp) / '.env'
            path.write_text('GEMINI_API_KEY=AQ.first-test-value\n')
            with patch.dict(config.os.environ, {'GEMINI_API_KEY': 'stale-exported-test-key'}):
                self.assertEqual(config.embedding_api_key(), 'AQ.first-test-value')
                path.write_text('GEMINI_API_KEY=AQ.updated-test-value\n')
                self.assertEqual(config.embedding_api_key(), 'AQ.updated-test-value')

    def test_backend_file_alias_and_environment_fallback(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, 'ROOT', Path(temp)):
            backend = Path(temp) / 'Backend'
            backend.mkdir()
            (backend / '.env').write_text('GEMINI_APIKEY="  AQ.alias-test-value  "\n')
            self.assertEqual(config.embedding_api_key(), 'AQ.alias-test-value')
            (backend / '.env').unlink()
            with patch.dict(config.os.environ, {'GEMINI_API_KEY': 'environment-test-value'}):
                self.assertEqual(config.embedding_api_key(), 'environment-test-value')

    def test_new_auth_keys_are_passed_unchanged_to_native_developer_api(self):
        with patch.object(embeddings, 'embedding_api_key', return_value='AQ.auth-format-test-value'), patch('google.genai.Client') as factory:
            embeddings.make_client()
            self.assertEqual(factory.call_args.kwargs['api_key'], 'AQ.auth-format-test-value')
            self.assertFalse(factory.call_args.kwargs['vertexai'])
            options = factory.call_args.kwargs['http_options']
            self.assertEqual(options.base_url, 'https://generativelanguage.googleapis.com')
            self.assertEqual(options.api_version, 'v1beta')

    def test_error_reason_is_preserved_but_credentials_and_raw_messages_are_not(self):
        provider = SimpleNamespace(code=401, details={'error': {'message': 'secret-key-must-not-leak', 'details': [
            {'reason': 'ACCESS_TOKEN_TYPE_UNSUPPORTED', 'metadata': {'key': 'another-secret'}}]}})
        error = embeddings.provider_error(provider)
        self.assertEqual(error.reason, 'ACCESS_TOKEN_TYPE_UNSUPPORTED')
        self.assertIn('ACCESS_TOKEN_TYPE_UNSUPPORTED', str(error))
        self.assertNotIn('secret', str(error))

    def test_connection_check_tests_embedding_method_once_and_closes_client(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] + [0.0] * 767)])
        with patch.object(embeddings, 'make_client', return_value=client):
            self.assertEqual(embeddings.check_connection()['status'], 'ready')
            client.models.get.assert_not_called()
            client.models.embed_content.assert_called_once()
            client.close.assert_called_once()
