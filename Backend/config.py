"""Project paths and backend-only configuration."""
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
_settings = {**dotenv_values(ROOT / 'Backend' / '.env'), **dotenv_values(ROOT / '.env'), **os.environ}
DATA_DIR = Path(_settings.get('DATA_DIR', str(ROOT / 'data'))).resolve()
CHROMA_DIR = DATA_DIR / 'chroma'
EMBEDDING_MODEL = _settings.get('EMBEDDING_MODEL', 'gemini-embedding-2')
EMBEDDING_DIMS = int(_settings.get('EMBEDDING_DIMS', '768'))
MAX_XML_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 2000
PIPELINE_VERSION = 'xml-v1'


def embedding_api_key():
    # Read the credential when creating each client. Editing .env must not reuse
    # an old value that load_dotenv previously copied into the running process.
    names = ('GEMINI_API_KEY', 'GEMINI_APIKEY', 'GOOGLE_API_KEY')
    for path in (ROOT / '.env', ROOT / 'Backend' / '.env'):
        values = dotenv_values(path, interpolate=False)
        for name in names:
            if values.get(name):
                return values[name].strip()
    return next((os.environ[name].strip() for name in names if os.environ.get(name)), None)
