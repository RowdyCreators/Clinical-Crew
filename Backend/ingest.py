"""Process XML with the same pipeline as the upload API."""
import argparse
import json
import sys
from pathlib import Path

from Backend.services.ingestion import IngestionError, ingest_xml
from Backend.config import MAX_XML_BYTES
from Backend.services.embeddings import EmbeddingError, close_clients


def main():
    parser = argparse.ArgumentParser(description='Convert one PMC/JATS XML file into Chroma embeddings.')
    parser.add_argument('xml', type=Path)
    parser.add_argument('--disease', required=True, help='Category, e.g. Alzheimer, diabetes, lyme_disease')
    args = parser.parse_args()
    try:
        if args.xml.stat().st_size > MAX_XML_BYTES:
            raise IngestionError('The XML file exceeds the 10 MB limit.')
        result = ingest_xml(args.xml.read_bytes(), args.xml.name, args.disease,
                            progress=lambda **fields: print(json.dumps(fields), file=sys.stderr))
        print(json.dumps(result, indent=2))
    except (IngestionError, EmbeddingError, OSError) as error:
        parser.exit(1, f'{error}\n')
    except Exception:
        parser.exit(1, 'Processing failed. Check backend configuration, provider access/quota, and storage.\n')
    finally:
        close_clients()


if __name__ == '__main__':
    main()
