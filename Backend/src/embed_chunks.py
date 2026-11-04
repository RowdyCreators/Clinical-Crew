"""Compatibility entry point for embedding existing JSONL chunks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Backend.services.embeddings import *
if __name__ == '__main__':
    main()
