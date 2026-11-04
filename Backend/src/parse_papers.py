"""Compatibility entry point; use python -m Backend.ingest for the complete pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Backend.services.parser import *
if __name__ == '__main__':
    run_all_diseases(only=sys.argv[1:] or None)
