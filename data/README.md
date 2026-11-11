# Data directory

This directory contains the article source library and the files produced by
the ingestion pipeline. Keep it separate from application code.

## Inputs

- `raw/<category>/*.xml` contains the source PMC/JATS articles. Add new XML
  papers here, using one category folder per disease.
- `papers.csv` is a human-maintained catalog of paper metadata. The current
  parser, API, and CLI do not read it, so adding a row does not trigger parsing
  or ingestion. The XML files and their category folders are the actual inputs.

## Generated files

- `processed/<category>/*.json` contains parsed paper records. The batch parser
  writes these files.
- `chunks/<category>/*.json` contains per-paper chunks. The batch chunker and
  single-paper ingestion pipeline write these files.
- `chunks/<category>/<category>_chunks.jsonl` and
  `chunks/all_chunks.jsonl` are combined chunk exports. They can be rebuilt from
  the per-paper chunk JSON files by running the batch chunker.
- `chroma/` is the persistent ChromaDB library of embeddings and metadata. The
  API, CLI, or embedding script can populate it. Preserve it if you want to
  retain the indexed vectors without embedding all papers again.
- `jobs/` contains temporary JSON progress records created by the local API.
  It is ignored by Git and can be recreated; unfinished jobs are marked
  interrupted after a server restart.

## Backups and regeneration

Back up `raw/` and `papers.csv` as the source collection. Back up `chroma/` if
you need to preserve the existing vector index. `processed/` and `chunks/` are
derived files and can be regenerated from `raw/`, although keeping them makes
the standalone embedding workflow available without repeating parsing and
chunking. Job progress files do not need to be backed up.

To rebuild the batch outputs from the source XML, run these from the project
root (the final step calls the embedding provider):

```powershell
python Backend/src/parse_papers.py
python Backend/src/chunk_papers.py
python Backend/src/embed_chunks.py
```

The default data root is this directory. Setting `DATA_DIR` relocates the
pipeline's source, generated files, Chroma database, and job records together.
The batch parser and chunker use the same configured root.
