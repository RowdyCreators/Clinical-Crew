# Clinical Crew

Clinical Crew prepares medical research papers for retrieval. Upload one PMC/JATS XML article and the backend parses it, creates chunks, generates Gemini embeddings, and stores the records in ChromaDB.

This version implements document ingestion and a symptom chat: a React frontend sends the user's description to `/api/v1/chat`, which retrieves the closest paper excerpts from ChromaDB and asks a Gemini model for a cited, non-definitive answer. Clinical accuracy evaluation is separate work.

## Run the upload page

From the project root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn Backend.main:app --reload --host 127.0.0.1 --port 8010
```

Open **http://127.0.0.1:8010**. API documentation is at **http://127.0.0.1:8010/docs**.

1. Select or enter a category, such as `Alzheimer`, `diabetes`, `lyme_disease`, or `asthma`.
2. Choose one PMC/JATS `.xml` file, up to 10 MB.
3. Select **Upload and process**.
4. Follow progress through parsing, chunking, embedding, and storing.
5. The result reports the paper ID, category, and chunk count. Re-uploading an unchanged paper reports **Already indexed** without generating more embeddings.

Use the same category name, including case, for an existing paper. Changing an already-indexed paper's content or category is rejected for review; this first version does not silently replace existing evidence.

The page is lightweight HTML/CSS/JavaScript served by FastAPI. There is no separate frontend install or build step.

## Run the chat frontend

The chat UI is a React + Vite + TypeScript app in `Frontend/`. Start the backend first (see above, port 8010), then in a second terminal:

```powershell
cd Frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api` to `http://127.0.0.1:8010`; set `VITE_API_TARGET` to use another backend address. `npm run build` type-checks and writes a production build to `Frontend/dist/`.

Type a description of the symptoms and press Enter (Shift+Enter adds a new line). Each answer cites numbered excerpts such as [1]; select a citation or **Show research excerpts** to see the paper, section, and similarity score behind it. The **Category** menu restricts retrieval to one indexed category. Earlier turns of the conversation are sent with each message so follow-up questions keep their context; **New chat** clears them.

Answers are decision support grounded in the indexed papers, not a diagnosis.

## Environment configuration

Keep the API key in the root `.env` file. `.env.example` documents the settings; do not overwrite an existing `.env`.

```dotenv
GEMINI_API_KEY=your-google-gemini-api-key
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMS=768
# Model that writes chat answers; uses the same Gemini key
ANSWER_MODEL=gemini-3.5-flash-lite
```

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are also accepted. For credentials, the root `.env` takes precedence, followed by `Backend/.env`, then environment variables. Each new embedding client rereads the files, so saving a corrected key takes effect on the next request without restarting. Other settings, including model and dimensions, are loaded at startup and require a restart. The answering-model API key is not needed for XML ingestion.

The default embedding model/dimensions match the existing library. An existing Chroma collection with conflicting model/dimension metadata is rejected. Use a separate `DATA_DIR` for a different embedding model rather than mixing vectors.

Optional configuration:

```dotenv
# DATA_DIR=C:/path/to/a/separate/library
```

The frontend never receives the key. Provider exceptions are converted into safe error messages. Both standard and newer Google auth-key formats are accepted and sent through the official SDK using the native Gemini Developer API. The app does not infer that a key is invalid from its prefix or silently switch providers/models.

Use **Check Gemini connection** on the page to test the actual embedding method with one small probe. It generates a test embedding but saves nothing and does not upload a paper. HTTP 401/403 means Google rejected authentication/access. Recognized provider reasons, such as `ACCESS_TOKEN_TYPE_UNSUPPORTED`, are shown safely rather than hiding everything behind a generic credential error. That specific reason requires checking the credential/project/API binding in Google AI Studio; it can also be reproduced independently of this pipeline. Saving a corrected key takes effect on the next request. Temporary provider errors use bounded retries; the connection probe makes one attempt.

## Command-line ingestion

The CLI runs the same complete pipeline as the browser:

```powershell
.\venv\Scripts\python.exe -m Backend.ingest "C:\papers\article.xml" --disease lyme_disease
```

Progress goes to stderr and the final result is JSON. Failures return a nonzero exit code. Relative XML paths are resolved from your current terminal directory.

## Folder structure

```text
Clinical-Crew/
  Backend/
    main.py                 FastAPI entry point and upload page
    config.py               Environment settings and paths
    ingest.py               Complete pipeline CLI
    api/
      ingestion.py          Upload and job-status routes
      embedding.py          Embedding connection diagnostic route
    services/
      parser.py             PMC/JATS XML -> paper dictionary
      chunker.py            Paper -> section-aware chunks
      embeddings.py         Gemini calls, validation, Chroma access, search helpers
      ingestion.py          Coordinates the complete pipeline stages
      ingestion_errors.py   Shared input validation and ingestion errors
      paper_identity.py     Stable paper IDs and duplicate/conflict checks
      artifact_store.py     Atomic artifact writes, JSONL rebuild, and rollback
      jobs.py               One local worker and JSON progress files
    api/chat.py             Chat and category endpoints
    services/chat.py        Retrieval + grounded answer generation
    static/
      index.html            Upload screen
      style.css             Page styling
      upload.js             Upload/progress interactions
    src/                    Compatibility entry points for the old stage scripts
    tests/
      test_ingestion.py     Isolated pipeline and API checks
      test_chat.py          Chat endpoint checks with mocked retrieval/generation
  Frontend/                 React + Vite + TypeScript chat UI
  data/
    README.md               Inputs, generated artifacts, and backup guidance
    raw/<category>/         Original XML papers
    processed/<category>/   Parsed paper JSON
    chunks/<category>/      Per-paper chunk JSON and category JSONL
    chunks/all_chunks.jsonl Combined chunk file
    chroma/                 The vector database
    jobs/                   Local JSON job status files, ignored by Git
  .env                      Private backend credentials, ignored by Git
  .env.example              Configuration template without real credentials
  requirements.txt          Pinned runtime dependencies
  requirements-dev.txt      Test dependencies
```

The API and CLI both call `services/ingestion.py`. It keeps the processing order
visible while delegating identity checks to `paper_identity.py` and filesystem
publication to `artifact_store.py`. Parsing, chunking, and embedding remain in
their own service modules. The scripts under `src/` are compatibility entry
points for the earlier separate pipeline stages; use the API or CLI for the
complete ingestion workflow.

## Where the chunks and embeddings are stored

**The vector database is ChromaDB, in `data/chroma/`. Its collection is `medical_papers`.**

Each collection record contains:

- A unique chunk ID.
- A 768-dimensional embedding vector.
- The original cleaned chunk text.
- Paper and section metadata: paper ID, category, title, authors, year, journal, DOI, source URL, license, heading, and parent section.
- For newly ingested records: source XML checksum, embedded-content checksum, pipeline version, embedding model, and dimensions.

Chroma manages `chroma.sqlite3` and its binary index directories. Its internal SQLite `embeddings` table stores record identities; vector payloads are persisted in Chroma's operation log and vector index. Use the Chroma collection API instead of editing these internal tables.

`raw/` contains source XML; `processed/` and `chunks/` are generated pipeline outputs, not additional vector databases. `papers.csv` is a manual catalog and is not read by the current parser or ingestion flow. See [`data/README.md`](data/README.md) for backup and regeneration guidance. Job progress uses JSON files rather than introducing another database.

The starting library contains **eight papers and 363 chunks**: Alzheimer (193), diabetes (53), and Lyme disease (117). Existing paper IDs are recognized through their source URL/DOI and retained, so re-uploading those papers does not create a second copy. Counts increase when new papers are successfully ingested.

## Processing and failure behavior

- Validates category names, file extension, nonempty input, and a 10 MB upload limit.
- Parses PMC/JATS articles with a title and usable abstract/body text. Arbitrary XML formats are not supported.
- Accepts normal JATS DOCTYPE declarations without fetching their external DTD. Entity declarations/expansion and external entity resolution are blocked.
- Uses PMCID for stable new paper IDs; falls back to a DOI/source hash.
- Reuses the existing section-aware chunker with sentence boundaries and overlap.
- Validates embedding count, dimensions, finite values, and nonzero vectors.
- Completes all embedding requests before writing the new paper to Chroma.
- Verifies stored IDs and text before publishing the raw/parsed/chunk artifacts.
- Rebuilds category/global JSONL files so the existing stage-script format stays usable.
- Attempts to remove the new records and restore artifacts if a normal save failure occurs. Existing papers are preserved.
- Changed indexed content is rejected rather than being mistaken for an unchanged duplicate.
- Writes progress files under `data/jobs/`; the page can resume checking the most recent job after refresh.
- On server restart, unfinished jobs are marked interrupted and can be retried by uploading again.

Progress states are `queued`, `parsing`, `chunking`, `embedding`, `storing`, `completed`, and `failed`.

This is a **local, single-process ingestion tool**. Run one backend worker per data directory and avoid running the CLI against that directory while the server is processing a paper. Job execution is not a durable distributed queue. Chroma and filesystem publication are not a shared transaction; abrupt process termination or disk failure may require inspecting/retrying the affected paper. Keep backups of the complete data directory.

The upload endpoints currently have no authentication and are intended to stay on localhost. Authentication and a server-backed deployment belong to the later shared-service version.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Upload page |
| GET | `/health` | Process health |
| POST | `/api/v1/embedding/check` | Test an embedding request without storing anything |
| POST | `/api/v1/ingestions` | Multipart `file` and `disease`; returns HTTP 202 and a job |
| GET | `/api/v1/ingestions/{job_id}` | Progress, safe error, or result |
| POST | `/api/v1/chat` | JSON `message`, optional `history`, `disease`, `top_k`; returns `answer` and cited `sources` |
| GET | `/api/v1/chat/categories` | Categories present in the vector database |

## Validation

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m unittest discover -s Backend/tests -v
.\venv\Scripts\python.exe -m compileall -q Backend
```

Tests use temporary Chroma databases and mocked embeddings. They cover actual Chroma storage/querying, duplicates and legacy IDs, changed-paper conflicts, malformed/unsafe XML, standard JATS declarations, provider/vector errors, rollback after a save failure, upload/status endpoints, credential-error redaction, and interrupted jobs. They do not change the project's real corpus or send real embedding requests.

A successful ingestion establishes that the document was processed and indexed. Clinical retrieval/answer quality needs separate evaluation.

## Contributors

- Muhammad Baibars Zain Ul Abideen
- Shiwin Fernando
- Vinh Thanh Nguyen
