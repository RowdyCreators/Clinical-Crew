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

The upload page is available at **http://127.0.0.1:8010/upload** when using the backend directly. API documentation is at **http://127.0.0.1:8010/docs**.

1. Select or enter a category, such as `Alzheimer`, `diabetes`, `lyme_disease`, or `asthma`.
2. Choose one PMC/JATS `.xml` file, up to 10 MB.
3. Select **Upload and process**.
4. Follow progress through parsing, chunking, embedding, and storing.
5. The result reports the paper ID, category, and chunk count. Re-uploading an unchanged paper reports **Already indexed** without generating more embeddings.

Use the same category name, including case, for an existing paper. Changing an already-indexed paper's content or category is rejected for review; this first version does not silently replace existing evidence.

The page is lightweight HTML/CSS/JavaScript served by FastAPI. There is no separate frontend install or build step.

## Run the chat frontend

The chat UI is a React + Vite + TypeScript app in `Frontend/`. Run the backend and frontend in separate terminals:

1. **Terminal 1:** Start the backend with the command in [Run the upload page](#run-the-upload-page) above. Leave it running on port `8010`.
2. **Terminal 2:** From the project root, install the locked frontend dependencies and start Vite:

```powershell
cd Frontend
npm.cmd ci
npm.cmd run dev
```

Node.js 20.19 or later (or 22.12 or later) is required. On macOS/Linux, use `npm ci` and `npm run dev` in place of `npm.cmd`.

Open **http://localhost:5173** for chat and **http://localhost:5173/upload** for the upload page. The Vite dev server proxies `/api`, `/upload`, and `/static` to `http://127.0.0.1:8010`; set `VITE_API_TARGET` to use another backend address. To type-check and create a production build in `Frontend/dist/`, run `npm.cmd run build` from `Frontend/` on Windows, or `npm run build` on macOS/Linux.

Type a description of the symptoms and press Enter (Shift+Enter adds a new line). Each answer cites numbered excerpts such as [1]; select a citation or **Show research excerpts** to see the paper, section, and similarity score behind it. The **Category** menu restricts retrieval to one indexed category. Earlier turns of the conversation are sent with each message so follow-up questions keep their context; **New chat** clears them.

If the first description fits more than one indexed disease, Clinical Crew asks up to three follow-up questions to tell them apart before answering. Pick an option for each question and select **Send answers**, type the answers instead, or select **Skip questions**. The backend retrieves a broad set of excerpts, and a triage step decides which of the retrieved diseases still fit. Only indexed categories can be candidates. Later rounds search only the remaining candidates. After two rounds, or when the symptoms look urgent, it answers right away. Choosing a single **Category** skips this step.

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

## Contributor map: where behavior belongs

Follow these paths when changing a feature:

- **Upload a paper:** [`api/ingestion.py`](Backend/api/ingestion.py) validates
  the HTTP upload and submits it to [`services/jobs.py`](Backend/services/jobs.py).
  The worker calls [`services/ingestion.py`](Backend/services/ingestion.py),
  which coordinates the stages below.

  ```text
  upload route -> job worker -> ingestion coordinator
                                -> parser -> paper identity / duplicate checks
                                -> chunker -> embeddings + Chroma
                                -> artifact store (raw, parsed, chunk files)
  ```

- **Change XML parsing:** edit [`services/parser.py`](Backend/services/parser.py).
  The coordinator uses it for uploads; the batch parser also uses it to process
  all files under `data/raw/`.
- **Change chunk boundaries or section handling:** edit
  [`services/chunker.py`](Backend/services/chunker.py). Both single-paper
  ingestion and the batch chunk command use this module.
- **Change embedding requests, vector validation, or Chroma access:** edit
  [`services/embeddings.py`](Backend/services/embeddings.py). Ingestion uses it
  to store vectors; chat uses it to search; the connection-check route uses it
  to test provider access.
- **Change duplicate rules or stable paper IDs:** edit
  [`services/paper_identity.py`](Backend/services/paper_identity.py).
- **Change generated files or rollback behavior:** edit
  [`services/artifact_store.py`](Backend/services/artifact_store.py). Job status
  JSON writes also use its atomic writer.
- **Change chat retrieval or answer construction:** start at
  [`api/chat.py`](Backend/api/chat.py) for HTTP validation and responses, and
  [`services/chat.py`](Backend/services/chat.py) for retrieval, prompt
  construction, model calls, and source formatting.
- **Change API setup, route registration, or startup/shutdown behavior:** edit
  [`Backend/main.py`](Backend/main.py). It creates the job worker and registers
  the routers.

The **chat path** is `api/chat.py -> services/chat.py -> services/embeddings.py`
for retrieval, then the chat service calls the answering model and returns the
answer with its sources. Batch scripts under `Backend/src/` call the same
parser, chunker, and embedding services in separate steps; they do not go
through the upload route or job worker.

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
