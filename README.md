# AI_Backend

FastAPI backend that orchestrates document processing, RAG search, and email automations.

## Authentication

- Requests to chat, RAG, document, and history endpoints must include a Supabase access token in the `Authorization: Bearer <token>` header.
- Configure `SUPABASE_JWT_SECRET` in the environment so the backend can validate Supabase-issued JWTs. Tokens that fail validation or use a different Supabase user are rejected with `401/403` responses.

## Docker Deployment

The repository provides a production-friendly Docker image that bundles Ghostscript and the native
libraries required by `camelot-py[cv]` so PDF table extraction works in containerized environments.

1. Build the image:
   ```bash
   docker build -t ai-backend .
   ```
2. Run the container (adjust mounts/secrets as needed):
   ```bash
   docker run \
     --env-file .env \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/credentials.json:/app/credentials.json:ro \
     ai-backend
   ```

### Notes
- `data/`, and secrets such as `credentials.json` are `.dockerignore`d to keep
  the image lean. Mount them at runtime if your deployment depends on them.
- The default command runs `uvicorn app:app --host 0.0.0.0 --port 8000`. Override the command if you use a different ASGI entrypoint.
- Ensure any required API keys or credentials are supplied through environment variables or mounted files.

## Document Upload API

- POST `/documents/upload` now accepts a `files` field containing one or more PDFs or ZIP archives of PDFs.
- Relative paths supplied by the browser (e.g., when uploading a folder) are preserved so nested directories are stored under `PDF_DIRECTORY`.
- When `SUPABASE_URL`/`SUPABASE_KEY` are supplied, ingested files are tracked in the Supabase table defined by `SUPABASE_DOCUMENTS_TABLE` (default `documents_metadata`). Otherwise the service falls back to the local JSON file defined by `DOCUMENTS_METADATA_PATH` (default `documents_metadata.json`).
- Configure `PINECONE_API_KEY` and `PINECONE_INDEX` to store vector embeddings in Pinecone; document deletions remove matching vectors automatically.
- Each successful ingest response enumerates processed files, directories, and indexed chunk counts; partial failures are reported alongside successes.
- GET `/documents/list` reads the metadata store (Supabase or JSON fallback) and returns the currently indexed documents so the UI can render selectable lists.
