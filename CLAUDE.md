# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run locally
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Build and run with Docker
docker build -t vision-ocr-demo .
docker run -p 8000:8000 --env-file .env vision-ocr-demo
```

No test suite is configured.

## Environment

Copy `.env.example` to `.env` and populate:
- `YANDEX_API_KEY` — Yandex Cloud API key
- `YANDEX_FOLDER_ID` — Yandex Cloud folder ID

## Architecture

Single-file FastAPI backend (`main.py`) + single-page frontend (`templates/index.html`).

**Request flow:**
1. Frontend Base64-encodes the image and POSTs to `/api/recognize`
2. `main.py` forwards the request to `ocr.api.cloud.yandex.net/ocr/v1/recognizeText` via `httpx`
3. Raw API response is returned as-is to the frontend
4. Frontend parses and renders results in one of several tabs (formatted text, entities/fields, JSON)

**Backend endpoints (`main.py`):**
- `GET /` — serves `templates/index.html` via Jinja2
- `GET /api/models` — returns model metadata (name, label, type, description)
- `GET /api/samples/{model}` — scans `static/{model}/` and returns filenames
- `POST /api/recognize` — proxies to Yandex Vision OCR API; accepts `model`, `image` (base64), and optional `mime_type`
- `GET /api/health` — health check

**Static images** in `static/` are organized by model name (e.g., `static/passport/`, `static/table/`) and served at `/static/{model}/{filename}`.

**Model types:**
- *Text models* (`page`, `page-column-sort`, `handwritten`, `table`, `markdown`, `math-markdown`) — return `blocks[].lines[].words[]` with recognized text
- *Template models* (`passport`, `driver-license-front`, `driver-license-back`, `vehicle-registration-front`, `vehicle-registration-back`, `license-plates`) — return `entities[]` with named fields

**Frontend (`templates/index.html`):**
- Vanilla JS, no build step; all logic is inline in the single HTML file
- `selectMode(mode)` switches between text/template model categories
- `recognize()` sends the request and calls `showResults(data, model)`
- `showResults()` branches on model type to render the appropriate tab content
- Yandex Cloud design system: CSS custom properties in `:root`, Inter font, color tokens (`--color-brand`, `--color-danger`, etc.)
