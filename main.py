import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
OCR_API_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
OCR_ASYNC_API_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeTextAsync"
OCR_GET_RECOGNITION_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/getRecognition"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB — лимит Yandex Vision OCR

SUPPORTED_MIME_TYPES = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".pdf": "application/pdf",
}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Yandex Vision OCR Demo")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

TEXT_MODELS = {
    "page": "Подойдет для изображений с любым количеством строк текста, сверстанного в одну колонку.",
    "page-column-sort": "Распознает многоколоночный текст.",
    "handwritten": "Распознает произвольное сочетание печатного и рукописного текста на русском и английском языках.",
    "table": "Подходит для распознавания таблиц на русском и английском языках.",
    "markdown": "Распознает текст на изображениях и возвращает результаты в формате Markdown.",
    "math-markdown": "Подойдет для распознавания математических формул. Возвращает результат в формате Markdown с формулами в синтаксисе LaTeX.",
}

TEMPLATE_MODELS = {
    "passport": "Распознавание паспорта. Извлекает стандартные поля: ФИО, дата рождения, номер паспорта, кем выдан и др.",
    "driver-license-front": "Распознавание водительского удостоверения (лицевая сторона). Извлекает ФИО, номер, дату рождения и сроки действия.",
    "driver-license-back": "Распознавание водительского удостоверения (оборотная сторона). Извлекает стаж, номер, даты выдачи и окончания срока действия.",
    "vehicle-registration-front": "Распознавание СТС (лицевая сторона). Извлекает номер авто, VIN, марку, модель, год выпуска, цвет.",
    "vehicle-registration-back": "Распознавание СТС (оборотная сторона). Извлекает ФИО собственника и номер СТС.",
    "license-plates": "Распознавание регистрационных номеров автомобилей. Обеспечивает высокую точность распознавания номерных знаков.",
}

MIME_MAP = SUPPORTED_MIME_TYPES


def _check_credentials() -> None:
    if not YANDEX_API_KEY:
        raise HTTPException(status_code=500, detail="YANDEX_API_KEY is not configured")
    if not YANDEX_FOLDER_ID:
        raise HTTPException(status_code=500, detail="YANDEX_FOLDER_ID is not configured")


def _auth_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
        "x-data-logging-enabled": "true",
    }


def _language_codes(model: str) -> list[str]:
    if model in ("handwritten", "table", "markdown", "math-markdown"):
        return ["ru", "en"]
    if model == "license-plates":
        return ["ru"]
    return ["*"]


async def _load_file(
    file: UploadFile | None, sample_path: str | None
) -> tuple[bytes, str]:
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: max {MAX_FILE_SIZE // 1024 // 1024} MB")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in SUPPORTED_MIME_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported format '{suffix}'. Supported: JPEG, PNG, PDF")
        return content, suffix
    if sample_path:
        rel_path = sample_path.removeprefix("/static/").removeprefix("static/")
        sample_file = (STATIC_DIR / rel_path).resolve()
        if not sample_file.is_relative_to(STATIC_DIR.resolve()):
            raise HTTPException(status_code=400, detail="Invalid sample path")
        if not sample_file.exists() or not sample_file.is_file():
            raise HTTPException(status_code=404, detail="Sample file not found")
        content = sample_file.read_bytes()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: max {MAX_FILE_SIZE // 1024 // 1024} MB")
        suffix = sample_file.suffix.lower()
        if suffix not in SUPPORTED_MIME_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported format '{suffix}'. Supported: JPEG, PNG, PDF")
        return content, suffix
    raise HTTPException(status_code=400, detail="No file provided")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    return {
        "text": [{"id": k, "description": v} for k, v in TEXT_MODELS.items()],
        "template": [{"id": k, "description": v} for k, v in TEMPLATE_MODELS.items()],
    }


@app.get("/api/samples/{model}")
async def get_samples(model: str):
    all_models = {**TEXT_MODELS, **TEMPLATE_MODELS}
    if model not in all_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_dir = STATIC_DIR / model
    if not model_dir.exists():
        return {"samples": []}

    allowed_ext = {".jpg", ".jpeg", ".png", ".pdf", ".gif", ".bmp", ".tiff", ".webp"}
    samples = []
    for f in sorted(model_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in allowed_ext:
            samples.append({"name": f.name, "url": f"static/{model}/{f.name}"})
    return {"samples": samples}


@app.post("/api/recognize")
async def recognize(
    model: str = Form(...),
    file: UploadFile | None = File(None),
    sample_path: str | None = Form(None),
):
    _check_credentials()

    content_bytes, suffix = await _load_file(file, sample_path)
    mime_type = MIME_MAP.get(suffix, "JPEG")
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    all_models = {**TEXT_MODELS, **TEMPLATE_MODELS}
    if model not in all_models:
        raise HTTPException(status_code=400, detail="Unknown model")

    body = {
        "mimeType": mime_type,
        "languageCodes": _language_codes(model),
        "model": model,
        "content": content_b64,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(OCR_API_URL, json=body, headers=_auth_headers())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Request to Yandex OCR failed: {e}")

    if resp.status_code != 200:
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text
        return JSONResponse(status_code=resp.status_code, content={"error": error_body})

    return resp.json()


@app.post("/api/recognize-async")
async def recognize_async(
    model: str = Form(...),
    file: UploadFile | None = File(None),
    sample_path: str | None = Form(None),
):
    _check_credentials()

    if model not in TEXT_MODELS:
        raise HTTPException(status_code=400, detail="Async recognition is only supported for text models")

    content_bytes, suffix = await _load_file(file, sample_path)
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Async recognition requires a PDF file")

    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    body = {
        "mimeType": "application/pdf",
        "languageCodes": _language_codes(model),
        "model": model,
        "content": content_b64,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(OCR_ASYNC_API_URL, json=body, headers=_auth_headers())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Request to Yandex OCR failed: {e}")

    if resp.status_code != 200:
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text
        return JSONResponse(status_code=resp.status_code, content={"error": error_body})

    op_data = resp.json()
    operation_id = op_data.get("id")
    if not operation_id:
        return JSONResponse(status_code=502, content={"error": "No operation ID returned by Yandex OCR"})

    return {"operation_id": operation_id}


@app.get("/api/recognize-status")
async def recognize_status(operation_id: str):
    headers = _auth_headers()
    del headers["Content-Type"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                OCR_GET_RECOGNITION_URL,
                params={"operationId": operation_id},
                headers=headers,
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Request to Yandex OCR failed: {e}")

    if resp.status_code == 200:
        pages = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    pages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return {"done": True, "pages": pages}

    try:
        error_data = resp.json()
    except Exception:
        error_data = {"message": resp.text}

    # Yandex returns HTTP 400 with code=9 (FAILED_PRECONDITION) while the operation is still processing
    msg = str(error_data).lower()
    is_pending = (
        isinstance(error_data, dict) and error_data.get("code") == 9
        or any(x in msg for x in ("not ready", "not completed", "in progress", "processing"))
    )
    if is_pending:
        return {"done": False}

    return JSONResponse(status_code=resp.status_code, content={"error": error_data})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
