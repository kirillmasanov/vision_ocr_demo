import base64
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

MIME_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".pdf": "application/pdf",
}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    return {
        "text": [
            {"id": k, "description": v} for k, v in TEXT_MODELS.items()
        ],
        "template": [
            {"id": k, "description": v} for k, v in TEMPLATE_MODELS.items()
        ],
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
            samples.append({
                "name": f.name,
                "url": f"static/{model}/{f.name}",
            })
    return {"samples": samples}


@app.post("/api/recognize")
async def recognize(
    model: str = Form(...),
    file: UploadFile | None = File(None),
    sample_path: str | None = Form(None),
):
    if not YANDEX_API_KEY:
        raise HTTPException(status_code=500, detail="YANDEX_API_KEY is not configured")

    if file and file.size and file.size > 0:
        content_bytes = await file.read()
        suffix = Path(file.filename or "image.jpg").suffix.lower()
    elif sample_path:
        rel_path = sample_path.removeprefix("/static/").removeprefix("static/")
        sample_file = STATIC_DIR / rel_path
        if not sample_file.exists() or not sample_file.is_file():
            raise HTTPException(status_code=404, detail="Sample file not found")
        content_bytes = sample_file.read_bytes()
        suffix = sample_file.suffix.lower()
    else:
        raise HTTPException(status_code=400, detail="No file provided")

    mime_type = MIME_MAP.get(suffix, "JPEG")
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    all_models = {**TEXT_MODELS, **TEMPLATE_MODELS}
    if model not in all_models:
        raise HTTPException(status_code=400, detail="Unknown model")

    language_codes = ["*"]
    if model in ("handwritten", "table", "markdown", "math-markdown"):
        language_codes = ["ru", "en"]
    elif model == "license-plates":
        language_codes = ["ru"]

    body = {
        "mimeType": mime_type,
        "languageCodes": language_codes,
        "model": model,
        "content": content_b64,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
        "x-data-logging-enabled": "true",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(OCR_API_URL, json=body, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Request to Yandex OCR failed: {e}")

    if resp.status_code != 200:
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": error_body},
        )

    return resp.json()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
