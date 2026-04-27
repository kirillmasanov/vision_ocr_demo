# Yandex Vision OCR — Demo

Демо-приложение для демонстрационного сценария сервиса [Yandex Vision OCR](https://yandex.cloud/ru/docs/vision/concepts/ocr/) в Yandex Cloud.

## Что это?

Веб-приложение, которое позволяет загрузить изображение или PDF-документ и распознать текст с помощью Yandex Vision OCR API. Поддерживает все модели сервиса — от распознавания печатного и рукописного текста до извлечения полей из паспортов, водительских удостоверений и СТС.

Ключевые возможности:

* **Распознавание текста** — печатный, многоколоночный, рукописный текст, таблицы, Markdown, математические формулы
* **Шаблонные документы** — паспорта, водительские удостоверения, СТС, автомобильные номера
* **Многостраничный PDF** — асинхронное распознавание до 200 страниц с постраничной навигацией
* **Тестовые примеры** — готовые изображения для каждой модели с предпросмотром по клику на лупу
* **Загрузка файлов** — JPEG, PNG, PDF до 10 МБ с поддержкой drag & drop
* **Просмотр результатов** — распознанный текст, поля документа, распознанные номера, сырой JSON с кнопкой копирования

## Структура проекта

```
vision_ocr_demo/
├── .env.example                # Пример конфигурации
├── pyproject.toml              # Зависимости проекта (uv)
├── Dockerfile                  # Docker конфигурация
├── .dockerignore               # Исключения для Docker
├── main.py                     # FastAPI сервер
├── templates/
│   └── index.html              # Веб-интерфейс
└── static/                     # Тестовые изображения
    ├── page/                   # Печатный текст (одна колонка)
    ├── page-column-sort/       # Многоколоночный текст
    ├── handwritten/            # Рукописный текст
    ├── table/                  # Таблицы
    ├── markdown/               # Markdown
    ├── math-markdown/          # Математические формулы
    ├── passport/               # Паспорта
    ├── driver-license-front/   # Водительские удостоверения (лицевая)
    ├── driver-license-back/    # Водительские удостоверения (оборотная)
    ├── vehicle-registration-front/  # СТС (лицевая)
    ├── vehicle-registration-back/   # СТС (оборотная)
    └── license-plates/         # Автомобильные номера
```

## Быстрый старт

### 1. Установка зависимостей

```bash
# Установить зависимости и создать виртуальное окружение
uv sync
```

> **Примечание:**
> * Если у вас не установлен uv, установите его: `curl -LsSf https://astral.sh/uv/install.sh | sh`
> * Команда `uv sync` автоматически создает виртуальное окружение и устанавливает все зависимости из `pyproject.toml`

### 2. Настройка

Создайте файл `.env` в корне проекта:

```bash
# Скопировать образец
cp .env.example .env

# Отредактировать и добавить свои credentials
nano .env
```

Содержимое `.env`:

```
YANDEX_API_KEY=your_api_key_here
YANDEX_FOLDER_ID=your_folder_id_here
```

### 3. Запуск

#### Вариант 1: Локальный запуск

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Приложение будет доступно по адресу: **http://localhost:8000**

#### Вариант 2: Запуск через Docker

```bash
# Сборка образа
docker build -t vision-ocr-demo .

# Запуск контейнера
docker run -p 8000:8000 --env-file .env vision-ocr-demo
```

Приложение будет доступно по адресу: **http://localhost:8000**

## Использование

### Веб-интерфейс

1. **Откройте** http://localhost:8000 в браузере
2. **Выберите тип модели** — «Модели для распознавания текста» или «Модели для распознавания шаблонных документов»
3. **Выберите модель** из выпадающего списка (с описанием каждой модели)
4. **Выберите изображение** — из тестовых примеров или загрузите своё
5. **Нажмите «Распознать»** и получите результат

### Модели распознавания текста

| Модель | Описание |
| --- | --- |
| `page` | Печатный текст, сверстанный в одну колонку |
| `page-column-sort` | Многоколоночный текст |
| `handwritten` | Печатный и рукописный текст (ru, en) |
| `table` | Таблицы (ru, en) |
| `markdown` | Текст с возвратом в формате Markdown |
| `math-markdown` | Математические формулы (Markdown + LaTeX) |

### Модели распознавания шаблонных документов

| Модель | Описание |
| --- | --- |
| `passport` | Паспорт — ФИО, дата рождения, номер, кем выдан |
| `driver-license-front` | Водительское удостоверение (лицевая сторона) |
| `driver-license-back` | Водительское удостоверение (оборотная сторона) |
| `vehicle-registration-front` | СТС (лицевая сторона) |
| `vehicle-registration-back` | СТС (оборотная сторона) |
| `license-plates` | Регистрационные номера автомобилей |

### API

```bash
# Health check
curl http://localhost:8000/api/health

# Список моделей
curl http://localhost:8000/api/models

# Тестовые изображения для модели
curl http://localhost:8000/api/samples/page

# Распознавание загруженного файла
curl -X POST http://localhost:8000/api/recognize \
  -F "model=page" \
  -F "file=@image.jpg"

# Распознавание тестового изображения
curl -X POST http://localhost:8000/api/recognize \
  -F "model=passport" \
  -F "sample_path=/static/passport/pasport_01.jpg"
```

## Как это работает

### Архитектура

* **Backend**: FastAPI + httpx для взаимодействия с Yandex Vision OCR API
* **Frontend**: Vanilla JS, кастомные компоненты, дизайн в стиле Yandex Cloud
* **API**: Yandex Vision OCR (`ocr.api.cloud.yandex.net/ocr/v1/recognizeText`)
* **Аутентификация**: API-ключ сервисного аккаунта Yandex Cloud

### Поток данных

**Синхронный режим** (изображения и одностраничные PDF):
1. Пользователь выбирает модель и файл
2. Файл кодируется в Base64 и отправляется на `POST /api/recognize`
3. Backend проксирует запрос в `recognizeText` Yandex Vision OCR API
4. Результат отображается: текст, таблица полей или список номеров + сырой JSON

**Асинхронный режим** (многостраничные PDF + текстовые модели):
1. Файл отправляется на `POST /api/recognize-async`
2. Backend передаёт его в `recognizeTextAsync` и возвращает `operation_id`
3. Фронтенд каждые 2 секунды опрашивает `GET /api/recognize-status?operation_id=...`
4. По завершении результат отображается постранично с навигацией

## Endpoints

| Метод | Путь | Описание |
| --- | --- | --- |
| GET | `/` | Веб-интерфейс |
| GET | `/api/health` | Health check |
| GET | `/api/models` | Список доступных моделей |
| GET | `/api/samples/{model}` | Тестовые изображения для модели |
| POST | `/api/recognize` | Синхронное распознавание (JPEG, PNG, PDF) |
| POST | `/api/recognize-async` | Асинхронная отправка многостраничного PDF |
| GET | `/api/recognize-status` | Проверка статуса и получение результата (`?operation_id=`) |

## Добавление тестовых изображений

Для добавления своих тестовых примеров поместите файлы (JPEG, PNG, PDF) в соответствующую папку:

```bash
static/<имя_модели>/
```

Например, для модели `page`:

```bash
static/page/my_image.jpg
```

Изображения автоматически появятся в веб-интерфейсе при выборе соответствующей модели.

## Документация

* [Yandex Vision OCR — Обзор сервиса](https://aistudio.yandex.ru/docs/ru/vision/concepts/ocr/)
* [Распознавание шаблонных документов](https://aistudio.yandex.ru/docs/ru/vision/concepts/ocr/template-recognition.html)
* [OCR API — Справочник](https://aistudio.yandex.ru/docs/ru/vision/ocr/api-ref/)
* [Квоты и лимиты](https://aistudio.yandex.ru/docs/ru/vision/concepts/limits.html)
* [Правила тарификации](https://aistudio.yandex.ru/docs/ru/vision/pricing.html)
