# Simple ASR Server

Простой сервер для автоматического распознавания речи (ASR) на базе OpenAI Whisper.

## Особенности

- Поддержка различных моделей Whisper (tiny, base, small, medium, large, turbo)
- Три формата вывода: plaintext, simple JSON, полный JSON
- Параметр speedup для ускорения аудио перед распознаванием
- Автоматическая конвертация аудио в поддерживаемый формат
- API ключи для безопасности
- Docker поддержка

## Быстрый старт

### Локальная установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Скопируйте и настройте переменные окружения:
```bash
cp .env.example .env
```

3. Запустите сервер:
```bash
python app.py
```

### Docker

1. Постройте и запустите контейнер:
```bash
docker-compose up --build
```

## API

### POST /transcribe

Распознавание речи из аудиофайла.

**Параметры:**
- `file` (файл) - Аудиофайл для распознавания
- `model_name` (опционально) - Модель Whisper для использования
- `output_format` - Формат вывода: `plaintext`, `simple`, или `json`
- `speedup` - Коэффициент ускорения аудио (0.25-4.0)

**Заголовки:**
- `x-api-key` - API ключ

**Примеры:**

```bash
# Простой текстовый вывод
curl -X POST "http://localhost:9854/transcribe?output_format=plaintext&speedup=1.5" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"

# JSON с только текстом
curl -X POST "http://localhost:9854/transcribe?output_format=simple" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"

# Полный JSON ответ с использованием другой модели
curl -X POST "http://localhost:9854/transcribe?output_format=json&model_name=base" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"
```

### GET /health

Проверка состояния сервера.

## Переменные окружения

См. `.env.example` для полного списка доступных переменных:

- `HOST` - Хост сервера (по умолчанию: 0.0.0.0)
- `PORT` - Порт сервера (по умолчанию: 9854)
- `DEFAULT_MODEL` - Модель по умолчанию (по умолчанию: turbo)
- `MODEL_DOWNLOAD_ROOT` - Папка для загрузки моделей
- `KEYS_FILE` - Файл с API ключами
