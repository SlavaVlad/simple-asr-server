import logging
import os
import json
import whisper
from pathlib import Path
from typing import Dict, Optional, Set, Literal, List, Union
from threading import Lock

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, Form
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модель для параметров транскрибации
class TranscribeParams(BaseModel):
    language: Optional[str] = Field(None, description="Язык аудио (auto-detect по умолчанию)")
    task: Optional[str] = Field("transcribe", description="transcribe или translate")
    temperature: Optional[float] = Field(0.0, description="Температура для генерации (0.0-1.0)")
    beam_size: Optional[int] = Field(None, description="Размер луча для поиска")
    best_of: Optional[int] = Field(None, description="Количество кандидатов для выбора лучшего")
    compression_ratio_threshold: Optional[float] = Field(None, description="Порог сжатия для фильтрации")
    logprob_threshold: Optional[float] = Field(None, description="Порог логарифмической вероятности")
    no_speech_threshold: Optional[float] = Field(None, description="Порог детекции отсутствия речи")
    condition_on_previous_text: Optional[bool] = Field(True, description="Использовать предыдущий текст как контекст")
    initial_prompt: Optional[str] = Field(None, description="Начальная подсказка для модели")
    word_timestamps: Optional[bool] = Field(False, description="Временные метки слов")
    prepend_punctuations: Optional[str] = Field(None, description="Знаки препинания для добавления в начало")
    append_punctuations: Optional[str] = Field(None, description="Знаки препинания для добавления в конец")
    clip_timestamps: Optional[List[float]] = Field(None, description="Временные метки для обрезки аудио")
    hallucination_silence_threshold: Optional[float] = Field(None, description="Порог тишины для детекции галлюцинаций")
    format: Optional[Literal["json", "simple", "text"]] = Field("json", description="Формат ответа")

# Глобальные переменные для модели и ключей
model = None
api_keys: Set[str] = set()
keys_lock = Lock()
keys_file_path = os.getenv("KEYS_FILE", "keys.txt")

# Схема безопасности
api_key_header = APIKeyHeader(name="X-API-Key")

app = FastAPI(title="Whisper ASR Service", version="1.0.0")

def load_api_keys():
    """Загружает API ключи из файла"""
    global api_keys
    try:
        if os.path.exists(keys_file_path):
            with open(keys_file_path, 'r') as f:
                keys = [line.strip() for line in f.readlines() if line.strip()]
                with keys_lock:
                    api_keys = set(keys)
                logger.info(f"Загружено {len(api_keys)} API ключей")
        else:
            logger.warning(f"Файл ключей {keys_file_path} не найден")
    except Exception as e:
        logger.error(f"Ошибка загрузки ключей: {e}")

def load_model():
    """Загружает модель Whisper"""
    global model
    model_name = os.getenv("DEFAULT_MODEL", "turbo")
    download_root = os.getenv("MODEL_DOWNLOAD_ROOT", "./models")
    device = os.getenv("MODEL_DEVICE", "cpu")

    try:
        logger.info(f"Загрузка модели Whisper: {model_name}")
        model = whisper.load_model(model_name, device=device, download_root=download_root)
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Проверяет API ключ"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API ключ не предоставлен")

    # Перезагружаем ключи для проверки обновлений
    load_api_keys()

    with keys_lock:
        if api_key not in api_keys:
            raise HTTPException(status_code=403, detail="Неверный API ключ")

    return api_key

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    load_api_keys()
    load_model()

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "model_loaded": model is not None, "current_model": str(model) if model else None}

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    params: TranscribeParams = Depends(),
    api_key: str = Depends(verify_api_key)
):
    """Транскрибирует аудиофайл"""
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    # Готовим параметры для whisper.transcribe()
    whisper_params = {}
    for field_name, field_value in params.dict(exclude_none=True, exclude={'format'}).items():
        whisper_params[field_name] = field_value

    # Формат ответа
    response_format = params.format

    temp_file_path = None
    try:
        # Сохраняем временный файл
        temp_file_path = f"/tmp/{audio_file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)

        # Транскрибируем
        logger.info(f"Транскрибация файла: {audio_file.filename} с параметрами: {whisper_params}")
        result = model.transcribe(temp_file_path, **whisper_params)

        # Удаляем временный файл
        os.unlink(temp_file_path)

        # Возвращаем результат в нужном формате
        if response_format == 'text':
            return PlainTextResponse(content=result['text'])
        elif response_format == 'simple':
            return {"text": result['text']}
        else:  # json - полный ответ по умолчанию
            return result

    except Exception as e:
        logger.error(f"Ошибка транскрибации: {e}")
        # Удаляем временный файл в случае ошибки
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Ошибка транскрибации: {str(e)}")

@app.post("/keys/reload")
async def reload_keys(api_key: str = Depends(verify_api_key)):
    """Перезагружает ключи из файла"""
    load_api_keys()
    with keys_lock:
        return {"message": f"Перезагружено {len(api_keys)} ключей"}

@app.get("/keys/count")
async def get_keys_count(api_key: str = Depends(verify_api_key)):
    """Возвращает количество активных ключей"""
    with keys_lock:
        return {"count": len(api_keys)}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9854"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False
    )
