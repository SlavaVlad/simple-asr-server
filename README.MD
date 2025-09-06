# ASR service на базе Whisper + AMD ROCm

Сервис для распознавания речи с использованием GPU AMD через ROCm.

## Требования

- GPU AMD с поддержкой ROCm (RX5000, RX6000 +)
- Установленный ROCm
- Docker + Compose plugin
- Debian based linux (для инструкции)
- Минимум 8GB RAM
- Минимум 2GB свободного места + место для ROCm + Модели (turbo ~3.5GB)

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SlavaVlad/simple-asr-server.git ./asr
cd asr
```

### 2. Создание Docker-образа

Перед запуском в файле docker-compose.yml можно изменить всякие переменные окружения, например модель по умолчанию.
```bash
docker compose up -d
```
Тут он сам всё соберёт и запустится.

## Управление API-ключами

При первом запуске сервиса автоматически создается файл `keys.txt` со случайно сгенерированным API-ключом. 

### Добавление новых ключей
1. Откройте файл `keys.txt`
2. Добавьте новые ключи, каждый с новой строки
3. Перезапустите сервис

Пример содержимого `keys.txt`:
```
f7a9c2b3d4e5a6b7c8d9e0f1a2b3c4d5
e8b9a2c3d4f5g6h7i8j9k0l1m2n3o4p5
```

## Использование API

### Распознавание речи

```bash
curl -X POST "http://localhost:9854/transcribe" \
     -H "x-api-key: ВАШ_КЛЮЧ" \
     -F "file=@путь_к_аудио_файлу" \
     -F "model_name=turbo" # (необязательно, по умолчанию turbo)
```

### Ответ API

```json
{
    "transcription": [...],
    "text": "распознанный текст",
    "metrics": {
        "processing_time_seconds": 1.23,
        "characters_per_second": 45.6,
        "audio_realtime_ratio": 2.34,
        "audio_duration": 5.67,
        "text_length": 89
    }
}
```

```text

```

## Метрики производительности

API возвращает следующие метрики:
- `processing_time_seconds`: время обработки в секундах
- `characters_per_second`: скорость обработки (символов в секунду)
- `audio_realtime_ratio`: отношение длительности аудио к времени обработки
- `audio_duration`: длительность аудио в секундах
- `text_length`: количество символов в распознанном тексте

## Поддерживаемые форматы

Поддерживаются все аудиоформаты, которые может обработать FFmpeg. Файлы автоматически конвертируются в нужный формат.

## Решение проблем

### Проверка статуса сервиса
```bash
curl http://localhost:9854/docs
```

### Частые проблемы

1. **Ошибка доступа к GPU**
   ```bash
   sudo usermod -a -G video,render $USER
   sudo reboot
   ```

2. **Ошибка 403**
   - Проверьте правильность API-ключа
   - Убедитесь, что ключ добавлен в файл keys.txt
   - Перезапустите сервис

## Лицензия

MIT License

