# Используем образ ROCm с предустановленным PyTorch
FROM rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

# Открываем порт, на котором будет работать приложение
EXPOSE 9854

# Устанавливаем переменные окружения для ROCm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ROCM_ARCH=gfx1030

# Команда для запуска приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9854", "--log-level", "debug"]
