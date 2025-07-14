FROM rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    python3-venv \

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

EXPOSE 9854

# Устанавливаем переменные окружения для ROCm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ROCM_ARCH=gfx1030

# Команда для запуска приложения
CMD ["python3", "app.py"]
