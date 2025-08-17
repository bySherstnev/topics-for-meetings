FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Установка системных зависимостей
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .
COPY requirements-dev.txt .

# Установка Python зависимостей
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директорий для логов и данных
RUN mkdir -p logs models data

# Установка прав на выполнение скриптов
RUN chmod +x scripts/*.py

# Переменные окружения
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Порт для метрик (опционально)
EXPOSE 8000

# Команда запуска
CMD ["python3", "main.py"]
