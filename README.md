# 🤖 Генератор тем для митапов (Telegram Bot)

Бот для автоматического анализа групповых чатов и генерации релевантных тем для обсуждения на митапах.

## 🎯 Возможности

- Анализ сообщений за последние 1-14 дней
- Генерация 3-7 релевантных тем с описаниями
- Использование локальной Llama 3.1 8B для качественной генерации
- Кластеризация сообщений с помощью HDBSCAN
- Кэширование результатов для оптимизации

## 🚀 Быстрый старт

### Требования

- Python 3.9+
- 16-24GB VRAM для Llama 3.1 8B
- Linux x86_64 (рекомендуется)
- Docker & Docker Compose

### 1. Клонирование и настройка

```bash
git clone <repository>
cd topics-for-meetings
cp .env.example .env
# Отредактируйте .env с вашими настройками
```

### 2. Запуск через Docker

```bash
docker-compose up -d
```

### 3. Запуск локально

```bash
# Установка зависимостей
pip install -r requirements.txt

# Загрузка моделей
python scripts/download_models.py

# Запуск бота
python main.py
```

## ⚙️ Конфигурация

### Переменные окружения (.env)

```env
# Telegram
BOT_TOKEN=your_bot_token
API_ID=your_api_id
API_HASH=your_api_hash
PHONE_NUMBER=your_phone_number

# Модели
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
QUANTIZATION=4bit
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Настройки анализа
DEFAULT_PERIOD=7d
MAX_PERIOD=14d
MIN_MESSAGES=50
MIN_CLUSTER_SIZE=8

# База данных
DATABASE_URL=sqlite:///bot_data.db
```

### Настройка Telegram

1. Создайте бота через @BotFather
2. Получите API_ID и API_HASH на https://my.telegram.org
3. Добавьте бота в группу с правами администратора

## 📊 Команды

- `/topics` - анализ за последние 7 дней
- `/topics 3d` - анализ за 3 дня (поддерживается: 1d, 3d, 7d, 14d)

## 🏗️ Архитектура

```
├── main.py                 # Точка входа
├── bot/
│   ├── __init__.py
│   ├── handlers.py        # Обработчики команд
│   ├── keyboards.py       # Inline клавиатуры
│   └── messages.py        # Тексты сообщений
├── core/
│   ├── __init__.py
│   ├── analyzer.py        # Основная логика анализа
│   ├── clustering.py      # Кластеризация сообщений
│   ├── embeddings.py      # Генерация эмбеддингов
│   └── llm.py            # Интеграция с Llama
├── data/
│   ├── __init__.py
│   ├── database.py        # Работа с БД
│   └── models.py          # Модели данных
├── utils/
│   ├── __init__.py
│   ├── telegram_client.py # Telethon клиент
│   └── text_processing.py # Обработка текста
└── scripts/
    └── download_models.py # Загрузка моделей
```

## 🔧 Разработка

### Установка для разработки

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Тестирование

```bash
pytest tests/
```

### Линтинг

```bash
black .
isort .
flake8 .
```

## 📈 Мониторинг

- Логи: `logs/bot.log`
- Метрики: Prometheus endpoint `/metrics`
- Мониторинг VRAM: встроенный в Docker

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License
