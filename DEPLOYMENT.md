# 🚀 Инструкция по развертыванию

## Предварительные требования

### Системные требования
- **ОС**: Linux x86_64 (Ubuntu 20.04+ рекомендуется)
- **RAM**: Минимум 32GB (рекомендуется 64GB)
- **VRAM**: 16-24GB для Llama 3.1 8B
- **GPU**: NVIDIA GPU с поддержкой CUDA 12.1+
- **Диск**: Минимум 50GB свободного места

### Программное обеспечение
- Python 3.9+
- Docker & Docker Compose
- NVIDIA Container Toolkit
- Git

## Пошаговое развертывание

### 1. Подготовка системы

```bash
# Обновляем систему
sudo apt update && sudo apt upgrade -y

# Устанавливаем необходимые пакеты
sudo apt install -y python3.11 python3.11-dev python3-pip git curl wget

# Устанавливаем Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Устанавливаем NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Клонирование и настройка

```bash
# Клонируем репозиторий
git clone <repository-url>
cd topics-for-meetings

# Копируем конфигурацию
cp env.example .env

# Редактируем конфигурацию
nano .env
```

### 3. Настройка Telegram

#### Создание бота
1. Найдите @BotFather в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям для создания бота
4. Сохраните токен бота

#### Получение API ключей
1. Перейдите на https://my.telegram.org
2. Войдите в свой аккаунт
3. Перейдите в "API development tools"
4. Создайте новое приложение
5. Сохраните `api_id` и `api_hash`

#### Настройка .env файла
```env
# Telegram
BOT_TOKEN=your_bot_token_here
API_ID=your_api_id_here
API_HASH=your_api_hash_here
PHONE_NUMBER=your_phone_number_here

# Остальные настройки оставьте по умолчанию
```

### 4. Запуск через Docker (рекомендуется)

```bash
# Собираем и запускаем контейнер
docker-compose up -d

# Проверяем логи
docker-compose logs -f bot
```

### 5. Запуск локально

```bash
# Создаем виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Загружаем модели
python scripts/download_models.py

# Запускаем бота
python main.py
```

## Настройка бота в группе

### 1. Добавление бота в группу
1. Создайте группу в Telegram
2. Добавьте бота в группу
3. Назначьте бота администратором с правами:
   - Чтение сообщений
   - Просмотр истории сообщений

### 2. Тестирование
1. Отправьте команду `/start` боту
2. В группе отправьте `/topics`
3. Проверьте, что бот отвечает корректно

## Мониторинг и обслуживание

### Просмотр логов
```bash
# Docker
docker-compose logs -f bot

# Локально
tail -f logs/bot.log
```

### Мониторинг ресурсов
```bash
# Использование GPU
nvidia-smi

# Использование памяти
docker stats topics-bot
```

### Очистка старых данных
```bash
# Автоматически (ежедневно)
# Или вручную через код:
python -c "from core.analyzer import topics_analyzer; topics_analyzer.cleanup_old_data()"
```

### Обновление бота
```bash
# Останавливаем контейнер
docker-compose down

# Обновляем код
git pull

# Пересобираем и запускаем
docker-compose up -d --build
```

## Устранение неполадок

### Проблемы с GPU
```bash
# Проверяем доступность GPU
nvidia-smi

# Проверяем Docker GPU поддержку
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Проблемы с памятью
```bash
# Уменьшаем квантизацию в .env
QUANTIZATION=8bit  # вместо 4bit

# Или используем CPU (медленнее)
# Удалите GPU секцию из docker-compose.yml
```

### Проблемы с Telegram API
```bash
# Проверяем права бота
# Бот должен быть администратором группы

# Проверяем лимиты API
# Не более 1 запроса в минуту на чат
```

### Проблемы с базой данных
```bash
# Проверяем права доступа к файлу БД
ls -la bot_data.db

# Пересоздаем БД (осторожно - потеря данных)
rm bot_data.db
```

## Оптимизация производительности

### Настройка CUDA
```bash
# В .env файле
CUDA_VISIBLE_DEVICES=0  # Используем только первую GPU
```

### Настройка памяти
```bash
# Увеличиваем swap (если нужно)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Настройка Docker
```bash
# В /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

## Безопасность

### Рекомендации
1. Используйте отдельного пользователя для запуска бота
2. Ограничьте доступ к файлам конфигурации
3. Регулярно обновляйте зависимости
4. Мониторьте логи на подозрительную активность

### Файрвол
```bash
# Ограничиваем доступ к портам
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp  # если нужны метрики
sudo ufw enable
```

## Резервное копирование

### Автоматическое резервное копирование
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_$DATE.tar.gz data/ logs/ .env
```

### Восстановление
```bash
tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz
```

## Масштабирование

### Горизонтальное масштабирование
```bash
# Используйте балансировщик нагрузки
# Каждый экземпляр бота с отдельной БД
```

### Вертикальное масштабирование
```bash
# Увеличьте RAM и VRAM
# Используйте более мощную GPU
```

## Поддержка

При возникновении проблем:
1. Проверьте логи: `docker-compose logs bot`
2. Проверьте системные ресурсы: `htop`, `nvidia-smi`
3. Проверьте конфигурацию в `.env`
4. Создайте issue в репозитории с подробным описанием проблемы
