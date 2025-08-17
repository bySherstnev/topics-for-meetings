#!/usr/bin/env python3
"""
Скрипт для загрузки моделей, необходимых для работы бота
"""

import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


def download_embeddings_model():
    """Загрузка модели эмбеддингов"""
    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    logger.info(f"Загрузка модели эмбеддингов: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Загружаем модель (это автоматически скачает её при первом использовании)
        model = SentenceTransformer(model_name)
        logger.info(f"Модель эмбеддингов загружена: {model_name}")
        
        # Проверяем размер модели
        model_path = model.model_path
        if model_path:
            size_mb = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"Размер модели: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
        return False


def download_llama_model():
    """Загрузка модели Llama (только метаданные)"""
    model_path = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    
    logger.info(f"Подготовка модели Llama: {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoConfig
        
        # Загружаем токенизатор и конфигурацию
        logger.info("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info("Загрузка конфигурации модели...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Модель Llama подготовлена: {model_path}")
        logger.info(f"Размер словаря: {config.vocab_size}")
        logger.info(f"Количество слоев: {config.num_hidden_layers}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка подготовки модели Llama: {e}")
        logger.warning("Модель Llama будет загружена при первом запуске бота")
        return False


def download_nltk_data():
    """Загрузка данных NLTK"""
    logger.info("Загрузка данных NLTK...")
    
    try:
        import nltk
        
        # Список необходимых данных NLTK
        nltk_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                logger.info(f"Загружены данные NLTK: {data}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить данные NLTK {data}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки данных NLTK: {e}")
        return False


def check_dependencies():
    """Проверка зависимостей"""
    logger.info("Проверка зависимостей...")
    
    required_packages = [
        'torch',
        'transformers',
        'sentence_transformers',
        'scikit-learn',
        'hdbscan',
        'numpy',
        'pandas',
        'sqlalchemy',
        'aiogram',
        'telethon',
        'nltk',
        'pymorphy2',
        'emoji',
        'python-dotenv',
        'pydantic',
        'loguru',
        'asyncio-throttle',
        'bitsandbytes',
        'accelerate'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} - не установлен")
    
    if missing_packages:
        logger.error(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
        logger.info("Установите их командой: pip install -r requirements.txt")
        return False
    
    logger.info("Все зависимости установлены")
    return True


def create_directories():
    """Создание необходимых директорий"""
    logger.info("Создание директорий...")
    
    directories = [
        "logs",
        "models",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Создана директория: {directory}")


def main():
    """Основная функция"""
    logger.info("Начинаем загрузку моделей и подготовку окружения...")
    
    # Создаем директории
    create_directories()
    
    # Проверяем зависимости
    if not check_dependencies():
        logger.error("Проверка зависимостей не пройдена")
        return False
    
    # Загружаем данные NLTK
    download_nltk_data()
    
    # Загружаем модель эмбеддингов
    if not download_embeddings_model():
        logger.error("Не удалось загрузить модель эмбеддингов")
        return False
    
    # Подготавливаем модель Llama
    download_llama_model()
    
    logger.info("✅ Загрузка моделей завершена успешно!")
    logger.info("Теперь можно запускать бота командой: python main.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Загрузка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)
