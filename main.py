import os
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from loguru import logger

# Загружаем переменные окружения
load_dotenv()

# Импортируем компоненты бота
from bot.handlers import router
from core.analyzer import topics_analyzer
from utils.telegram_client import telegram_client


class TopicsBot:
    """Основной класс бота"""
    
    def __init__(self):
        self.bot_token = os.getenv("BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("Не указан BOT_TOKEN в переменных окружения")
        
        self.bot = Bot(token=self.bot_token, parse_mode=ParseMode.MARKDOWN)
        self.dp = Dispatcher()
        
        # Настройка логирования
        self._setup_logging()
        
        # Регистрация роутеров
        self.dp.include_router(router)
        
        # Обработчики сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Настройка логирования"""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_retention = int(os.getenv("LOG_RETENTION_DAYS", "30"))
        
        # Удаляем стандартный обработчик
        logger.remove()
        
        # Добавляем обработчик для консоли
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Добавляем обработчик для файла
        logger.add(
            "logs/bot.log",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention=f"{log_retention} days",
            compression="zip"
        )
        
        logger.info("Логирование настроено")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"Получен сигнал {signum}, начинаем graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def startup(self):
        """Инициализация при запуске"""
        logger.info("Запуск бота...")
        
        try:
            # Подключаемся к Telegram
            await telegram_client.connect()
            logger.info("Telegram клиент подключен")
            
            # Очищаем старые данные
            topics_analyzer.cleanup_old_data()
            
            logger.info("Бот успешно запущен")
            
        except Exception as e:
            logger.error(f"Ошибка при запуске: {e}")
            raise
    
    async def shutdown(self):
        """Завершение работы"""
        logger.info("Завершение работы бота...")
        
        try:
            # Отключаемся от Telegram
            await telegram_client.disconnect()
            logger.info("Telegram клиент отключен")
            
            # Закрываем бота
            await self.bot.session.close()
            logger.info("Сессия бота закрыта")
            
        except Exception as e:
            logger.error(f"Ошибка при завершении работы: {e}")
        
        logger.info("Бот завершил работу")
        sys.exit(0)
    
    async def run(self):
        """Запуск бота"""
        try:
            # Инициализация
            await self.startup()
            
            # Запуск бота
            logger.info("Бот запущен и готов к работе")
            await self.dp.start_polling(self.bot)
            
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            await self.shutdown()


@asynccontextmanager
async def lifespan():
    """Контекстный менеджер для управления жизненным циклом"""
    bot = TopicsBot()
    try:
        await bot.startup()
        yield bot
    finally:
        await bot.shutdown()


async def main():
    """Основная функция"""
    async with lifespan() as bot:
        await bot.run()


if __name__ == "__main__":
    # Создаем директорию для логов
    os.makedirs("logs", exist_ok=True)
    
    # Запускаем бота
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)
