import os
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from data.database import db
from utils.telegram_client import telegram_client
from utils.text_processing import text_processor
from core.embeddings import embeddings_generator
from core.clustering import message_clusterer
from core.llm import llama_generator


class TopicsAnalyzer:
    """Основной класс для анализа и генерации тем"""
    
    def __init__(self):
        self.min_messages = int(os.getenv("MIN_MESSAGES", "50"))
        self.default_period = os.getenv("DEFAULT_PERIOD", "7d")
        self.max_period = os.getenv("MAX_PERIOD", "14d")
    
    def parse_period(self, period_str: str) -> int:
        """Парсинг периода из строки (1d, 3d, 7d, 14d)"""
        if not period_str:
            period_str = self.default_period
        
        # Удаляем пробелы и приводим к нижнему регистру
        period_str = period_str.strip().lower()
        
        # Парсим период
        if period_str.endswith('d'):
            try:
                days = int(period_str[:-1])
                max_days = int(self.max_period[:-1])
                return min(days, max_days)
            except ValueError:
                logger.warning(f"Неверный формат периода: {period_str}")
                return int(self.default_period[:-1])
        
        # Если формат не распознан, возвращаем значение по умолчанию
        logger.warning(f"Неизвестный формат периода: {period_str}")
        return int(self.default_period[:-1])
    
    async def analyze_chat(self, chat_id: int, period: str = None, 
                          force_regenerate: bool = False) -> Dict[str, Any]:
        """Основной метод анализа чата"""
        start_time = time.time()
        
        try:
            # Парсим период
            days = self.parse_period(period)
            period_key = f"{days}d"
            
            # Проверяем кэш (если не требуется принудительная регенерация)
            if not force_regenerate:
                cached_result = db.get_analysis_cache(chat_id, period_key)
                if cached_result:
                    logger.info(f"Возвращаем кэшированный результат для чата {chat_id}")
                    return {
                        "topics": cached_result.get_topics(),
                        "message_count": cached_result.message_count,
                        "cluster_count": cached_result.cluster_count,
                        "cached": True,
                        "processing_time": 0
                    }
            
            # Проверяем права администратора
            bot_info = await self._get_bot_info()
            if bot_info:
                is_admin = await telegram_client.check_admin_rights(chat_id, bot_info["id"])
                if not is_admin:
                    return {
                        "error": "Нет прав администратора",
                        "error_type": "admin_required"
                    }
            
            # Получаем сообщения
            messages = await self._get_messages(chat_id, days)
            
            if len(messages) < self.min_messages:
                return {
                    "error": f"Недостаточно сообщений для анализа",
                    "error_type": "insufficient_data",
                    "message_count": len(messages),
                    "min_required": self.min_messages
                }
            
            # Анализируем сообщения
            analysis_result = await self._analyze_messages(messages)
            
            # Сохраняем в кэш
            db.save_analysis_cache(
                chat_id=chat_id,
                period=period_key,
                topics=analysis_result["topics"],
                message_count=len(messages),
                cluster_count=analysis_result["cluster_count"]
            )
            
            processing_time = time.time() - start_time
            
            return {
                "topics": analysis_result["topics"],
                "message_count": len(messages),
                "cluster_count": analysis_result["cluster_count"],
                "cached": False,
                "processing_time": processing_time,
                "stats": analysis_result.get("stats", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа чата {chat_id}: {e}")
            return {
                "error": "Временные проблемы с анализом",
                "error_type": "processing_error"
            }
    
    async def _get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Получить информацию о боте"""
        try:
            # Здесь можно получить информацию о боте через aiogram
            # Пока возвращаем заглушку
            return {"id": 123456789}  # Заглушка
        except Exception as e:
            logger.error(f"Ошибка получения информации о боте: {e}")
            return None
    
    async def _get_messages(self, chat_id: int, days: int) -> List[Dict[str, Any]]:
        """Получение сообщений из чата"""
        logger.info(f"Получение сообщений из чата {chat_id} за {days} дней")
        
        # Сначала пробуем получить из базы данных
        db_messages = db.get_messages_for_period(chat_id, days)
        
        if len(db_messages) >= self.min_messages:
            logger.info(f"Используем {len(db_messages)} сообщений из базы данных")
            return [
                {
                    "text": msg.text,
                    "normalized_text": text_processor.normalize_text(msg.text),
                    "date": msg.date,
                    "user_id": msg.user_id
                }
                for msg in db_messages
            ]
        
        # Если в базе недостаточно данных, получаем через Telegram API
        logger.info("Получение сообщений через Telegram API")
        messages = await telegram_client.get_messages_for_period(chat_id, days)
        
        return messages
    
    async def _analyze_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ сообщений и генерация тем"""
        logger.info("Начинаем анализ сообщений")
        
        # Извлекаем тексты сообщений
        texts = [msg["normalized_text"] for msg in messages if msg.get("normalized_text")]
        
        if not texts:
            return {"topics": [], "cluster_count": 0}
        
        # Фильтруем и удаляем дубликаты
        filtered_texts = text_processor.filter_messages(texts)
        unique_texts = text_processor.remove_duplicates(filtered_texts)
        
        logger.info(f"Обработано {len(unique_texts)} уникальных сообщений")
        
        if len(unique_texts) < 10:
            return {"topics": [], "cluster_count": 0}
        
        # Генерируем эмбеддинги
        logger.info("Генерация эмбеддингов")
        embeddings = embeddings_generator.generate_embeddings(unique_texts)
        
        if len(embeddings) == 0:
            return {"topics": [], "cluster_count": 0}
        
        # Кластеризация
        logger.info("Кластеризация сообщений")
        cluster_data = message_clusterer.cluster_messages(embeddings, unique_texts)
        
        if not cluster_data["clusters"]:
            return {"topics": [], "cluster_count": 0}
        
        # Генерация тем
        logger.info("Генерация тем")
        topics = llama_generator.generate_topics(cluster_data["clusters"])
        
        # Валидация тем
        valid_topics = llama_generator.validate_topics(topics)
        
        # Статистика кластеризации
        stats = message_clusterer.get_clustering_stats(cluster_data)
        
        return {
            "topics": valid_topics,
            "cluster_count": len(cluster_data["clusters"]),
            "stats": stats
        }
    
    async def get_chat_statistics(self, chat_id: int, days: int) -> Dict[str, Any]:
        """Получение статистики чата"""
        try:
            # Информация о чате
            chat_info = await telegram_client.get_chat_info(chat_id)
            
            # Количество сообщений
            message_count = await telegram_client.get_message_count(chat_id, days)
            
            # Участники чата
            participants = await telegram_client.get_chat_participants(chat_id)
            
            return {
                "chat_info": chat_info,
                "message_count": message_count,
                "participants_count": len(participants),
                "period_days": days,
                "min_required": self.min_messages,
                "has_sufficient_data": message_count >= self.min_messages
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики чата: {e}")
            return {"error": "Не удалось получить статистику"}
    
    def cleanup_old_data(self):
        """Очистка старых данных"""
        try:
            db.cleanup_old_messages(days=30)
            db.cleanup_old_cache(days=7)
            logger.info("Очистка старых данных завершена")
        except Exception as e:
            logger.error(f"Ошибка очистки старых данных: {e}")


# Глобальный экземпляр анализатора
topics_analyzer = TopicsAnalyzer()
