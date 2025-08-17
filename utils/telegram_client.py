import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from telethon import TelegramClient, events
from telethon.tl.types import User, Chat, Channel, Message as TelegramMessage
from telethon.errors import ChatAdminRequiredError, FloodWaitError
from loguru import logger

from data.database import db
from utils.text_processing import text_processor


class TelegramMessageClient:
    """Класс для работы с Telegram API через Telethon"""
    
    def __init__(self):
        self.api_id = os.getenv("API_ID")
        self.api_hash = os.getenv("API_HASH")
        self.phone_number = os.getenv("PHONE_NUMBER")
        
        if not all([self.api_id, self.api_hash, self.phone_number]):
            raise ValueError("Необходимо указать API_ID, API_HASH и PHONE_NUMBER в .env")
        
        self.client = TelegramClient('bot_session', self.api_id, self.api_hash)
        self._is_connected = False
    
    async def connect(self):
        """Подключение к Telegram"""
        if not self._is_connected:
            await self.client.start(phone=self.phone_number)
            self._is_connected = True
            logger.info("Telegram клиент подключен")
    
    async def disconnect(self):
        """Отключение от Telegram"""
        if self._is_connected:
            await self.client.disconnect()
            self._is_connected = False
            logger.info("Telegram клиент отключен")
    
    async def get_chat_info(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Получить информацию о чате"""
        try:
            await self.connect()
            entity = await self.client.get_entity(chat_id)
            
            if isinstance(entity, (Chat, Channel)):
                return {
                    'id': entity.id,
                    'title': getattr(entity, 'title', ''),
                    'type': 'channel' if isinstance(entity, Channel) else 'group',
                    'participants_count': getattr(entity, 'participants_count', 0)
                }
            return None
        except Exception as e:
            logger.error(f"Ошибка получения информации о чате {chat_id}: {e}")
            return None
    
    async def check_admin_rights(self, chat_id: int, bot_id: int) -> bool:
        """Проверить права администратора у бота"""
        try:
            await self.connect()
            entity = await self.client.get_entity(chat_id)
            
            if isinstance(entity, (Chat, Channel)):
                # Получаем информацию о боте в чате
                participant = await self.client.get_participants(
                    entity, 
                    filter=lambda x: x.id == bot_id
                )
                
                if participant:
                    # Проверяем права администратора
                    return hasattr(participant[0], 'admin_rights') and participant[0].admin_rights
                
            return False
        except Exception as e:
            logger.error(f"Ошибка проверки прав администратора: {e}")
            return False
    
    async def get_messages_for_period(self, chat_id: int, days: int) -> List[Dict[str, Any]]:
        """Получить сообщения за указанный период"""
        try:
            await self.connect()
            entity = await self.client.get_entity(chat_id)
            
            # Вычисляем дату начала периода
            start_date = datetime.now() - timedelta(days=days)
            
            messages = []
            async for message in self.client.iter_messages(entity, offset_date=start_date):
                # Пропускаем служебные сообщения
                if not message.text or message.text.strip() == "":
                    continue
                
                # Пропускаем сообщения от ботов
                if message.sender and hasattr(message.sender, 'bot') and message.sender.bot:
                    continue
                
                # Нормализуем текст
                normalized_text = text_processor.normalize_text(message.text)
                if not normalized_text or len(normalized_text.split()) < 3:
                    continue
                
                message_data = {
                    'chat_id': chat_id,
                    'message_id': message.id,
                    'user_id': message.sender_id if message.sender_id else 0,
                    'date': message.date,
                    'text': message.text,
                    'normalized_text': normalized_text,
                    'is_bot': message.sender.bot if message.sender and hasattr(message.sender, 'bot') else False
                }
                
                messages.append(message_data)
                
                # Сохраняем в базу данных
                db.save_message(
                    chat_id=message_data['chat_id'],
                    message_id=message_data['message_id'],
                    user_id=message_data['user_id'],
                    date=message_data['date'],
                    text=message_data['text'],
                    is_bot=message_data['is_bot']
                )
            
            logger.info(f"Получено {len(messages)} сообщений из чата {chat_id} за {days} дней")
            return messages
            
        except ChatAdminRequiredError:
            logger.error(f"Нет прав администратора для доступа к истории чата {chat_id}")
            raise
        except FloodWaitError as e:
            logger.warning(f"Превышен лимит запросов, ожидание {e.seconds} секунд")
            await asyncio.sleep(e.seconds)
            return await self.get_messages_for_period(chat_id, days)
        except Exception as e:
            logger.error(f"Ошибка получения сообщений из чата {chat_id}: {e}")
            return []
    
    async def get_message_count(self, chat_id: int, days: int) -> int:
        """Получить количество сообщений за период"""
        try:
            await self.connect()
            entity = await self.client.get_entity(chat_id)
            
            start_date = datetime.now() - timedelta(days=days)
            count = 0
            
            async for message in self.client.iter_messages(entity, offset_date=start_date):
                if message.text and message.text.strip() and not message.sender.bot:
                    count += 1
            
            return count
        except Exception as e:
            logger.error(f"Ошибка подсчета сообщений: {e}")
            return 0
    
    async def get_chat_participants(self, chat_id: int) -> List[Dict[str, Any]]:
        """Получить список участников чата"""
        try:
            await self.connect()
            entity = await self.client.get_entity(chat_id)
            
            participants = []
            async for participant in self.client.iter_participants(entity):
                participants.append({
                    'id': participant.id,
                    'username': participant.username,
                    'first_name': participant.first_name,
                    'last_name': participant.last_name,
                    'is_bot': participant.bot if hasattr(participant, 'bot') else False
                })
            
            return participants
        except Exception as e:
            logger.error(f"Ошибка получения участников чата: {e}")
            return []


# Глобальный экземпляр клиента
telegram_client = TelegramMessageClient()
