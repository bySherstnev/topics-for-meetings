import asyncio
from typing import Dict, Any
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from asyncio_throttle import Throttler
from loguru import logger

from .messages import *
from .keyboards import *
from core.analyzer import topics_analyzer

# Роутер для обработчиков
router = Router()

# Рейт-лимитер (1 запрос в минуту на чат)
throttler = Throttler(rate_limit=1, period=60)


@router.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    await message.answer(WELCOME_MESSAGE, parse_mode="Markdown")


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    await message.answer(HELP_MESSAGE, parse_mode="Markdown")


@router.message(Command("topics"))
async def cmd_topics(message: Message):
    """Обработчик команды /topics"""
    # Проверяем, что сообщение из группы
    if message.chat.type == "private":
        await message.answer("❌ Эта команда работает только в группах.")
        return
    
    # Парсим аргументы
    args = message.text.split()
    period = args[1] if len(args) > 1 else "7d"
    
    # Проверяем валидность периода
    if period not in ["1d", "3d", "7d", "14d"]:
        await message.answer(ERROR_INVALID_PERIOD, parse_mode="Markdown")
        return
    
    # Проверяем рейт-лимит
    chat_id = message.chat.id
    if not await throttler.acquire(f"chat_{chat_id}"):
        await message.answer("⏳ Подождите минуту перед следующим запросом.")
        return
    
    # Отправляем сообщение о начале анализа
    status_msg = await message.answer(ANALYZING_MESSAGES)
    
    try:
        # Анализируем чат
        result = await topics_analyzer.analyze_chat(chat_id, period)
        
        if "error" in result:
            await _handle_error(message, result)
        else:
            await _send_topics_result(message, result, period)
            
    except Exception as e:
        logger.error(f"Ошибка обработки команды /topics: {e}")
        await message.answer(ERROR_UNKNOWN, parse_mode="Markdown")
    finally:
        # Удаляем статусное сообщение
        try:
            await status_msg.delete()
        except:
            pass


@router.callback_query(F.data.startswith("regenerate:"))
async def callback_regenerate(callback: CallbackQuery):
    """Обработчик кнопки 'Сгенерировать заново'"""
    period = callback.data.split(":")[1]
    chat_id = callback.message.chat.id
    
    # Проверяем рейт-лимит
    if not await throttler.acquire(f"chat_{chat_id}"):
        await callback.answer("⏳ Подождите минуту перед следующим запросом.", show_alert=True)
        return
    
    # Обновляем сообщение
    await callback.message.edit_text(ANALYZING_MESSAGES)
    
    try:
        # Анализируем чат с принудительной регенерацией
        result = await topics_analyzer.analyze_chat(chat_id, period, force_regenerate=True)
        
        if "error" in result:
            await _handle_error_callback(callback, result)
        else:
            await _update_topics_result(callback, result, period)
            
    except Exception as e:
        logger.error(f"Ошибка регенерации тем: {e}")
        await callback.message.edit_text(ERROR_UNKNOWN, parse_mode="Markdown")


@router.callback_query(F.data.startswith("stats:"))
async def callback_stats(callback: CallbackQuery):
    """Обработчик кнопки 'Показать статистику'"""
    period = callback.data.split(":")[1]
    chat_id = callback.message.chat.id
    
    try:
        # Получаем статистику
        days = topics_analyzer.parse_period(period)
        stats = await topics_analyzer.get_chat_statistics(chat_id, days)
        
        if "error" in stats:
            await callback.answer("❌ Не удалось получить статистику", show_alert=True)
            return
        
        # Формируем сообщение со статистикой
        stats_text = f"""
📊 **Статистика чата**

• **Название:** {stats.get('chat_info', {}).get('title', 'Неизвестно')}
• **Тип:** {stats.get('chat_info', {}).get('type', 'Неизвестно')}
• **Участников:** {stats['participants_count']}
• **Сообщений за {days} дней:** {stats['message_count']}
• **Минимум для анализа:** {stats['min_required']}
• **Достаточно данных:** {'✅' if stats['has_sufficient_data'] else '❌'}
        """
        
        keyboard = get_stats_keyboard(period)
        await callback.message.edit_text(stats_text, parse_mode="Markdown", reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        await callback.answer("❌ Ошибка получения статистики", show_alert=True)


@router.callback_query(F.data == "help")
async def callback_help(callback: CallbackQuery):
    """Обработчик кнопки 'Помощь'"""
    keyboard = get_help_keyboard()
    await callback.message.edit_text(HELP_MESSAGE, parse_mode="Markdown", reply_markup=keyboard)


@router.callback_query(F.data.startswith("back_to_topics:"))
async def callback_back_to_topics(callback: CallbackQuery):
    """Обработчик кнопки 'Назад к темам'"""
    period = callback.data.split(":")[1]
    chat_id = callback.message.chat.id
    
    try:
        # Получаем кэшированный результат
        result = await topics_analyzer.analyze_chat(chat_id, period)
        
        if "error" in result:
            await callback.answer("❌ Результаты больше не доступны", show_alert=True)
            return
        
        await _update_topics_result(callback, result, period)
        
    except Exception as e:
        logger.error(f"Ошибка возврата к темам: {e}")
        await callback.answer("❌ Ошибка загрузки тем", show_alert=True)


@router.callback_query(F.data == "back")
async def callback_back(callback: CallbackQuery):
    """Обработчик кнопки 'Назад'"""
    # Возвращаемся к предыдущему состоянию (пока просто закрываем)
    await callback.message.delete()


@router.callback_query(F.data == "cancel")
async def callback_cancel(callback: CallbackQuery):
    """Обработчик кнопки 'Отмена'"""
    await callback.message.delete()


async def _handle_error(message: Message, result: Dict[str, Any]):
    """Обработка ошибок"""
    error_type = result.get("error_type")
    
    if error_type == "admin_required":
        await message.answer(ERROR_ADMIN_REQUIRED, parse_mode="Markdown")
    elif error_type == "insufficient_data":
        error_text = ERROR_INSUFFICIENT_DATA.format(
            days=result.get("message_count", 0),
            count=result.get("message_count", 0),
            min_required=result.get("min_required", 50)
        )
        await message.answer(error_text, parse_mode="Markdown")
    elif error_type == "processing_error":
        await message.answer(ERROR_PROCESSING, parse_mode="Markdown")
    else:
        await message.answer(ERROR_UNKNOWN, parse_mode="Markdown")


async def _handle_error_callback(callback: CallbackQuery, result: Dict[str, Any]):
    """Обработка ошибок в callback"""
    error_type = result.get("error_type")
    
    if error_type == "admin_required":
        await callback.message.edit_text(ERROR_ADMIN_REQUIRED, parse_mode="Markdown")
    elif error_type == "insufficient_data":
        error_text = ERROR_INSUFFICIENT_DATA.format(
            days=result.get("message_count", 0),
            count=result.get("message_count", 0),
            min_required=result.get("min_required", 50)
        )
        await callback.message.edit_text(error_text, parse_mode="Markdown")
    elif error_type == "processing_error":
        await callback.message.edit_text(ERROR_PROCESSING, parse_mode="Markdown")
    else:
        await callback.message.edit_text(ERROR_UNKNOWN, parse_mode="Markdown")


async def _send_topics_result(message: Message, result: Dict[str, Any], period: str):
    """Отправка результатов анализа"""
    topics = result["topics"]
    
    if not topics:
        await message.answer("❌ Не удалось сгенерировать темы. Попробуйте позже.")
        return
    
    # Формируем текст с темами
    topics_text = TOPICS_HEADER
    
    for i, topic in enumerate(topics, 1):
        topics_text += TOPIC_FORMAT.format(
            number=i,
            title=topic["title"],
            summary=topic["summary"]
        ) + "\n\n"
    
    topics_text += TOPICS_FOOTER
    
    # Добавляем статистику
    if not result.get("cached", False):
        stats_text = STATS_MESSAGE.format(
            message_count=result["message_count"],
            cluster_count=result["cluster_count"],
            processing_time=result["processing_time"],
            cached="Да" if result.get("cached", False) else "Нет"
        )
        topics_text += stats_text
    
    # Отправляем результат с клавиатурой
    keyboard = get_topics_keyboard(period)
    await message.answer(topics_text, parse_mode="Markdown", reply_markup=keyboard)


async def _update_topics_result(callback: CallbackQuery, result: Dict[str, Any], period: str):
    """Обновление результатов анализа в callback"""
    topics = result["topics"]
    
    if not topics:
        await callback.message.edit_text("❌ Не удалось сгенерировать темы. Попробуйте позже.")
        return
    
    # Формируем текст с темами
    topics_text = TOPICS_HEADER
    
    for i, topic in enumerate(topics, 1):
        topics_text += TOPIC_FORMAT.format(
            number=i,
            title=topic["title"],
            summary=topic["summary"]
        ) + "\n\n"
    
    topics_text += TOPICS_FOOTER
    
    # Добавляем статистику
    if not result.get("cached", False):
        stats_text = STATS_MESSAGE.format(
            message_count=result["message_count"],
            cluster_count=result["cluster_count"],
            processing_time=result["processing_time"],
            cached="Да" if result.get("cached", False) else "Нет"
        )
        topics_text += stats_text
    
    # Обновляем сообщение с клавиатурой
    keyboard = get_topics_keyboard(period)
    await callback.message.edit_text(topics_text, parse_mode="Markdown", reply_markup=keyboard)
