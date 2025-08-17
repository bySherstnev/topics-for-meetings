from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from .messages import BUTTON_REGENERATE, BUTTON_STATS, BUTTON_HELP


def get_topics_keyboard(period: str = "7d") -> InlineKeyboardMarkup:
    """Клавиатура для результатов анализа тем"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text=BUTTON_REGENERATE,
                callback_data=f"regenerate:{period}"
            )
        ],
        [
            InlineKeyboardButton(
                text=BUTTON_STATS,
                callback_data=f"stats:{period}"
            ),
            InlineKeyboardButton(
                text=BUTTON_HELP,
                callback_data="help"
            )
        ]
    ])
    return keyboard


def get_stats_keyboard(period: str = "7d") -> InlineKeyboardMarkup:
    """Клавиатура для статистики"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🔙 Назад к темам",
                callback_data=f"back_to_topics:{period}"
            )
        ],
        [
            InlineKeyboardButton(
                text=BUTTON_HELP,
                callback_data="help"
            )
        ]
    ])
    return keyboard


def get_help_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура для помощи"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🔙 Назад",
                callback_data="back"
            )
        ]
    ])
    return keyboard


def get_period_selection_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура для выбора периода"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text="1 день",
                callback_data="period:1d"
            ),
            InlineKeyboardButton(
                text="3 дня",
                callback_data="period:3d"
            )
        ],
        [
            InlineKeyboardButton(
                text="7 дней",
                callback_data="period:7d"
            ),
            InlineKeyboardButton(
                text="14 дней",
                callback_data="period:14d"
            )
        ],
        [
            InlineKeyboardButton(
                text="❌ Отмена",
                callback_data="cancel"
            )
        ]
    ])
    return keyboard
