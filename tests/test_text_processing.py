import pytest
from utils.text_processing import text_processor


def test_normalize_text():
    """Тест нормализации текста"""
    # Тест с обычным текстом
    text = "Привет! Как дела? 😊"
    normalized = text_processor.normalize_text(text)
    assert "привет" in normalized
    assert "😊" not in normalized  # emoji должны быть удалены
    
    # Тест с URL
    text_with_url = "Посмотрите на https://example.com этот сайт"
    normalized = text_processor.normalize_text(text_with_url)
    assert "https://example.com" not in normalized
    
    # Тест с упоминаниями
    text_with_mentions = "Привет @username и #hashtag"
    normalized = text_processor.normalize_text(text_with_mentions)
    assert "@username" not in normalized
    assert "#hashtag" not in normalized
    
    # Тест с коротким текстом
    short_text = "Hi"
    normalized = text_processor.normalize_text(short_text)
    assert normalized == ""  # Слишком короткий текст должен быть отфильтрован


def test_lemmatize_text():
    """Тест лемматизации текста"""
    text = "программисты программируют программы"
    lemmatized = text_processor.lemmatize_text(text)
    # Проверяем, что слова приведены к нормальной форме
    assert "программист" in lemmatized or "программа" in lemmatized


def test_extract_keywords():
    """Тест извлечения ключевых слов"""
    texts = [
        "программирование на python",
        "python разработка",
        "код на python"
    ]
    keywords = text_processor.extract_keywords(texts, max_keywords=5)
    assert len(keywords) > 0
    assert "python" in [kw.lower() for kw in keywords]


def test_calculate_similarity():
    """Тест вычисления схожести"""
    text1 = "программирование на python"
    text2 = "разработка на python"
    similarity = text_processor.calculate_similarity(text1, text2)
    assert 0 <= similarity <= 1
    
    # Одинаковые тексты должны иметь максимальную схожесть
    same_similarity = text_processor.calculate_similarity(text1, text1)
    assert same_similarity > 0.8


def test_remove_duplicates():
    """Тест удаления дубликатов"""
    texts = [
        "программирование на python",
        "разработка на python",
        "код на python",
        "программирование на python"  # дубликат
    ]
    unique_texts = text_processor.remove_duplicates(texts, similarity_threshold=0.7)
    assert len(unique_texts) < len(texts)  # Дубликаты должны быть удалены


def test_filter_messages():
    """Тест фильтрации сообщений"""
    messages = [
        "Короткое",  # слишком короткое
        "Нормальное сообщение с достаточным количеством слов",
        "",  # пустое
        "Еще одно хорошее сообщение для тестирования"
    ]
    filtered = text_processor.filter_messages(messages)
    assert len(filtered) == 2  # Только нормальные сообщения должны остаться
