import re
import emoji
from typing import List, Set
import pymorphy2
from loguru import logger

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()


class TextProcessor:
    """Класс для обработки и нормализации текста"""
    
    def __init__(self):
        # Слова-фильтры для исключения
        self.stop_words = {
            'это', 'так', 'как', 'то', 'все', 'он', 'она', 'оно', 'они', 'мы', 'вы',
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'мой', 'твой', 'наш',
            'ваш', 'его', 'ее', 'их', 'себя', 'кто', 'что', 'какой', 'какая', 'какое',
            'какие', 'чей', 'чья', 'чье', 'чьи', 'где', 'куда', 'откуда', 'когда',
            'почему', 'зачем', 'как', 'сколько', 'столько', 'тот', 'та', 'то', 'те',
            'этот', 'эта', 'это', 'эти', 'такой', 'такая', 'такое', 'такие', 'сей',
            'сия', 'сие', 'сии', 'всякий', 'всякая', 'всякое', 'всякие', 'каждый',
            'каждая', 'каждое', 'каждые', 'любой', 'любая', 'любое', 'любые', 'иной',
            'иная', 'иное', 'иные', 'другой', 'другая', 'другое', 'другие', 'сам',
            'сама', 'само', 'сами', 'самый', 'самая', 'самое', 'самые', 'весь',
            'вся', 'все', 'все', 'вот', 'вон', 'тут', 'там', 'здесь', 'туда',
            'сюда', 'оттуда', 'отсюда', 'теперь', 'сейчас', 'потом', 'тогда',
            'всегда', 'никогда', 'иногда', 'часто', 'редко', 'очень', 'слишком',
            'почти', 'примерно', 'точно', 'именно', 'просто', 'только', 'лишь',
            'даже', 'уже', 'еще', 'уже', 'все', 'все', 'все', 'все', 'все'
        }
    
    def normalize_text(self, text: str) -> str:
        """Нормализация текста сообщения"""
        if not text or len(text.strip()) < 10:
            return ""
        
        # Удаление emoji
        text = emoji.replace_emojis(text, replace='')
        
        # Удаление URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Удаление упоминаний и хештегов
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """Лемматизация текста"""
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # Пропускаем короткие слова и цифры
            if len(word) < 3 or word.isdigit():
                continue
            
            # Лемматизация
            parsed = morph.parse(word)
            if parsed:
                lemma = parsed[0].normal_form
                if lemma not in self.stop_words:
                    lemmatized_words.append(lemma)
        
        return ' '.join(lemmatized_words)
    
    def extract_keywords(self, texts: List[str], max_keywords: int = 10) -> List[str]:
        """Извлечение ключевых слов из списка текстов"""
        word_freq = {}
        
        for text in texts:
            normalized = self.normalize_text(text)
            if not normalized:
                continue
                
            lemmatized = self.lemmatize_text(normalized)
            words = lemmatized.split()
            
            for word in words:
                if len(word) >= 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Сортировка по частоте
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ ключевых слов
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def filter_messages(self, messages: List[str]) -> List[str]:
        """Фильтрация сообщений по качеству"""
        filtered = []
        
        for message in messages:
            normalized = self.normalize_text(message)
            if normalized and len(normalized.split()) >= 3:
                filtered.append(normalized)
        
        return filtered
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисление схожести между двумя текстами (Jaccard similarity)"""
        words1 = set(self.lemmatize_text(self.normalize_text(text1)).split())
        words2 = set(self.lemmatize_text(self.normalize_text(text2)).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def remove_duplicates(self, texts: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """Удаление дубликатов на основе схожести"""
        if not texts:
            return []
        
        unique_texts = [texts[0]]
        
        for text in texts[1:]:
            is_duplicate = False
            for unique_text in unique_texts:
                if self.calculate_similarity(text, unique_text) > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
        
        return unique_texts


# Глобальный экземпляр процессора
text_processor = TextProcessor()
