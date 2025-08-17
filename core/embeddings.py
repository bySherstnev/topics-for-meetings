import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingsGenerator:
    """Класс для генерации эмбеддингов текста"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели эмбеддингов"""
        try:
            logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Модель эмбеддингов загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Генерация эмбеддингов для списка текстов"""
        if not texts:
            return np.array([])
        
        try:
            # Фильтруем пустые тексты
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            # Генерируем эмбеддинги
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            logger.info(f"Сгенерированы эмбеддинги для {len(valid_texts)} текстов")
            
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддингов: {e}")
            return np.array([])
    
    def generate_single_embedding(self, text: str) -> Optional[np.ndarray]:
        """Генерация эмбеддинга для одного текста"""
        if not text or not text.strip():
            return None
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Ошибка генерации эмбеддинга: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Вычисление косинусного сходства между эмбеддингами"""
        try:
            # Нормализация векторов
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Косинусное сходство
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Ошибка вычисления сходства: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, embeddings: np.ndarray, 
                         top_k: int = 5) -> List[tuple]:
        """Поиск наиболее похожих эмбеддингов"""
        try:
            if len(embeddings) == 0:
                return []
            
            # Вычисляем сходства
            similarities = []
            for i, embedding in enumerate(embeddings):
                similarity = self.calculate_similarity(query_embedding, embedding)
                similarities.append((i, similarity))
            
            # Сортируем по убыванию сходства
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Ошибка поиска похожих эмбеддингов: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """Получить информацию о модели"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "loaded"
        }


# Глобальный экземпляр генератора эмбеддингов
embeddings_generator = EmbeddingsGenerator()
