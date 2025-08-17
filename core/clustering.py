import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
from loguru import logger

from utils.text_processing import text_processor


class MessageClusterer:
    """Класс для кластеризации сообщений"""
    
    def __init__(self, method: str = "hdbscan", min_cluster_size: int = 8):
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.scaler = StandardScaler()
    
    def cluster_messages(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Кластеризация сообщений по эмбеддингам"""
        if len(embeddings) == 0 or len(texts) == 0:
            return {"clusters": [], "cluster_count": 0, "noise_count": 0}
        
        try:
            # Нормализация эмбеддингов
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            if self.method == "hdbscan":
                clusters = self._hdbscan_clustering(embeddings_scaled)
            elif self.method == "kmeans":
                clusters = self._kmeans_clustering(embeddings_scaled)
            else:
                raise ValueError(f"Неизвестный метод кластеризации: {self.method}")
            
            # Формируем результат
            cluster_data = self._process_clusters(clusters, texts, embeddings)
            
            logger.info(f"Кластеризация завершена: {len(cluster_data['clusters'])} кластеров")
            return cluster_data
            
        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")
            return {"clusters": [], "cluster_count": 0, "noise_count": 0}
    
    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Кластеризация с помощью HDBSCAN"""
        # Автоматический подбор параметров
        min_cluster_size = min(self.min_cluster_size, max(3, len(embeddings) // 20))
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        return clusterer.fit_predict(embeddings)
    
    def _kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Кластеризация с помощью K-means"""
        # Автоматический подбор количества кластеров
        n_clusters = min(
            max(3, len(embeddings) // 20),  # Минимум 3 кластера
            min(12, len(embeddings) // 5)   # Максимум 12 кластеров
        )
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        return kmeans.fit_predict(embeddings)
    
    def _process_clusters(self, cluster_labels: np.ndarray, texts: List[str], 
                         embeddings: np.ndarray) -> Dict[str, Any]:
        """Обработка результатов кластеризации"""
        clusters = []
        unique_clusters = set(cluster_labels)
        
        # Удаляем шумовые точки (метка -1 для HDBSCAN)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cluster_id in unique_clusters:
            # Индексы сообщений в кластере
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) < 3:  # Пропускаем слишком маленькие кластеры
                continue
            
            # Тексты кластера
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # Эмбеддинги кластера
            cluster_embeddings = embeddings[cluster_indices]
            
            # Центроид кластера
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Ключевые слова кластера
            keywords = text_processor.extract_keywords(cluster_texts, max_keywords=8)
            
            # Репрезентативные сообщения (ближайшие к центроиду)
            representative_messages = self._get_representative_messages(
                cluster_texts, cluster_embeddings, centroid, max_messages=5
            )
            
            cluster_data = {
                "id": int(cluster_id),
                "size": len(cluster_indices),
                "texts": cluster_texts,
                "keywords": keywords,
                "representative_messages": representative_messages,
                "centroid": centroid.tolist()
            }
            
            clusters.append(cluster_data)
        
        # Сортируем кластеры по размеру (от большего к меньшему)
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        # Пересчитываем ID кластеров
        for i, cluster in enumerate(clusters):
            cluster["id"] = i
        
        noise_count = len(np.where(cluster_labels == -1)[0])
        
        return {
            "clusters": clusters,
            "cluster_count": len(clusters),
            "noise_count": noise_count,
            "total_messages": len(texts)
        }
    
    def _get_representative_messages(self, texts: List[str], embeddings: np.ndarray, 
                                   centroid: np.ndarray, max_messages: int = 5) -> List[str]:
        """Получить репрезентативные сообщения кластера"""
        if len(texts) == 0:
            return []
        
        # Вычисляем расстояния до центроида
        distances = []
        for i, embedding in enumerate(embeddings):
            distance = np.linalg.norm(embedding - centroid)
            distances.append((i, distance))
        
        # Сортируем по расстоянию (ближайшие к центроиду)
        distances.sort(key=lambda x: x[1])
        
        # Выбираем репрезентативные сообщения
        representative_indices = [idx for idx, _ in distances[:max_messages]]
        representative_texts = [texts[idx] for idx in representative_indices]
        
        return representative_texts
    
    def get_cluster_summary(self, cluster: Dict[str, Any]) -> str:
        """Получить краткое описание кластера"""
        keywords = ", ".join(cluster["keywords"][:5])
        size = cluster["size"]
        
        # Выбираем самое короткое репрезентативное сообщение
        if cluster["representative_messages"]:
            shortest_message = min(cluster["representative_messages"], key=len)
            # Обрезаем до 100 символов
            if len(shortest_message) > 100:
                shortest_message = shortest_message[:97] + "..."
        else:
            shortest_message = "Нет репрезентативных сообщений"
        
        return f"Кластер {cluster['id']} ({size} сообщений): {keywords}. Пример: {shortest_message}"
    
    def get_clustering_stats(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
        """Получить статистику кластеризации"""
        clusters = cluster_data["clusters"]
        
        if not clusters:
            return {
                "cluster_count": 0,
                "avg_cluster_size": 0,
                "max_cluster_size": 0,
                "min_cluster_size": 0,
                "noise_percentage": 0
            }
        
        sizes = [cluster["size"] for cluster in clusters]
        
        return {
            "cluster_count": len(clusters),
            "avg_cluster_size": np.mean(sizes),
            "max_cluster_size": max(sizes),
            "min_cluster_size": min(sizes),
            "noise_percentage": (cluster_data["noise_count"] / cluster_data["total_messages"]) * 100
        }


# Глобальный экземпляр кластеризатора
message_clusterer = MessageClusterer()
