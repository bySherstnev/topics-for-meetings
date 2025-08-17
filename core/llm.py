import os
import json
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger


class LlamaGenerator:
    """Класс для работы с Llama 3.1 8B"""
    
    def __init__(self):
        self.model_path = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
        self.quantization = os.getenv("QUANTIZATION", "4bit")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Llama"""
        try:
            logger.info(f"Загрузка модели Llama: {self.model_path}")
            
            # Настройка квантизации
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                quantization_config = None
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загрузка модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Модель Llama загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Llama: {e}")
            raise
    
    def _create_prompt(self, clusters: List[Dict[str, Any]]) -> str:
        """Создание промпта для генерации тем"""
        prompt = """Ты - помощник для генерации тем митапов на основе анализа групповых чатов.

Проанализируй следующие кластеры сообщений и сгенерируй 3-7 релевантных тем для обсуждения на митапе.

Кластеры сообщений:
"""
        
        for i, cluster in enumerate(clusters[:7]):  # Ограничиваем до 7 кластеров
            keywords = ", ".join(cluster["keywords"][:5])
            size = cluster["size"]
            
            # Добавляем примеры сообщений
            examples = cluster["representative_messages"][:3]
            examples_text = "\n".join([f"- {msg[:100]}..." if len(msg) > 100 else f"- {msg}" for msg in examples])
            
            prompt += f"""
Кластер {i+1} ({size} сообщений):
Ключевые слова: {keywords}
Примеры сообщений:
{examples_text}
"""
        
        prompt += """
Задача: Сгенерируй JSON-массив с темами для митапа в следующем формате:
[
  {
    "title": "Краткий заголовок (2-5 слов)",
    "summary": "Описание темы в одном предложении"
  }
]

Требования:
- 3-7 тем
- Заголовки на русском языке, 2-5 слов
- Описания на русском языке, одно предложение
- Темы должны быть актуальными и интересными для обсуждения
- Избегай дублирования тем

JSON ответ:
"""
        
        return prompt
    
    def generate_topics(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Генерация тем на основе кластеров"""
        if not clusters:
            return []
        
        try:
            # Создаем промпт
            prompt = self._create_prompt(clusters)
            
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование ответа
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем JSON из ответа
            json_text = self._extract_json(generated_text)
            
            if json_text:
                topics = json.loads(json_text)
                logger.info(f"Сгенерировано {len(topics)} тем")
                return topics
            else:
                logger.warning("Не удалось извлечь JSON из ответа модели")
                return self._generate_fallback_topics(clusters)
                
        except Exception as e:
            logger.error(f"Ошибка генерации тем: {e}")
            return self._generate_fallback_topics(clusters)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Извлечение JSON из текста ответа"""
        try:
            # Ищем JSON в тексте
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx + 1]
                # Проверяем валидность JSON
                json.loads(json_text)
                return json_text
            
            return None
        except json.JSONDecodeError:
            return None
    
    def _generate_fallback_topics(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Генерация резервных тем на основе ключевых слов"""
        topics = []
        
        for i, cluster in enumerate(clusters[:5]):  # Максимум 5 тем
            keywords = cluster["keywords"][:3]
            if keywords:
                title = " ".join(keywords[:2]).title()
                summary = f"Обсуждение вопросов, связанных с {', '.join(keywords)}"
                
                topics.append({
                    "title": title,
                    "summary": summary
                })
        
        return topics
    
    def validate_topics(self, topics: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Валидация и очистка тем"""
        valid_topics = []
        
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            
            title = topic.get("title", "").strip()
            summary = topic.get("summary", "").strip()
            
            # Проверяем требования
            if (title and summary and 
                len(title.split()) <= 5 and 
                len(summary) > 10 and 
                len(summary) < 200):
                
                valid_topics.append({
                    "title": title,
                    "summary": summary
                })
        
        # Удаляем дубликаты
        unique_topics = []
        seen_titles = set()
        
        for topic in valid_topics:
            title_lower = topic["title"].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_topics.append(topic)
        
        return unique_topics[:7]  # Ограничиваем до 7 тем
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_path": self.model_path,
            "quantization": self.quantization,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "status": "loaded"
        }


# Глобальный экземпляр генератора
llama_generator = LlamaGenerator()
