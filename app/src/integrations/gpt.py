from typing import Optional
from g4f.client import Client

from app.src.config import settings


class G4FLLM:
    """
    Класс для взаимодействия с GPT-моделью.

    Атрибуты:
        _client (Client): Клиент для взаимодействия с GPT API.
    """

    def __init__(self):
        """Инициализация G4FLLM."""
        self._client = Client()

    def __call__(self, prompt: str, max_tokens: int = 500, **kwargs) -> Optional[str]:
        """
        Вызов модели для генерации ответа.

        Аргументы:
            prompt (str): Запрос для модели.
            max_tokens (int): Максимальное количество токенов в ответе.
            **kwargs: Дополнительные параметры.

        Возвращает:
            Optional[str]: Ответ модели или None, если произошла ошибка.
        """
        try:
            response = self._client.chat.completions.create(
                model=settings.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying the model: {e}")
            return None


class GPTClient:
    """
    Класс для взаимодействия с GPT-моделью и анализа данных.

    Атрибуты:
        model (str): Модель GPT, используемая для генерации ответов.
        llm (G4FLLM): Экземпляр класса G4FLLM для взаимодействия с моделью.
    """

    def __init__(self):
        """Инициализация GPTClient."""
        self.model = settings.gpt_model
        self.llm = G4FLLM()

    def query_documents(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Отправляет запрос в модель и возвращает ответ.

        Аргументы:
            prompt (str): Запрос для модели.
            max_tokens (int): Максимальное количество токенов в ответе.

        Возвращает:
            Optional[str]: Ответ модели или None, если произошла ошибка.
        """
        try:
            return self.llm(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Ошибка при отправке запроса модели: {e}")
            return None