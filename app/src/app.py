import os
import pandas as pd
from typing import Optional

from app.src.integrations.gpt import GPTClient
from app.src.config import settings


class FreelancerAnalyzer:
    """
    Класс для анализа данных фрилансеров.

    Атрибуты:
        client (GPTClient): Клиент для взаимодействия с GPT-моделью.
        data_text (Optional[str]): Данные в текстовом формате.
    """

    def __init__(self) -> None:
        """Инициализация FreelancerAnalyzer."""
        self.client = GPTClient()
        self.data_text = None

    def load_data(self) -> None:
        """
        Загрузка данных из файла, указанного в настройках.

        Преобразует данные в текстовый формат для дальнейшего анализа.
        """
        try:
            file_path = settings.dataset_path
            data = pd.read_csv(file_path)
            self.data_text = data.to_string(index=False)
            print("Данные успешно загружены.")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            self.data_text = None

    def ask_question(self, question: str) -> Optional[str]:
        """
        Ответ на вопрос с использованием данных.

        Аргументы:
            question (str): Вопрос, на который нужно ответить.

        Возвращает:
            Optional[str]: Ответ модели или None, если произошла ошибка.
        """
        if not self.data_text:
            return "Данные не загружены. Сначала вызовите load_data()."

        prompt = f"""
        Данные для анализа:
        '{str(self.data_text)}'

        Вопрос: {question}
        Отвечай всегда на русском языке и используй только предоставленные данные.
        """

        try:
            response = self.client.query_documents(prompt, max_tokens=500)
            if response is None:
                return "No answer from the model."
            return response
        except Exception as e:
            print(f"Ошибка при отправке запроса модели: {e}")
            return None