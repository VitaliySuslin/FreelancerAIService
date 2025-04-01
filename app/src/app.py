import os
import shutil
from typing import Optional
import kagglehub

from app.src.integrations.gpt import GPTClient
from app.src.config import settings


class FreelancerAnalyzer:
    def __init__(self):
        self.client = GPTClient()
        self._configure_kaggle()
        self._ensure_dataset()

    def _configure_kaggle(self) -> None:
        os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
        os.environ["KAGGLE_KEY"] = settings.kaggle_key

    def _ensure_dataset(self) -> None:
        if not os.path.exists(settings.dataset_path):
            print("Downloading dataset...")
            download_path = kagglehub.dataset_download("shohinurpervezshohan/freelancer-earnings-and-job-trends")
            print(f"Dataset downloaded to: {download_path}")
            
            # Перемещаем файлы в целевую директорию
            self._move_dataset_files(download_path, settings.dataset_path)

    def _move_dataset_files(self, src: str, dst: str) -> None:
        """Move dataset files from download location to target directory"""
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dst_path = os.path.join(dst, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.move(src_path, dst_path)
        
        print(f"Files moved to: {dst}")

    def _find_data_file(self) -> str:
        """Find first data file in dataset directory"""
        supported_extensions = (".csv", ".xlsx", ".xls", ".json")
        for root, _, files in os.walk(settings.dataset_path):
            for file in files:
                if file.endswith(supported_extensions):
                    return os.path.join(root, file)
        raise FileNotFoundError(
            f"No data file found in {settings.dataset_path}. "
            "Please check the dataset contents."
        )

    def load_data(self) -> None:
        try:
            data_path = self._find_data_file()
            print(f"Loading data from: {data_path}")
            self.client.load_documents(data_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)

    def ask_question(self, question: str) -> Optional[str]:
        return self.client.query_documents(question)
        