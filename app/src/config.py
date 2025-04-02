from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings configuration.

    Loads settings from environment variables with fallback to default values.
    """

    gpt_model: str = "gpt-4o-mini"
    """Default GPT model to use for text generation."""

    dataset_path: str = "data/freelancer-earnings-and-job-trends/freelancer_earnings_bd.csv"
    """Path to store and load the dataset."""

    kaggle_username: str = "ЗАМЕНИТЬ"
    """Kaggle username for dataset download."""

    kaggle_key: str = "ЗАМЕНИТЬ"
    """Kaggle API key for authentication."""

    class Config:
        """
        Pydantic configuration for settings.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()