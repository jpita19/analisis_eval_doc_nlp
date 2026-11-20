import os
from typing import List

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "API Evaluacion Docente & NLP")
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "pysentimiento/robertuito-sentiment-analysis")
        CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
        BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
        MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./model_cache")
        BACKEND_CORS_ORIGINS: List[str] = ["*"]

        class Config:
            case_sensitive = True
except ImportError:
    class Settings:
        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "API Evaluacion Docente & NLP")
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "pysentimiento/robertuito-sentiment-analysis")
        CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
        BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
        MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./model_cache")
        BACKEND_CORS_ORIGINS: List[str] = ["*"]

settings = Settings()
