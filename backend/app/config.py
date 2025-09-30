import os
from typing import Optional
from pydantic_settings import BaseSettings  # ✅ Updated import

class Settings(BaseSettings):
    # === API Configuration ===
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Insulyn AI"

    # === ML Model ===
    MODEL_PATH: str = "./data/diabetes_model.pkl"

    # === Groq LLM Configuration ===
    GROQ_API_KEY: str  # required (no default)
    LLM_MODEL_NAME: str  # required (no default)
    LLM_TEMPERATURE: float = 0.0  # default set to 0.0

    # === Application Environment ===
    ENVIRONMENT: str = "development"  # development | production
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # === Server Settings ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    class Config:
        env_file = ".env"        # Load environment variables from .env file
        env_file_encoding = "utf-8"
        case_sensitive = True    # Environment variable names are case-sensitive

# ✅ Instantiate settings
settings = Settings()
