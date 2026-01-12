"""
Configuration settings for Solar Forecasting API
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    APP_NAME: str = "Solar Forecasting API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    
    # CORS Settings - IMPORTANT: Restrict in production!
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:5000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # File Upload Settings
    UPLOAD_FOLDER: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
    MAX_SEQUENCE_IMAGES: int = 100
    MIN_SEQUENCE_IMAGES: int = 10
    
    # Model Paths
    CNN_MODEL_PATH: str = "models/best_efficientnet_model.pth"
    LSTM_MODEL_PATH: str = "models/best_lstm_model.pth"
    HYBRID_MODEL_PATH: str = "models/best_hybrid_model.pth"
    HYBRID_MODEL_PATH_FALLBACK: str = "models1/best_hybrid_model.pth"
    
    # Model Settings
    IMAGE_TARGET_SIZE: tuple = (240, 320)
    SEQUENCE_LENGTH: int = 20
    FORECAST_HORIZON: int = 4
    MIN_IRRADIANCE_SEQUENCE_LENGTH: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    
    # Gemini AI Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash" or "gemini-1.5-pro"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_TOKENS: int = 2048
    GEMINI_TIMEOUT: int = 30  # seconds
    
    # Directories
    STATIC_DIR: str = "static"
    TEMPLATES_DIR: str = "templates"
    MODELS_DIR: str = "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
