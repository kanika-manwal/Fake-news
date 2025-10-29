import os
class Config:
    APP_TITLE = "🔍 Oh The Website - Fake News Detector"
    APP_DESCRIPTION = "Advanced AI-Powered News Verification System"
    VERSION = "1.0.0"
    MODEL_TYPE = "logistic_regression"
    MAX_FEATURES = 5000
    MIN_CONFIDENCE = 0.6
    THEME_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    SUCCESS_COLOR = "#4ecdc4"
    ERROR_COLOR = "#ff6b6b"
    DATA_DIR = "data"
    MODELS_DIR = "models/saved_models"
    ASSETS_DIR = "assets"
    API_TIMEOUT = 30
    MAX_TEXT_LENGTH = 10000
    MIN_TEXT_LENGTH = 50
    ENABLE_ANALYTICS = True
    MAX_QUEUE_SIZE = 100
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
