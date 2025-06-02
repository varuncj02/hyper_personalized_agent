from pydantic_settings import BaseSettings  # Changed from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database paths
    IMESSAGE_DB_PATH: str = os.path.expanduser("~/Library/Messages/chat.db")
    PERSONA_DB_PATH: str = "./data/persona_vectors.faiss"
    
    # API Keys (set in .env file)
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "PersonaAgent/1.0"
    
    # Vector Database Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    
    # Agent Settings
    MAX_CONTEXT_LENGTH: int = 2000
    RESPONSE_TEMPERATURE: float = 0.7
    
    # Real-time monitoring
    ENABLE_REALTIME: bool = False
    MONITOR_INTERVAL: int = 5  # seconds
    
    class Config:
        env_file = ".env"

settings = Settings()