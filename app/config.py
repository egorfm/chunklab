import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "ChunkLab"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 8501

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    DB_PATH: Path = BASE_DIR / "data" / "chunklab.db"

    # Qdrant Vector Store
    # If empty or ":memory:", qdrant uses in-memory or embedded local storage
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str | None = None
    QDRANT_STORAGE_PATH: Path = BASE_DIR / "data" / "qdrant_storage"

    # Default Embeddings
    # Defaulting to a high quality, fast model sentence-transformers/all-MiniLM-L6-v2
    # but supporting BAAI/bge-m3 or intfloat/multilingual-e5-base seamlessly
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # Evaluation defaults
    DEFAULT_TOP_K: int = 5
    IOU_THRESHOLD: float = 0.2  # Bounding box IoU threshold for relevance match

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
