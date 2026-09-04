"""All settings live here, read from environment variables / .env with defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

try:  # .env support is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv not installed
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR") or ROOT / "data")
STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR") or ROOT / "storage")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path = DATA_DIR
    storage_dir: Path = STORAGE_DIR

    chunk_size: int = _int("RAG_CHUNK_SIZE", 800)
    chunk_overlap: int = _int("RAG_CHUNK_OVERLAP", 150)
    top_k: int = _int("RAG_TOP_K", 4)

    embedder: str = os.getenv("RAG_EMBEDDER", "auto")
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "auto")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.storage_dir = Path(self.storage_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
