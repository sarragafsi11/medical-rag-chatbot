"""Configuration centralisée du projet, chargée depuis les variables d'environnement / .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="mistral:7b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"))
    chroma_collection_name: str = Field(default="medical_documents")

    # Découpage des documents
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # Récupération
    retriever_top_k: int = Field(default=4, ge=1)

    # Stockage
    data_dir: Path = Field(default=Path("./data/uploads"))

    # LLM
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    def ensure_dirs(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas encore."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()