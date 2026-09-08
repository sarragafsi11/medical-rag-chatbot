"""Indexation et récupération vectorielle via ChromaDB, avec embeddings générés par Ollama."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from src.config import settings


def get_embedding_function() -> OllamaEmbeddings:
    """Retourne la fonction d'embeddings basée sur le modèle Ollama configuré."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vector_store() -> Chroma:
    """Retourne (ou crée) la collection ChromaDB persistante."""
    settings.ensure_dirs()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def index_documents(chunks: list[Document]) -> int:
    """Ajoute une liste de chunks à l'index vectoriel.

    Returns:
        Le nombre de chunks effectivement indexés.
    """
    if not chunks:
        return 0
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Construit un retriever de similarité sur la collection ChromaDB."""
    store = get_vector_store()
    return store.as_retriever(
        search_kwargs={"k": k or settings.retriever_top_k}
    )