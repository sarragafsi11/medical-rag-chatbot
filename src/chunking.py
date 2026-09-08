"""Découpage des pages extraites en chunks adaptés à l'indexation vectorielle."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.pdf_extractor import ExtractedPage


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """Construit le splitter avec les paramètres définis dans la configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def pages_to_documents(pages: list[ExtractedPage]) -> list[Document]:
    """Convertit les pages extraites en Document LangChain avec métadonnées."""
    return [
        Document(
            page_content=p.text,
            metadata={"source": p.source, "page": p.page_number},
        )
        for p in pages
    ]


def chunk_documents(pages: list[ExtractedPage]) -> list[Document]:
    """Découpe les pages en chunks, en conservant la source et le numéro de page.

    Chaque chunk reçoit en plus un identifiant unique (chunk_id).
    """
    splitter = build_text_splitter()
    documents = pages_to_documents(pages)
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks