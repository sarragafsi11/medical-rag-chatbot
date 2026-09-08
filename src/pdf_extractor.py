"""Extraction du texte des fichiers PDF, page par page, pour permettre la citation des sources."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ExtractedPage:
    """Représente le texte extrait d'une page de PDF."""

    source: str
    page_number: int
    text: str


def extract_pdf_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extrait le texte de chaque page d'un PDF.

    Args:
        pdf_path: chemin vers le fichier PDF.

    Returns:
        Une liste d'ExtractedPage (une entrée par page non vide).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages: list[ExtractedPage] = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = " ".join(raw_text.split())
        if cleaned:
            pages.append(
                ExtractedPage(source=pdf_path.name, page_number=i, text=cleaned)
            )

    return pages


def extract_multiple_pdfs(pdf_paths: list[str | Path]) -> list[ExtractedPage]:
    """Extrait le texte de plusieurs PDF et concatène les résultats."""
    all_pages: list[ExtractedPage] = []
    for path in pdf_paths:
        all_pages.extend(extract_pdf_pages(path))
    return all_pages