"""Pipeline RAG avec LangChain, ChromaDB et Mistral via Ollama."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import settings
from src.vector_store import get_vector_store


MEDICAL_DISCLAIMER = (
    "Ceci est une information générale issue des documents fournis, et non un avis "
    "médical. Consultez un professionnel de santé pour toute décision clinique."
)


SYSTEM_PROMPT = """Tu es un assistant qui répond UNIQUEMENT à partir des documents médicaux
fournis par l'utilisateur.

Règles strictes :
- Utilise uniquement les informations présentes dans le contexte.
- Si l'information demandée n'est pas présente dans le contexte, réponds :
  "Les documents fournis ne contiennent pas d'informations suffisantes pour répondre à cette question."
- N'utilise jamais tes connaissances générales pour compléter la réponse.
- Ne donne jamais de diagnostic ni de prescription.
- Ne crée jamais de fausses informations.
- Ne mentionne aucune source dans ta réponse.
- Réponds en français.

Contexte :
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm() -> ChatOllama:
    """Instancie Mistral-7B via Ollama en mode CPU."""
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=2048,
        num_gpu=0,
    )


def format_docs(docs: list) -> str:
    """Formate les documents récupérés."""
    formatted = []

    for doc in docs:
        formatted.append(doc.page_content)

    return "\n\n---\n\n".join(formatted)


def retrieve_relevant_documents(question: str, k: int | None = None) -> list:
    """Récupère uniquement les documents suffisamment pertinents."""

    store = get_vector_store()

    results = store.similarity_search_with_relevance_scores(
        question,
        k=k or settings.retriever_top_k,
    )

    # Seuil de pertinence
    relevance_threshold = 0.35

    relevant_docs = []

    for doc, score in results:
        if score >= relevance_threshold:
            relevant_docs.append(doc)

    return relevant_docs


def deduplicate_sources(docs: list) -> list:
    """Supprime les sources identiques."""

    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        key = (source, page)

        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "source": source,
                    "page": page,
                }
            )

    return sources


def answer_question(question: str, k: int | None = None) -> dict:
    """Répond à une question avec le contexte pertinent."""

    # Recherche des documents pertinents
    source_docs = retrieve_relevant_documents(question, k=k)

    # Aucun document pertinent
    if not source_docs:
        return {
            "answer": (
                "Les documents fournis ne contiennent pas d'informations suffisantes "
                "pour répondre à cette question."
            ),
            "sources": [],
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # Création du contexte
    context = format_docs(source_docs)

    # LLM
    llm = get_llm()

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Si le LLM indique que l'information est absente
    unavailable = (
        "Les documents fournis ne contiennent pas d'informations suffisantes"
    )

    if unavailable.lower() in answer.lower():
        sources = []
    else:
        sources = deduplicate_sources(source_docs)

    return {
        "answer": answer,
        "sources": sources,
        "disclaimer": MEDICAL_DISCLAIMER,
    }