"""Interface Streamlit du chatbot RAG pour documents médicaux."""

import tempfile
from pathlib import Path

import streamlit as st

from src.chunking import chunk_documents
from src.config import settings
from src.pdf_extractor import extract_pdf_pages
from src.rag_pipeline import MEDICAL_DISCLAIMER, answer_question
from src.vector_store import index_documents

st.set_page_config(page_title="Assistant Documents Médicaux (RAG)", page_icon="🩺", layout="wide")

settings.ensure_dirs()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

st.title("🩺 Assistant RAG — Documents médicaux")
st.caption(
    "Cet outil répond à des questions à partir des documents PDF que vous téléversez. "
    "Il ne remplace pas un avis médical."
)

# --- Barre latérale : gestion des documents ---
with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Téléverser un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Indexer les documents", type="primary"):
        with st.spinner("Extraction, découpage et indexation en cours..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = Path(tmp.name)

                pages = extract_pdf_pages(tmp_path)
                for p in pages:
                    p.source = uploaded_file.name  # nom lisible pour la citation

                chunks = chunk_documents(pages)
                total_chunks += index_documents(chunks)

                if uploaded_file.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(uploaded_file.name)

                tmp_path.unlink(missing_ok=True)

        st.success(f"{total_chunks} extraits indexés depuis {len(uploaded_files)} fichier(s).")

    if st.session_state.indexed_files:
        st.subheader("Documents indexés")
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")

    st.divider()
    top_k = st.slider(
        "Nombre d'extraits utilisés (top-k)",
        min_value=2,
        max_value=10,
        value=settings.retriever_top_k,
    )

st.divider()

# --- Historique de conversation ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} — page {s['page']}")

# --- Zone de saisie ---
question = st.chat_input("Posez une question sur vos documents médicaux…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                result = answer_question(question, k=top_k)
                st.markdown(result["answer"])
                st.caption(MEDICAL_DISCLAIMER)
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"- {s['source']} — page {s['page']}")
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Erreur lors de la génération de la réponse : {e}")