# 🩺 Medical RAG Chatbot

Un chatbot intelligent permettant d'interroger des **documents médicaux PDF en langage naturel**, basé sur l'approche **RAG (Retrieval-Augmented Generation)**.

Le projet fonctionne **entièrement en local**, sans dépendance à une API externe. Les documents sont indexés dans une base vectorielle, puis les passages pertinents sont récupérés afin de générer des réponses contextualisées avec **Mistral-7B via Ollama**.

---

## 📌 Description

Dans le domaine médical, les professionnels doivent consulter de nombreux documents tels que :

* 📄 Comptes-rendus médicaux
* 💊 Ordonnances
* 🏥 Protocoles cliniques
* 🧪 Résultats d'analyses
* 📚 Guides et recommandations médicales

Retrouver rapidement une information précise peut être long et fastidieux.

**Medical RAG Chatbot** propose une interface conversationnelle permettant de charger plusieurs documents PDF et de poser des questions en langage naturel.

Le système :

1. Extrait le texte des documents PDF.
2. Découpe le contenu en petits passages (*chunks*).
3. Transforme les passages en vecteurs (*embeddings*).
4. Stocke les vecteurs dans **ChromaDB**.
5. Recherche les passages les plus pertinents lorsqu'une question est posée.
6. Utilise ces passages comme contexte pour **Mistral-7B**.
7. Génère une réponse accompagnée des **sources utilisées**.

> 🔎 Le modèle est configuré pour répondre uniquement à partir du contenu récupéré dans les documents fournis. Lorsque l'information recherchée n'est pas présente dans les documents, le système l'indique explicitement.

---

## 🏗️ Architecture du pipeline

```text
                    ┌──────────────────┐
                    │    PDF Upload    │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  PDF Text Extraction│
                  │       (pypdf)       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      Chunking       │
                  │     (LangChain)     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Embeddings      │
                  │  nomic-embed-text   │
                  │      + Ollama       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      ChromaDB       │
                  │  Vector Database    │
                  └──────────┬──────────┘
                             │
                             │
              ┌──────────────▼──────────────┐
              │      User Question          │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Semantic Retrieval  │
                  │     Top-K chunks    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Relevance Filtering │
                  │      Score ≥ 0.35   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Context + Prompt  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      Mistral-7B     │
                  │       + Ollama      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Answer + Sources    │
                  │ File + Page Number  │
                  └─────────────────────┘
```

---

## 🧩 Structure du projet

```text
medical-rag-chatbot/
│
├── src/
│   ├── app.py
│   ├── pdf_extractor.py
│   ├── chunking.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── config.py
│
├── data/
│   └── chroma_db/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

### Rôle des principaux fichiers

| Fichier                | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| `src/app.py`           | Interface Streamlit et gestion du chatbot              |
| `src/pdf_extractor.py` | Extraction du texte des fichiers PDF                   |
| `src/chunking.py`      | Découpage du texte en chunks                           |
| `src/vector_store.py`  | Création et recherche dans ChromaDB                    |
| `src/rag_pipeline.py`  | Pipeline RAG : retrieval, filtrage et génération       |
| `src/config.py`        | Configuration de l'application                         |
| `.env.example`         | Exemple de configuration des variables d'environnement |

---

## 🛠️ Technologies utilisées

| Technologie           | Utilisation                    |
| --------------------- | ------------------------------ |
| **Python**            | Langage principal              |
| **Streamlit**         | Interface web interactive      |
| **LangChain**         | Orchestration du pipeline RAG  |
| **ChromaDB**          | Base de données vectorielle    |
| **Ollama**            | Exécution locale des modèles   |
| **Mistral-7B**        | Génération des réponses        |
| **nomic-embed-text**  | Génération des embeddings      |
| **pypdf**             | Extraction du texte des PDF    |
| **Pydantic Settings** | Gestion de la configuration    |
| **uv**                | Gestion des dépendances Python |

---

## ✨ Fonctionnalités

* 📂 Upload de plusieurs fichiers PDF
* 📑 Extraction automatique du contenu
* ✂️ Découpage intelligent en chunks avec chevauchement
* 🔎 Recherche sémantique dans les documents
* 🧠 Génération augmentée par récupération (**RAG**)
* 🤖 Mistral-7B exécuté localement
* 🔐 Aucun appel obligatoire à une API externe
* 💾 Stockage persistant avec ChromaDB
* 📚 Citation des sources utilisées
* 📄 Affichage du nom du fichier et du numéro de page
* 💬 Historique conversationnel dans la session
* 🎯 Paramètre `Top-K` configurable
* ⚠️ Avertissement médical affiché avec les réponses
* 🚫 Refus explicite lorsque l'information n'est pas disponible dans les documents

---

## ⚙️ Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir installé :

* **Python 3.10+**
* **Ollama**
* **uv**

### 1. Cloner le projet

```bash
git clone https://github.com/sarragafsi11/medical-rag-chatbot.git

cd medical-rag-chatbot
```

### 2. Installer les dépendances

```bash
uv sync
```

### 3. Télécharger les modèles Ollama

```bash
ollama pull mistral:7b
ollama pull nomic-embed-text
```

Vérifier les modèles installés :

```bash
ollama list
```

### 4. Configurer l'environnement

Copier le fichier `.env.example` :

```bash
cp .env.example .env
```

Sous Windows PowerShell, vous pouvez utiliser :

```powershell
Copy-Item .env.example .env
```

### 5. Lancer l'application

```bash
uv run streamlit run src/app.py
```

L'application sera accessible à :

```text
http://localhost:8501
```

---

## 🔧 Configuration

Les paramètres de l'application peuvent être configurés dans `.env` :

```env
OLLAMA_BASE_URL=http://localhost:11434

OLLAMA_LLM_MODEL=mistral:7b

OLLAMA_EMBEDDING_MODEL=nomic-embed-text

CHROMA_PERSIST_DIR=./data/chroma_db

CHROMA_COLLECTION_NAME=medical_documents

CHUNK_SIZE=1000

CHUNK_OVERLAP=150

RETRIEVER_TOP_K=4

LLM_TEMPERATURE=0.1
```

### Paramètres principaux

| Paramètre                | Description                         |
| ------------------------ | ----------------------------------- |
| `OLLAMA_BASE_URL`        | Adresse du serveur Ollama           |
| `OLLAMA_LLM_MODEL`       | Modèle utilisé pour la génération   |
| `OLLAMA_EMBEDDING_MODEL` | Modèle utilisé pour les embeddings  |
| `CHROMA_PERSIST_DIR`     | Emplacement de la base ChromaDB     |
| `CHUNK_SIZE`             | Taille maximale des chunks          |
| `CHUNK_OVERLAP`          | Chevauchement entre les chunks      |
| `RETRIEVER_TOP_K`        | Nombre de passages récupérés        |
| `LLM_TEMPERATURE`        | Température du modèle de génération |

---

## 🔄 Fonctionnement du RAG

Le système suit quatre étapes principales.

### 1. Ingestion des documents

Les fichiers PDF sont chargés par l'utilisateur puis leur contenu textuel est extrait **page par page** avec `pypdf`.

Chaque passage conserve ses métadonnées, notamment :

```text
filename
page_number
```

---

### 2. Vectorisation

Le texte est découpé en chunks avec :

```text
RecursiveCharacterTextSplitter
```

Chaque chunk est ensuite transformé en vecteur numérique grâce au modèle :

```text
nomic-embed-text
```

Les embeddings sont stockés dans :

```text
ChromaDB
```

---

### 3. Retrieval

Lorsque l'utilisateur pose une question, celle-ci est transformée en embedding.

ChromaDB recherche ensuite les passages les plus similaires sémantiquement.

Le système applique également un **seuil de pertinence** afin d'éviter d'utiliser des passages trop éloignés de la question.

---

### 4. Generation

Les passages récupérés sont transmis à **Mistral-7B** sous forme de contexte.

Le modèle génère ensuite une réponse basée sur ce contexte.

La réponse affiche également les sources utilisées :

```text
📄 document_medical.pdf — Page 4
📄 guide_clinique.pdf — Page 12
```

---

## 🛡️ Grounding et limitation des hallucinations

Une attention particulière est portée au **grounding** des réponses.

Le système est conçu pour :

* utiliser uniquement les passages récupérés ;
* éviter de compléter une information absente avec les connaissances générales du LLM ;
* signaler lorsqu'aucune information suffisamment pertinente n'est trouvée ;
* afficher les documents et pages utilisés pour produire la réponse.

Cette approche permet de rendre les réponses **plus traçables et vérifiables**.

---

## 🖥️ Interface utilisateur

L'application fournit une interface Streamlit permettant de :

1. 📤 Charger les documents PDF.
2. 🔄 Indexer automatiquement leur contenu.
3. 💬 Poser des questions en langage naturel.
4. 🔎 Consulter les passages utilisés.
5. 📚 Vérifier les sources.
6. ⚠️ Consulter l'avertissement médical associé à la réponse.

---

## 📊 Exemple de scénario

### Documents

```text
WHO_Guidelines.pdf
Clinical_Guide.pdf
Medical_Report.pdf
```

### Question

```text
Quels sont les facteurs de risque mentionnés dans les documents ?
```

### Pipeline

```text
Question
   ↓
Embedding
   ↓
ChromaDB
   ↓
Relevant chunks
   ↓
Context
   ↓
Mistral-7B
   ↓
Réponse + Sources
```

---

## 🔒 Confidentialité

Le projet est conçu pour fonctionner **localement** :

```text
PDF
 ↓
Local processing
 ↓
Local embeddings
 ↓
Local ChromaDB
 ↓
Local Mistral-7B
 ↓
Response
```

Les documents ne sont donc pas nécessairement envoyés vers un service cloud ou une API externe.

> ⚠️ Malgré son fonctionnement local, il est recommandé de ne pas utiliser de données médicales personnelles réelles dans un environnement de démonstration ou de développement sans mettre en place les mesures de sécurité et de conformité appropriées.

---

## ⚠️ Avertissement médical

> **Important :** Medical RAG Chatbot est un projet éducatif et expérimental. Il ne constitue pas un dispositif médical et ne remplace en aucun cas l'avis, le diagnostic ou la décision d'un professionnel de santé.
>
> Les réponses générées doivent être vérifiées à partir des documents sources et, pour toute décision médicale, auprès d'un professionnel de santé qualifié.

---

## 🚀 Améliorations futures

Quelques pistes d'évolution :

* [ ] Support des PDF contenant des images et tableaux
* [ ] OCR pour les documents numérisés
* [ ] Reranking des passages récupérés
* [ ] Support de plusieurs modèles LLM
* [ ] Évaluation automatique de la qualité du retrieval
* [ ] Évaluation des hallucinations
* [ ] Interface d'administration des documents
* [ ] Authentification des utilisateurs
* [ ] Chiffrement des données sensibles
* [ ] Déploiement Docker
* [ ] Support multilingue
* [ ] Export des conversations
* [ ] Dashboard d'évaluation RAG

---

---

## 📄 Licence

Ce projet est développé à des fins **éducatives et de recherche**.

Consultez le fichier `LICENSE` du dépôt pour les conditions d'utilisation.
