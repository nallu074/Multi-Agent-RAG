# Multi-Agent-RAG
Multi-Agent RAG System using LangGraph that routes user queries between a vector database and external tools (Wikipedia search). Implements web content ingestion, token-based text chunking, MiniLM embeddings stored in AstraDB, and stateful Llama-3 agents for context-aware retrieval and response generation across multiple data sources.

# Multi-Agent RAG Chatbot using LangGraph & AstraDB

A **multi-agent Retrieval-Augmented Generation (RAG) chatbot** that intelligently routes user queries between a private vector database (AstraDB) and external tools (Wikipedia search). Built using **LangGraph** for agent orchestration, **OpenAI embeddings** for semantic retrieval, and **Llama-3.3-70B-Versatile** for high-quality reasoning and answer generation.

---

## 🚀 Overview

This project demonstrates an **agentic RAG architecture** where the system decides *where knowledge should come from* before generating an answer. Instead of relying on a single vector store, a router agent dynamically chooses between:

- **Vector Database (AstraDB)** – for questions related to indexed internal documents
- **Wikipedia Search Tool** – for general or out-of-domain knowledge

This results in more accurate, grounded, and explainable responses.

---

## 🧠 Motivation

Traditional RAG pipelines assume all relevant knowledge exists in one vector store. In real-world systems:

- Internal policies, guidelines, or reports live in private document repositories
- General definitions and background knowledge live on the public web

This project treats **retrieval routing as a first-class problem**, improving relevance and reducing hallucinations.

---

## 🧩 Key Components

| Component | Purpose |
|----------|---------|
| **LangGraph** | Agent orchestration and conditional routing |
| **AstraDB** | Scalable vector database for private knowledge |
| **Huggingface Embeddings** | Semantic embeddings (`ll-MiniLM-L6-v2`) |
| **Llama-3.3-70B-Versatile** | Routing decisions and answer generation |
| **Wikipedia Tool** | External knowledge fallback |

---

## 📂 Project Structure
```text
├── README.md
├── app.py # Streamlit application entrypoint
├── requirements.txt # Python dependencies
└── src/
  ├── config.py # Environment & configuration
  ├── graph.py # LangGraph workflow definition
  ├── vectorstore.py # AstraDB vector store utilities
  ├── ingest.py # Ingesting vector data into AstraDB
  └── tools.py # External tools (Wikipedia search) 

```
---

## ✨ Features

- Intelligent **agent-based query routing**
- Context-aware answer generation
- Separation of private vs public knowledge
- Modular and extensible architecture
- Production-inspired design patterns

---

## ⚙️ Setup & Installation

### Prerequisites

Create a `.env` file with the following variables:

```text
ASTRA_DB_ID=...
ASTRA_DB_APPLICATION_TOKEN=...
GROQ_API_KEY=...
OPENAI_API_KEY=...

LLM_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=text-embedding-3-small
VECTOR_TABLE=qa_mini_demo
```


### Install Dependencies

```text
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### Running the application

```text
streamlit run app.py
``` 
