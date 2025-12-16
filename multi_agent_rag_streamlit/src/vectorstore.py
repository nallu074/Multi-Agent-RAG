
from __future__ import annotations

import os
from typing import List

import cassio
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import Settings


def init_cassio(settings: Settings) -> None:
    # cassio.init is idempotent-ish; still keep it in one place
    cassio.init(
        token=settings.astra_db_application_token,
        database_id=settings.astra_db_id,
    )


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    # Only needed for gated models; harmless otherwise
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    return HuggingFaceEmbeddings(model_name=settings.hf_model_name)


def get_vectorstore(settings: Settings) -> Cassandra:
    init_cassio(settings)
    embeddings = get_embeddings(settings)

    # session/keyspace None means it uses the cassio global connection
    return Cassandra(
        embedding=embeddings,
        table_name=settings.astra_table_name,
        session=None,
        keyspace=None,
    )


def get_retriever(settings: Settings, k: int = 4):
    vs = get_vectorstore(settings)
    return vs.as_retriever(search_kwargs={"k": k})


def add_documents(settings: Settings, docs: List[Document]) -> int:
    """Ingest documents into Astra table. Use ONLY from ingest script."""
    vs = get_vectorstore(settings)
    vs.add_documents(docs)
    return len(docs)
