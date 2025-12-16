
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import get_settings
from .vectorstore import add_documents


DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def load_and_split(urls: List[str] = DEFAULT_URLS, chunk_size: int = 500, chunk_overlap: int = 0) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [d for sub in docs for d in sub]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs_list)


def main() -> None:
    settings = get_settings()
    flag = Path(settings.ingest_flag_path)

    if flag.exists():
        print(f"Skip ingestion: found flag file at {flag.resolve()}")
        return

    docs = load_and_split()
    n = add_documents(settings, docs)
    flag.write_text(f"ingested={n}\n", encoding="utf-8")
    print(f"Inserted {n} documents into Astra table '{settings.astra_table_name}'.")
    print(f"Wrote ingestion flag: {flag.resolve()}")


if __name__ == "__main__":
    main()
