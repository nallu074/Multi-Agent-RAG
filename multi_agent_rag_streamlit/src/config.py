
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    astra_db_id: str
    astra_db_application_token: str
    astra_table_name: str = "qa_mini_demo"

    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"

    hf_model_name: str = "all-MiniLM-L6-v2"
    hf_token: str | None = None  # optional (only needed for gated models)

    # used to prevent accidental re-ingestion from Streamlit
    ingest_flag_path: str = ".ingested"


def get_settings() -> Settings:
    load_dotenv()
    astra_db_id = os.getenv("ASTRA_DB_ID", "").strip()
    astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "").strip()
    if not astra_db_id or not astra_token:
        raise ValueError("Missing ASTRA_DB_ID / ASTRA_DB_APPLICATION_TOKEN in .env")

    return Settings(
        astra_db_id=astra_db_id,
        astra_db_application_token=astra_token,
        astra_table_name=os.getenv("ASTRA_TABLE_NAME", "qa_mini_demo").strip() or "qa_mini_demo",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        hf_model_name=os.getenv("HF_EMBED_MODEL", "all-MiniLM-L6-v2"),
        hf_token=os.getenv("HF_TOKEN"),
        ingest_flag_path=os.getenv("INGEST_FLAG_PATH", ".ingested"),
    )
