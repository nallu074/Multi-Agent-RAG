
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.config import get_settings
from src.graph import build_app


st.set_page_config(page_title="Multi-Agent RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent RAG Chatbot (LangGraph + AstraDB)")
st.caption("Router chooses between AstraDB vector search and Wikipedia tool, then LLM generates the answer.")


@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    load_dotenv()
    settings = get_settings()
    return settings, build_app(settings)


settings, graph = get_compiled_graph()

with st.sidebar:
    st.subheader("Settings")
    st.write(f"**Astra table:** `{settings.astra_table_name}`")
    st.write(f"**Embeddings:** `{settings.hf_model_name}`")
    st.write(f"**LLM:** `{settings.llm_model}`")
    st.divider()
    st.markdown("**Ingestion note**")
    st.write("This app does **not** ingest documents on every run.")
    st.write("Run ingestion once from terminal:")
    st.code("python -m src.ingest", language="bash")


question = st.text_input("Ask a question", placeholder="e.g., What are the types of agent memory?")

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Run", type="primary", use_container_width=True)
with col2:
    show_debug = st.toggle("Show debug (route + sources)", value=False)

if ask and question.strip():
    with st.spinner("Thinking..."):
        result = graph.invoke({"question": question.strip()})

    st.subheader("Answer")
    st.write(result.get("answer", ""))

    if show_debug:
        st.subheader("Debug")
        st.write(f"**Route:** `{result.get('route')}`")

        if result.get("route") == "vectorstore":
            docs = result.get("documents", []) or []
            st.write(f"**Retrieved docs:** {len(docs)}")
            for i, d in enumerate(docs, 1):
                with st.expander(f"Doc {i}"):
                    st.write(d.page_content)
                    st.caption(d.metadata)
        else:
            with st.expander("Wikipedia tool output"):
                st.write(result.get("tool_result", ""))

elif ask:
    st.warning("Please enter a question.")
