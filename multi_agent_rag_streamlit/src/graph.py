
from __future__ import annotations

from typing import Literal, TypedDict, List

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

from .config import Settings
from .vectorstore import get_retriever
from .tools import get_wiki_tool


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search"] = Field(...)


class GraphState(TypedDict, total=False):
    question: str
    route: str
    documents: List[Document]
    tool_result: str
    answer: str


def build_llm(settings: Settings) -> ChatGroq:
    if not settings.groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env for Groq inference.")
    return ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model, temperature=0)


def build_app(settings: Settings):
    llm = build_llm(settings)

    # Router (structured output)
    system = """You are an expert at routing a user question to a vectorstore or wikipedia.
                The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    structured_router = llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_router

    retriever = get_retriever(settings, k=4)
    wiki = get_wiki_tool(top_k_results=1, doc_content_chars_max=1200)

    # Nodes
    def route_question(state: GraphState) -> GraphState:
        q = state["question"]
        decision = question_router.invoke({"question": q})
        return {"route": decision.datasource}

    def retrieve(state: GraphState) -> GraphState:
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def wiki_search(state: GraphState) -> GraphState:
        q = state["question"]
        result = wiki.run(q)
        return {"tool_result": result}

    def generate(state: GraphState) -> GraphState:
        q = state["question"]
        route = state.get("route", "")

        if route == "vectorstore":
            docs = state.get("documents", [])
            context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Answer using ONLY the provided context. If not in context, say you don't know."),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            )
            msg = prompt.format_messages(question=q, context=context)
            ans = llm.invoke(msg).content
            return {"answer": ans}

        # wiki_search
        tool_text = state.get("tool_result", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer concisely using the tool output."),
                ("human", "Question: {question}\n\nTool output:\n{tool_text}"),
            ]
        )
        msg = prompt.format_messages(question=q, tool_text=tool_text)
        ans = llm.invoke(msg).content
        return {"answer": ans}

    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges(
        "route_question",
        lambda s: s["route"],
        {"vectorstore": "retrieve", "wiki_search": "wiki_search"},
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("wiki_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
