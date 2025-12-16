
from __future__ import annotations

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


def get_wiki_tool(top_k_results: int = 1, doc_content_chars_max: int = 800) -> WikipediaQueryRun:
    wrapper = WikipediaAPIWrapper(
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max,
    )
    return WikipediaQueryRun(api_wrapper=wrapper)
