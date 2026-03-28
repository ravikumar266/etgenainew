"""
agent/tools_search.py
─────────────────────
Web search and webpage fetching tools.
"""

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool

from agent.config import tavily_client


@tool
def search_web(query: str) -> str:
    """Search the web using Tavily and return top results with titles, URLs, and summaries."""
    try:
        response = tavily_client.search(query=query, max_results=5)
        results = [
            f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r.get('content', '')[:300]}"
            for r in response.get("results", [])
        ]
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def fetch_webpage(url: str) -> str:
    """Fetch and extract readable text content from a webpage URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ", strip=True).split())
        return text[:5000] or "(page appears empty)"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        results_list = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results_list.append(
                    f"Title: {r.get('title')}\n"
                    f"Link: {r.get('href')}\n"
                    f"Snippet: {r.get('body')}"
                )
        return "\n\n".join(results_list) if results_list else "No results found."
    except Exception as e:
        return f"DuckDuckGo search failed: {str(e)}"