"""
rag_engine.py  –  Medical RAG logic
  • Web scraping      : requests + BeautifulSoup
  • Embeddings        : sentence-transformers (all-MiniLM-L6-v2, local/free)
  • Dense retrieval   : FAISS (in-memory vector search)
  • Sparse retrieval  : BM25 (keyword-based search)
  • Fusion            : Reciprocal Rank Fusion (RRF) combining FAISS + BM25
  • Medical search    : PubMed API (free, no key needed)
  • Live web search   : Tavily Search API (free tier)
  • LLM               : Groq API (llama / mixtral / gemma)
"""

from __future__ import annotations
import re
import time
import textwrap
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# PubMed Search  (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

class PubMedSearch:
    """Searches PubMed via NCBI E-utilities (free, no key required)."""

    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Return list of {title, url, content, authors, year} dicts."""
        try:
            # Step 1: get PMIDs
            r = requests.get(self.ESEARCH, params={
                "db": "pubmed", "term": query,
                "retmax": max_results, "retmode": "json",
                "sort": "relevance",
            }, timeout=10)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            # Step 2: fetch abstracts
            r2 = requests.get(self.EFETCH, params={
                "db": "pubmed", "id": ",".join(ids),
                "rettype": "abstract", "retmode": "xml",
            }, timeout=15)
            r2.raise_for_status()

            root = ET.fromstring(r2.content)
            results = []
            for article in root.findall(".//PubmedArticle"):
                try:
                    title = article.findtext(".//ArticleTitle") or "Untitled"
                    abstract_parts = article.findall(".//AbstractText")
                    abstract = " ".join(
                        (p.get("Label", "") + ": " if p.get("Label") else "") + (p.text or "")
                        for p in abstract_parts
                    ).strip()
                    pmid = article.findtext(".//PMID") or ""
                    year = article.findtext(".//PubDate/Year") or ""
                    authors_els = article.findall(".//Author")
                    authors = ", ".join(
                        f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
                        for a in authors_els[:3]
                    )
                    if authors_els and len(authors_els) > 3:
                        authors += " et al."

                    if abstract:
                        results.append({
                            "title": title,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "content": abstract,
                            "authors": authors,
                            "year": year,
                        })
                except Exception:
                    continue
            return results
        except Exception as e:
            return [{"title": "PubMed Error", "url": "", "content": str(e), "authors": "", "year": ""}]


# ─────────────────────────────────────────────────────────────────────────────
# Live Web Search  (Tavily)
# ─────────────────────────────────────────────────────────────────────────────

class LiveSearch:
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
        }
        try:
            resp = requests.post(self.ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = []
            if data.get("answer"):
                results.append({
                    "title": "Web Search Answer",
                    "url": "tavily://answer",
                    "content": data["answer"],
                })
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                })
            return results
        except Exception as e:
            return [{"title": "Search Error", "url": "", "content": str(e)}]


# ─────────────────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────────────────

class WebScraper:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    def scrape_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        visited: set[str] = set()
        all_pages: List[Dict] = []
        self._crawl(url, url, depth, visited, all_pages)
        return {"pages": all_pages, "total": len(all_pages)}

    def _crawl(self, base: str, url: str, depth: int, visited: set, pages: list):
        if url in visited or depth < 0:
            return
        visited.add(url)
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception:
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        body = soup.get_text(separator=" ")
        text = clean_text(body)

        if len(text) > 100:
            pages.append({"url": url, "title": title, "text": text})

        if depth > 1:
            for a in soup.find_all("a", href=True):
                href = urljoin(base, a["href"])
                if urlparse(href).netloc == urlparse(base).netloc:
                    self._crawl(base, href, depth - 1, visited, pages)
                    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Vector + BM25 Store with RRF
# ─────────────────────────────────────────────────────────────────────────────

class HybridStore:
    """
    Combines FAISS (dense) + BM25 (sparse) retrieval
    with Reciprocal Rank Fusion (RRF) for final ranking.
    """

    def __init__(self, dim: int = 384, rrf_k: int = 60):
        self.dim = dim
        self.rrf_k = rrf_k
        # FAISS
        self.index = faiss.IndexFlatL2(dim)
        # BM25 (rebuilt on each add)
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_chunks: List[List[str]] = []
        # Shared storage
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], sources: List[str]):
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        self.metadata.extend([{"source": s} for s in sources])
        # Rebuild BM25 index
        self.tokenized_chunks.extend([tokenize(c) for c in chunks])
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5):
        if self.index.ntotal == 0:
            return [], []

        n = self.index.ntotal
        k = min(top_k * 3, n)  # fetch more candidates for fusion

        # ── FAISS dense retrieval ─────────────────────────────────────────────
        q = query_embedding.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(q, k)
        faiss_ranks: Dict[int, int] = {}
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx != -1 and dist < 2.0:
                faiss_ranks[int(idx)] = rank

        # ── BM25 sparse retrieval ─────────────────────────────────────────────
        bm25_ranks: Dict[int, int] = {}
        if self.bm25:
            scores = self.bm25.get_scores(tokenize(query_text))
            top_bm25 = np.argsort(scores)[::-1][:k]
            for rank, idx in enumerate(top_bm25):
                if scores[idx] > 0:
                    bm25_ranks[int(idx)] = rank

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        all_ids = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores: Dict[int, float] = {}
        for idx in all_ids:
            score = 0.0
            if idx in faiss_ranks:
                score += 1.0 / (self.rrf_k + faiss_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[idx])
            rrf_scores[idx] = score

        top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = [self.chunks[i] for i in top_ids]
        srcs    = [self.metadata[i]["source"] for i in top_ids]
        return results, srcs

    def clear(self):
        self.index.reset()
        self.chunks.clear()
        self.metadata.clear()
        self.tokenized_chunks.clear()
        self.bm25 = None

    @property
    def total(self):
        return self.index.ntotal


# ─────────────────────────────────────────────────────────────────────────────
# Medical RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """
You are MediAssist AI, an advanced medical information assistant powered by RAG.
You help healthcare professionals and patients understand medical information clearly.

STRICT RULES:
1. Always recommend consulting a qualified healthcare professional for diagnosis/treatment.
2. Never provide specific dosage advice without noting professional consultation is required.
3. Cite sources when available.
4. Be clear about the difference between general information and medical advice.
5. If a question involves emergency symptoms, always advise seeking immediate medical attention.
6. Be empathetic, clear, and use plain language alongside medical terminology.
""".strip()


class RAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        tavily_api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        top_k: int = 5,
        temperature: float = 0.3,
    ):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.tavily_api_key = tavily_api_key
        self.live_search: Optional[LiveSearch] = (
            LiveSearch(tavily_api_key) if tavily_api_key else None
        )
        self.pubmed = PubMedSearch()

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = HybridStore(dim=384)
        self.scraper = WebScraper()
        self._source_set: set[str] = set()

    def set_tavily_key(self, key: str):
        self.tavily_api_key = key
        self.live_search = LiveSearch(key)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_url(self, url: str, depth: int = 1) -> Dict[str, Any]:
        result = self.scraper.scrape_url(url, depth=depth)
        pages = result["pages"]
        if not pages:
            raise ValueError(f"No content extracted from {url}")

        total_chunks = 0
        for page in pages:
            chunks = chunk_text(page["text"])
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            sources = [page["url"]] * len(chunks)
            self.vector_store.add(np.array(embeddings), chunks, sources)
            self._source_set.add(page["url"])
            total_chunks += len(chunks)

        return {"chunks": total_chunks, "pages": len(pages)}

    def add_text(self, text: str, label: str = "manual") -> Dict[str, Any]:
        text = clean_text(text)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No usable text found.")
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        sources = [label] * len(chunks)
        self.vector_store.add(np.array(embeddings), chunks, sources)
        self._source_set.add(label)
        return {"chunks": len(chunks)}

    # ── Query pipeline ────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        q_emb = self.embedder.encode([question], show_progress_bar=False)[0]

        # 1. Hybrid FAISS + BM25 retrieval
        chunks, sources = self.vector_store.search(q_emb, question, top_k=self.top_k)

        # 2. PubMed search (always runs for medical queries)
        pubmed_results = self.pubmed.search(question, max_results=3)
        pubmed_context = ""
        pubmed_sources = []
        if pubmed_results:
            parts = []
            for r in pubmed_results:
                if r["content"] and "Error" not in r["title"]:
                    src_label = f"{r['title']} ({r.get('year','')}, {r.get('authors','')})"
                    parts.append(f"[PubMed: {src_label}]\n{r['content']}")
                    pubmed_sources.append(r["url"])
            pubmed_context = "\n\n---\n\n".join(parts)

        used_web_search = False
        web_sources = []

        # 3. Build context from available sources
        if chunks:
            doc_context = "\n\n---\n\n".join(
                f"[Source: {s}]\n{c}" for c, s in zip(chunks, sources)
            )
            context_section = f"INDEXED DOCUMENTS:\n{doc_context}"
        elif self.live_search:
            used_web_search = True
            web_results = self.live_search.search(question, max_results=5)
            web_parts = []
            for r in web_results:
                if r["content"]:
                    web_parts.append(f"[Web: {r['url']}]\n{r['title']}\n{r['content']}")
                    web_sources.append(r["url"])
            context_section = "LIVE WEB RESULTS:\n" + "\n\n---\n\n".join(web_parts)
        else:
            context_section = ""

        # 4. Assemble system prompt
        context_block = ""
        if context_section:
            context_block += context_section + "\n\n"
        if pubmed_context:
            context_block += f"PUBMED LITERATURE:\n{pubmed_context}"

        if context_block:
            system_prompt = MEDICAL_SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context_block}"
        else:
            system_prompt = MEDICAL_SYSTEM_PROMPT + (
                "\n\nNo external context available. Answer from medical training knowledge "
                "and always recommend professional consultation."
            )

        # 5. Call Groq LLM
        chat = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer = chat.choices[0].message.content
        all_sources = sources + pubmed_sources + web_sources
        unique_sources = [s for s in list(dict.fromkeys(all_sources))
                          if s and s != "tavily://answer"]

        return {
            "answer": answer,
            "sources": unique_sources,
            "pubmed_sources": pubmed_sources,
            "chunks_used": len(chunks),
            "used_web_search": used_web_search,
            "used_pubmed": bool(pubmed_results and pubmed_context),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, int]:
        return {
            "chunks": self.vector_store.total,
            "sources": len(self._source_set),
        }

    def clear(self):
        self.vector_store.clear()
        self._source_set.clear()
