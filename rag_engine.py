from __future__ import annotations
import re
import requests
import textwrap
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse

import numpy as np
import faiss

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import Groq


MEDICAL_KEYWORDS = [
    "disease", "symptom", "medicine",
    "treatment", "fever", "pain",
    "infection", "covid", "diabetes",
    "asthma", "blood pressure",
    "headache", "cancer"
]

SYMPTOM_DB = {
    "fever": "Possible viral infection or flu.",
    "headache": "Possible migraine or stress.",
    "cough": "Possible respiratory infection.",
    "chest pain": "Possible cardiac issue."
}

MEDICINE_DB = {
    "paracetamol": "Used for fever and pain relief.",
    "ibuprofen": "Pain relief medicine.",
    "cetirizine": "Used for allergies."
}


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=80):

    words = text.split()

    chunks = []

    i = 0

    while i < len(words):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

        i += chunk_size - overlap

    return chunks


class WebScraper:

    HEADERS = {
        "User-Agent": "Mozilla/5.0"
    }

    def scrape_url(self, url):

        response = requests.get(
            url,
            headers=self.HEADERS,
            timeout=15
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer"
        ]):
            tag.decompose()

        text = soup.get_text(" ")

        return clean_text(text)


class VectorStore:

    def __init__(self, dim=384):

        self.index = faiss.IndexFlatL2(dim)

        self.chunks = []

        self.sources = []

    def add(self, embeddings, chunks, sources):

        self.index.add(
            embeddings.astype("float32")
        )

        self.chunks.extend(chunks)

        self.sources.extend(sources)

    def search(self, embedding, top_k=4):

        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(
            embedding.reshape(1, -1).astype("float32"),
            top_k
        )

        results = []

        for idx in I[0]:

            if idx != -1:

                results.append(self.chunks[idx])

        return results


class RAGEngine:

    def __init__(
        self,
        groq_api_key,
        tavily_api_key="",
        model="llama-3.3-70b-versatile"
    ):

        self.client = Groq(api_key=groq_api_key)

        self.model = model

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.scraper = WebScraper()

        self.vector_store = VectorStore()

    def detect_medical_query(self, question):

        q = question.lower()

        return any(k in q for k in MEDICAL_KEYWORDS)

    def symptom_checker(self, question):

        q = question.lower()

        for symptom in SYMPTOM_DB:

            if symptom in q:

                return SYMPTOM_DB[symptom]

        return None

    def medicine_lookup(self, question):

        q = question.lower()

        for med in MEDICINE_DB:

            if med in q:

                return MEDICINE_DB[med]

        return None

    def add_url(self, url):

        text = self.scraper.scrape_url(url)

        chunks = chunk_text(text)

        embeddings = self.embedder.encode(chunks)

        self.vector_store.add(
            np.array(embeddings),
            chunks,
            [url] * len(chunks)
        )

        return {
            "chunks": len(chunks)
        }

    def query(self, question):

        q_emb = self.embedder.encode([question])[0]

        chunks = self.vector_store.search(
            q_emb,
            top_k=4
        )

        medical_context = []

        symptom = self.symptom_checker(question)

        if symptom:
            medical_context.append(
                f"Symptom Analysis: {symptom}"
            )

        medicine = self.medicine_lookup(question)

        if medicine:
            medical_context.append(
                f"Medicine Info: {medicine}"
            )

        context = "\n\n".join(chunks)

        system_prompt = textwrap.dedent(f"""
        You are MediBot 🏥, a professional medical AI assistant.

        Rules:
        - Give medical answers only
        - Use medical context
        - Explain clearly
        - Never give dangerous advice
        - Recommend consulting a doctor

        Medical Context:
        {chr(10).join(medical_context)}

        Retrieved Context:
        {context}
        """)

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        answer = chat.choices[0].message.content

        return {
            "answer": answer
        }
