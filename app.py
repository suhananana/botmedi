import streamlit as st
import os
from rag_engine import RAGEngine

st.set_page_config(
    page_title="MediBot 🏥",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 MediBot — Medical RAG Assistant")

if "rag" not in st.session_state:
    st.session_state.rag = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🔑 API Keys")

    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

    if groq_key and st.session_state.rag is None:
        st.session_state.rag = RAGEngine(
            groq_api_key=groq_key,
            tavily_api_key=tavily_key
        )

    st.markdown("---")
    st.markdown("### 🌐 Add Medical Website")

    url = st.text_input("Medical Website URL")

    if st.button("Scrape & Index"):
        if st.session_state.rag:
            result = st.session_state.rag.add_url(url)
            st.success(f"Indexed {result['chunks']} chunks")

st.markdown("## 💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask medical questions...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = st.session_state.rag.query(prompt)

            answer = result["answer"]

            answer += """

⚠️ Disclaimer:
This chatbot is for educational purposes only.
Consult a doctor for medical advice.
"""

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
