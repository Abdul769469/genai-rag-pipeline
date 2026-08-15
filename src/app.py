"""
Web UI for the RAG pipeline, built with Streamlit.

Run with:  streamlit run src/app.py

This reuses the exact same RagPipeline class from rag_pipeline.py — the
retrieval and generation logic doesn't change at all. Streamlit just gives
us a browser-based chat interface instead of a terminal loop.
"""

import streamlit as st
from rag_pipeline import RagPipeline

st.set_page_config(page_title="Acme Robotics RAG demo", page_icon="🤖")

st.title("🤖 Acme Robotics knowledge assistant")
st.caption("Ask questions about Acme Robotics — answers are grounded in the documents in data/sample_docs/")


@st.cache_resource(show_spinner="Loading embedding model and vector index...")
def load_pipeline():
    """
    Streamlit reruns the whole script on every user interaction. Without
    caching, we'd reload the embedding model and FAISS index on every
    single message — slow and wasteful. @st.cache_resource keeps one
    shared instance alive across reruns.
    """
    return RagPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("No index found. Run `python src/build_index.py` in your terminal first, then reload this page.")
    st.stop()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

# Chat history lives in Streamlit's session state so it survives reruns
# (but resets if you refresh the page — that's expected for this demo).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay previous messages so the chat history stays visible after each rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

# Chat input box pinned to the bottom of the page
if user_question := st.chat_input("Ask a question about Acme Robotics..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = pipeline.answer(user_question)
        st.markdown(result["answer"])
        sources = sorted(set(result["sources"]))
        st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": sources}
    )

with st.sidebar:
    st.header("How this works")
    st.markdown(
        """
        1. Your question is embedded into a vector
        2. FAISS finds the most similar document chunks
        3. Those chunks + your question are sent to Groq (Llama 3.3)
        4. The model answers using only that retrieved context
        """
    )
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()
