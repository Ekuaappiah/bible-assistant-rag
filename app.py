import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from scripts.ingest import parse_bible_file
from scripts.pipeline import build_rag_chain


def main():
    # Load environment variables
    load_dotenv()

    # Initialize model and embeddings
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Set up file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_dir, 'db', 'chroma_bible.db')
    file_path = os.path.join(current_dir, 'data', 'kjv.txt')

    # Load vector store if not already created
    parse_bible_file(file_path, persistent_directory, embedding_model)

    # Build the RAG chain
    rag_chain = build_rag_chain(
        llm=llm,
        persistent_directory=persistent_directory,
        embeddings=embedding_model
    )

    # Streamlit App UI
    st.set_page_config(page_title="Bible Chat Assistant", page_icon="📖")
    st.title("📖 Bible Chat Assistant")
    st.markdown("Ask Bible-based questions and get scripture-grounded responses.")

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.chat_input("Ask your question here...")

    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.markdown(msg)

    # Process new query
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching ..."):
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history[:-1],
                })
                answer = response.get("answer", "No answer returned.")
                st.markdown(answer)
                st.session_state.chat_history.append(answer)


if __name__ == "__main__":
    main()
