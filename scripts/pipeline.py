from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_rag_chain(llm, persistent_directory, embeddings):
    """
    Constructs a history-aware Retrieval-Augmented Generation (RAG) chain for Bible study.

    This function initializes a Chroma vector store using the specified persistent directory
    and embedding model. It sets up a retriever with similarity search and enhances it with
    conversational memory awareness. The chain then uses a Bible-specific QA prompt to generate
    grounded responses from retrieved documents.

    The final chain supports:
    - Rephrasing user questions in context-aware form using prior chat history.
    - Answering Bible-related queries strictly based on scripture.
    - Generating Bible reading plans, summaries, and structured content derived directly from verses.

    Parameters:
        llm: A language model (LLM) instance compatible with LangChain.
        persistent_directory (str): Path to the persisted Chroma vector store.
        embeddings: Embedding function or model used to represent the text in vector space.

    Returns:
        rag_chain: A LangChain RAG chain object that performs context-aware retrieval
                   and structured document-based answering.

    Notes:
        - The retriever returns top 3 similar documents based on vector similarity.
        - All prompts explicitly restrict the assistant to biblical content with no personal
          interpretation unless scripturally derived.
    """
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 3}
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for studying the Bible. Rephrase the latest user question as a standalone question, taking prior dialogue into account."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Make retriever history-aware
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a knowledgeable assistant trained on the Bible. Use only the scriptures provided below to help the user.\n\n"
         "You may:\n"
         "- Answer questions directly from the Bible.\n"
         "- Generate Bible study or reading plans based on themes or queries.\n"
         "- Provide summaries, lists, or tasks—but always grounded in scripture.\n\n"
         "Important rules:\n"
         "- Do NOT answer from personal opinion or add interpretation unless it is clearly derived from scripture.\n"
         "- If the Bible does not provide a clear answer, say: \"The Bible does not provide a clear answer to that.\"\n"
         "- Always cite chapter and verse when appropriate.\n\n"
         "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine retriever and QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain
