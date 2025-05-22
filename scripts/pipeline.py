from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rag_chain(llm, persistent_directory, embeddings):
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

