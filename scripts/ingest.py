import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_bible_file(file_path: str, persistent_directory: str, embeddings):
    """
    Parses a Bible text file and initializes a Chroma vector store using LangChain.

    This function checks if a persistent vector store directory exists. If it does not,
    it will:
    - Validate the file path.
    - Load the text from the specified file using a LangChain TextLoader.
    - Split the text into manageable chunks using a RecursiveCharacterTextSplitter.
    - Generate embeddings for the document chunks.
    - Create and persist a Chroma vector store using those embeddings.

    If the persistent directory already exists, the function will skip initialization.

    Parameters:
        file_path (str): Absolute or relative path to the Bible text file.
        persistent_directory (str): Path where the Chroma vector store should be persisted.
        embeddings: Embedding model or embedding function compatible with LangChain
                    to convert document chunks into vector representations.

    Raises:
        FileNotFoundError: If the specified Bible text file does not exist.

    Note:
        This function assumes that the embeddings parameter is already a valid,
        initialized embedding model compatible with the Chroma vector store.
    """
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the text file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path."
            )

        # Read the text content from the file
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = embeddings

        # Create the vector store and persist it automatically
        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persistent_directory
        )

    else:
        print("Vector store already exists. No need to initialize.")
