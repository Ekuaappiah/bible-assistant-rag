import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_bible_file(file_path: str, persistent_directory: str, embeddings):
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