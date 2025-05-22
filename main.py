import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


from scripts.ingest import  parse_bible_file
from scripts.pipeline import build_rag_chain

# Load environment variables from .env
load_dotenv()

# Load LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_bible.db')
file_path = os.path.join(current_dir, 'data', 'kjv.txt')

# initialize embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

query = "What does God say about anxiety? How does this relate to what we discussed earlier about trust?"

parse_bible_file(file_path, persistent_directory,embedding_model)
rag_chain = build_rag_chain(
    llm=llm,
    persistent_directory=persistent_directory,
    embeddings=embedding_model
)

result = rag_chain.invoke({
    "input": query,
    "chat_history": [],
})

print(result)



