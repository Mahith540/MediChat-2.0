"""Build the Chroma index from every .txt file in data/.

Runs at Docker build time so the image ships with a fresh index. app.py also
builds it on first request if embeddings/ is missing or empty.
"""
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"
DB_PATH = "embeddings"


def ingest_documents():
    documents = []
    for file in sorted(os.listdir(DATA_PATH)):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DATA_PATH, file), encoding="utf-8")
            documents.extend(loader.load())
    if not documents:
        raise SystemExit(f"No .txt files found in {DATA_PATH}/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    db.persist()
    print(f"Indexed {len(chunks)} chunks from {len(documents)} files into {DB_PATH}/")


if __name__ == "__main__":
    ingest_documents()
