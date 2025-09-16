"""
Module: chromadb_connector.py
Description: Handles loading chunks, generating embeddings, and adding them to ChromaDB.
"""

import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.schema import Document


class ChromaDBConnector:
    def __init__(self, 
                 chunk_path: str = "resources/processed/chunks.pkl",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "db/chromadb_data",
                 collection_name: str = "infofusion_chunks"):
        self.chunk_path = chunk_path
        self.embedding_model_name = embedding_model_name
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.all_chunks: List[Document] = []
        self.embeddings = None
        self.client = None
        self.collection = None
        self.embedder = None

    def load_chunks(self):
        if os.path.exists(self.chunk_path):
            with open(self.chunk_path, "rb") as f:
                self.all_chunks = pickle.load(f)
            print(f"Loaded {len(self.all_chunks)} chunks from {self.chunk_path}")
        else:
            raise FileNotFoundError(f"Chunk file not found: {self.chunk_path}. Please run the ingestion pipeline first.")

    def load_embedding_model(self):
        self.embedder = SentenceTransformer(self.embedding_model_name)
        print(f"Embedding model '{self.embedding_model_name}' loaded.")

    def generate_embeddings(self):
        texts = [chunk.page_content for chunk in self.all_chunks]
        print(f"Computing embeddings for {len(texts)} chunks.")
        self.embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
        print(f"Generated {len(self.embeddings)} embeddings.")

    def init_chroma_client(self):
        self.client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings()
        )
        print(f"ChromaDB client initialized at '{self.chroma_db_path}'")

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' created or retrieved.")

    def add_chunks_to_collection(self):
        if not self.collection or self.embeddings is None:
            raise RuntimeError("Chroma client or embeddings not initialized.")

        docs_to_add = []
        for i, chunk in enumerate(self.all_chunks):
            doc = {
                "id": str(i),
                "embedding": self.embeddings[i],
                "document": chunk.page_content,
                "metadata": chunk.metadata
            }
            docs_to_add.append(doc)

        self.collection.add(
            ids=[doc["id"] for doc in docs_to_add],
            embeddings=self.embeddings,
            documents=[doc["document"] for doc in docs_to_add],
            metadatas=[doc["metadata"] for doc in docs_to_add]
        )
        print(f"Added {len(docs_to_add)} chunks to ChromaDB collection.")

    def run(self):
        self.load_chunks()
        self.load_embedding_model()
        self.generate_embeddings()
        self.init_chroma_client()
        self.add_chunks_to_collection()
        print("ChromaDB creation and chunk addition completed.")