"""
Module: pdf_loader.py
Description: PDF ingestion and chunking encapsulated in a class.
"""

import os
import pickle
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class PDFLoader:
    def __init__(self, data_folder: str = "data", chunk_size: int = 1000, chunk_overlap: int = 100, save_path: str = "research_notebooks/processed/chunks.pkl"):
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.save_path = save_path

    def list_pdfs(self) -> List[str]:
        """List all PDF files in the data folder."""
        pdf_files = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".pdf")]
        return pdf_files

    def load_pdfs(self, pdf_files: List[str]) -> List[List[Document]]:
        """Load PDFs and extract raw documents."""
        all_docs = []
        for pdf_file in pdf_files:
            file_path = os.path.join(self.data_folder, pdf_file)
            print(f"Loading file: {pdf_file}")
            loader = PyPDFLoader(file_path=file_path)
            documents = loader.load()
            all_docs.append(documents)
        return all_docs

    def split_documents_into_chunks(self, docs: List[List[Document]]) -> List[Document]:
        """Split raw documents into chunks with overlap."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        all_chunks = []
        for doc in docs:
            chunks = splitter.split_documents(doc)
            all_chunks.extend(chunks)
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["source"] = source
        return all_chunks

    def save_chunks(self, chunks: List[Document]):
        """Save chunks to a pickle file."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"Chunks saved to {self.save_path}")

    def run(self):
        """Full pipeline execution: list, load, split, and save."""
        pdf_files = self.list_pdfs()
        print(f"Found {len(pdf_files)} PDF files for ingestion.")

        all_docs = self.load_pdfs(pdf_files)
        print(f"Loaded {len(all_docs)} raw documents from PDFs.")

        all_chunks = self.split_documents_into_chunks(all_docs)
        print(f"Split raw documents into {len(all_chunks)} chunks.")

        num_docs = len(pdf_files)
        num_chunks = len(all_chunks)
        avg_chunk_len = sum(len(chunk.page_content) for chunk in all_chunks) / num_chunks
        print(f"Number of source PDF documents: {num_docs}")
        print(f"Total text chunks created: {num_chunks}")
        print(f"Average chunk length (characters): {avg_chunk_len:.2f}")

        self.save_chunks(all_chunks)
