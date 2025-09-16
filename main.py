from src.infofusion_rag.ingestion.pdf_loader import PDFLoader
from src.infofusion_rag.retrieval.chromadb_connector import ChromaDBConnector



def data_ingestion_and_chunking():
    loader = PDFLoader(data_folder="data", save_path="resources/processed/chunks.pkl")
    loader.run()


def vectordb_building():
    connector = ChromaDBConnector(
        chunk_path="resources/processed/chunks.pkl",
        chroma_db_path="resources/db/chromadb_data"
    )
    connector.run()


if __name__ == "__main__":
    # Run ingestion and vectordb build sequentially if needed
    data_ingestion_and_chunking()
    vectordb_building()

  