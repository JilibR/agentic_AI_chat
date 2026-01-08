import logging
import os
from typing import List
from xml.dom.minidom import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from utils.data_loader import PdfExtractor

MODEL_EMBEDDING = "embeddinggemma:latest"


class VectorStoreManager:
    """Manages the persistence and updating of a Chroma vector store for PDF document chunks.

    This class provides methods to initialize, load, and update a Chroma vector store
    with document chunks, using Ollama embeddings. It supports both creating a new store
    and adding new documents to an existing one.

    Attributes:
        persist_directory (str): Directory where the vector store is persisted.
        embeddings (OllamaEmbeddings): Embedding model used for vectorization.
        vector_store (Chroma): The Chroma vector store instance.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = MODEL_EMBEDDING
    ):
        """Initializes the VectorStoreManager with a persist directory and embedding model.

        Args:
            persist_directory (str): Path to the directory for persisting the vector store.
            embedding_model (str): Name of the Ollama embedding model to use.
        """
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None

    def init_or_load(self, chunks: List[Document] = None) -> Chroma:
        """Initializes or loads the vector store.

        If the persist directory exists and is not empty, the store is loaded.
        If chunks are provided and no store exists, a new store is created and persisted.

        Args:
            chunks (List[Document], optional): List of document chunks to initialize a new store.

        Returns:
            Chroma: The initialized or loaded vector store.

        Raises:
            ValueError: If neither an existing store nor chunks are provided.
        """
        if os.path.exists(self.persist_directory) and \
            os.listdir(self.persist_directory):
            self.vector_store = Chroma(
                collection_name="pdf_agentic_AI",
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logging.info("Vector store loaded from disk.")
        elif chunks:
            self.vector_store = Chroma.from_documents(
                collection_name="pdf_agentic_AI",
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            logging.info("New vector store created and persisted.")
        else:
            raise ValueError("No existing vector store \
                             and no chunks provided for creation.")
        return self.vector_store

    def add_new_files_to_storage(self, data_directory: str = "./data") -> None:
        """Adds only new PDF files (not already in the vector store) to the storage.

        Scans the data directory for PDFs, checks which are not already in the store,
        and adds their chunks to the vector store.

        Args:
            data_directory (str): Path to the directory containing PDF files to check.

        Raises:
            ValueError: If the vector store is not initialized.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. \
                             Call init_or_load first.")

        collection = self.vector_store.get()
        existing_files = {os.path.basename(meta['source']) \
                          for meta in collection['metadatas']}
        data_files = [f for f in os.listdir(data_directory) \
                      if f.endswith('.pdf')]
        missing_files = [f for f in data_files if f not in existing_files]

        if not missing_files:
            logging.info("No new PDF files to add.")
            return

        extractor = PdfExtractor(data_directory)
        new_chunks = extractor.split_new_files(missing_files)
        if new_chunks:
            self.vector_store.add_documents(documents=new_chunks)
            logging.info(f"Added new files to vector store: {missing_files}")
        else:
            logging.info("No new chunks to add.")
