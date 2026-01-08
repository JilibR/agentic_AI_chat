from typing import List
from xml.dom.minidom import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import logging
import os
from utils.data_loader import PDF_Extractor

MODEL_EMBEDDING = "embeddinggemma:latest"

class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = MODEL_EMBEDDING
    ):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = None

    def init_or_load(self, chunks: List[Document] = None) -> Chroma:
        """
        Initialize or load the vector store. If chunks are 
        provided and no store exists, create a new one.
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

    def add_new_files_to_storage(self, 
                                 data_directory: str = "./data") -> None:
        """
        Add only new PDF files (not already in the vector store) 
        to the storage.
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

        extractor = PDF_Extractor(data_directory)
        new_chunks = extractor.split_new_files(missing_files)
        if new_chunks:
            self.vector_store.add_documents(documents=new_chunks)
            logging.info(f"Added new files to vector store: {missing_files}")
        else:
            logging.info("No new chunks to add.")
