from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDF_Extractor:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.loader = DirectoryLoader(
            directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )

    def load_and_split(self) -> List[Document]:
        """Load and split all PDFs in the directory into chunks."""
        try:
            docs = self.loader.load()
            if not docs:
                logging.warning("No PDF files found in the directory.")
                return []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(docs)
            logging.info(f"Created {len(chunks)} chunks from PDFs.")
            return chunks
        except Exception as e:
            logging.error(f"Error loading or splitting PDFs: {e}")
            raise

    def split_new_files(self, file_paths: List[str]) -> List[Document]:
        """
        Split only the specified PDF files into chunks.
        Useful for adding new files to an existing vector store.
        """
        if not file_paths:
            logging.warning("No file paths provided.")
            return []

        chunks = []
        for file_path in file_paths:
            try:
                full_path = os.path.join(self.directory_path, file_path)
                loader = PyPDFLoader(full_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=100,
                    add_start_index=True
                )
                chunks.extend(text_splitter.split_documents(docs))
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue
        logging.info(f"Created {len(chunks)} chunks from new files.")
        return chunks
