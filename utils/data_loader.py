import logging
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class PdfExtractor:
    """Extracts and chunks the content of PDF files in a directory.

    This class allows loading PDF files from a directory, splitting them into configurable chunks,
    and managing the addition of new files for updating a vector store.

    Attributes:
        directory_path (str): Path to the directory containing the PDF files.
        loader (DirectoryLoader): Instance of DirectoryLoader configured to load PDFs.
    """

    def __init__(self, directory_path: str):
        """Initializes the PdfExtractor with the directory path to monitor.

        Args:
            directory_path (str): Absolute or relative path to the directory containing the PDFs.
        """
        self.directory_path = directory_path
        self.loader = DirectoryLoader(
            directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )

    def load_and_split(self) -> List[Document]:
        """Loads and splits all PDFs in the directory into chunks.

        Returns:
            List[Document]: List of chunks generated from the PDFs.
                          Returns an empty list if no PDFs are found.

        Raises:
            Exception: If an error occurs during loading or splitting of PDFs.
        """
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
        """Splits only the specified PDF files into chunks.

        Useful for adding new files to an existing vector store.

        Args:
            file_paths (List[str]): List of relative file paths to process.

        Returns:
            List[Document]: List of chunks generated from the new files.
                          Returns an empty list if no file paths are provided.

        Note:
            File paths should be relative to `directory_path`.
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
