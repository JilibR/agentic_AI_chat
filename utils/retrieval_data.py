from langchain.tools import tool
from utils.vector_store import VectorStoreManager

class Retriever:
    def __init__(self):
        pass

    @tool
    def get_product_specs(self, query: str, vector_store: VectorStoreManager):
        """
        Recherche les spécifications techniques et
        compatibilités des outils Bosch.
        """
        docs = vector_store.similarity_search(query, k=5)
        return "\n\n".join([doc.page_content for doc in docs])
    