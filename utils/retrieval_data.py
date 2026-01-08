from langchain.tools import tool
from utils.vector_store import VectorStoreManager

class Retriever:
    def __init__(self):
        pass

    @tool
    def get_product_specs(query: str, vector_store: VectorStoreManager):
        """Recherche les spécifications techniques et compatibilités des outils Bosch."""
        docs = vector_store.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    

    #@tool
    #def validate_specs(technical_data: TechnicalValidation):
    #    """
    #    Utilise cet outil pour valider et structurer les specs techniques 
    #    AVANT de donner une réponse finale au client.
    #    """
    #    # Ici, l'agent envoie un objet structuré, pas juste du texte
    #    return f"Données validées pour {technical_data.product_name}"