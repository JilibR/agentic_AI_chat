from enum import Enum
from typing import Any, Dict, TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from utils.data_loader import PdfExtractor
from utils.vector_store import VectorStoreManager
from langgraph.graph.message import add_messages

import logging


logging.basicConfig(level=logging.INFO)

class ToolCategory(str, Enum):
    """
    Tool Category build to make defference between tools
    """
    PERCEUSE = "Perceuse-visseuse"
    MEULEUSE = "Meuleuse"
    PERFORATEUR = "Perforateur"
    SCIE = "Scie"
    PONCEUSE = "Ponceuse"
    INCONNU = "Inconnu"

class TechnicalValidation(BaseModel):
    """
    Class to force Agent to find attribute before giving
    a response.
    """
    product_name: str = Field(
        description="Nom du produit analysé")
    category: ToolCategory = Field(
        description="Catégogie des procution (ex: Meuleuse)")
    voltage_v: Optional[int] = Field(
        description="Tension en Volts (ex: 18)")
    battery_system: str = Field(
        description="Gamme de batterie (ex: Professional 18V)")
    is_compatible: bool = Field(
        description="Vrai si les systèmes de batterie sont identiques")
    reasoning: str = Field(
        description="Explication courte de la compatibilité")


# Init du VectorStore
def init_vector_store(file_path: str):
    pdf = PdfExtractor(file_path).load_and_split()
    vector_chroma = VectorStoreManager()
    return vector_chroma.init_or_load(pdf)


# --- Define Graph ---

class AgentState(TypedDict):
    # This 'Annotated' with 'add_messages' is crucial
    messages: Annotated[List[BaseMessage], add_messages]


# Define Agent
def agent_node(state: Dict[str, Any], model, parser) -> Dict[str, Any]:
    try:
        messages = state["messages"]
        
        # Ensure we are passing the full list to the model
        response = model.invoke(messages)
        
        # We return a list containing ONLY the new message. 
        # Thanks to 'add_messages' in AgentState, this will be 
        # automatically appended to the existing history.
        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in agent_node: {e}")
        # Return a message that explains the error without breaking the graph
        return {"messages": [AIMessage(content="Désolé, j'ai rencontré une erreur lors de l'analyse.")]}