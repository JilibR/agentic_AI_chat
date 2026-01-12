"""
Main fonction to produce Agentic Ai
"""
import logging
from enum import Enum
from typing import Any, Dict, TypedDict, Annotated, List, Optional
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from utils.data_loader import PdfExtractor
from utils.vector_store import VectorStoreManager

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
def init_vector_store(file_path: str) -> Chroma:
    pdf = PdfExtractor(file_path).load_and_split()
    vector_chroma = VectorStoreManager()
    return vector_chroma.init_or_load(pdf)


# --- Define Graph ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "history messages"]


# Define Agent
def agent_node(state: Dict[str, Any], model, parser: JsonOutputParser) -> Dict[str, Any]:
    try:
        messages = state["messages"]
        format_instructions = parser.get_format_instructions()
        system_instruction = SystemMessage(
            content=(
                "Tu es un expert technique Bosch Professional.\n"
                "CONNAISSANCES DE BASE :\n"
                "- GSR = Perceuse-visseuse (Grip, Screw, Rotary)\n"
                "- GWS = Meuleuse angulaire (Grinder, Wheel, Sanding)\n"
                "- GBH = Perforateur (Hammer)\n\n"
                "RÈGLES :\n"
                "- Si les appareils ne sont pas de la même catégorie, ils ne peuvent pas être comparés.\n"
                "- Une meuleuse ne peut pas percer des murs et une perceuse ne coupe pas.\n"
                "- Chaque catégorie a sa fonction propre et elles sont complémentaires.\n"
                "- Réponds toujours au format JSON suivant :\n"
                f"{format_instructions}"
            )
        )
        response = model.invoke([system_instruction] + messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in agent_node: {e}")
        return {"messages": [AIMessage(content="Désolé, une erreur est survenue.")]}