import os
import logging
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from utils.data_loader import PDF_Extractor
from utils.vector_store import VectorStoreManager
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from enum import Enum

class ToolCategory(str, Enum):
    PERCEUSE = "Perceuse-visseuse"
    MEULEUSE = "Meuleuse"
    PERFORATEUR = "Perforateur"
    INCONNU = "Inconnu"

class TechnicalValidation(BaseModel):
    product_name: str = Field(description="Nom du produit analysé")
    category: ToolCategory = Field(description="Catégogie des procution (ex: Meuleuse)")
    voltage_v: Optional[int] = Field(description="Tension en Volts (ex: 18)")
    battery_system: str = Field(description="Gamme de batterie (ex: Professional 18V)")
    is_compatible: bool = Field(description="Vrai si les systèmes de batterie sont identiques")
    reasoning: str = Field(description="Explication courte de la compatibilité")


parser = JsonOutputParser(pydantic_object=TechnicalValidation)

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
model = ChatOllama(model="gemma3:4b", temperature=0)

# Init du VectorStore
pdf = PDF_Extractor('./data').load_and_split()
vector_chroma = VectorStoreManager()
vector_store = vector_chroma.init_or_load(pdf)

# --- DEFINITION DE L'OUTIL ---
@tool
def get_product_specs(query: str):
    """Specific technical & compatibility reaseah between Bosch Tools."""
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [get_product_specs]

# --- Define Graph ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "history messages"]


# Define Agent
def agent_node(state: AgentState):

    messages = state["messages"]
    # On prépare les instructions de formatage JSON de Pydantic
    format_instructions = parser.get_format_instructions()
    # Intermediate instruction to 
    system_instruction = SystemMessage(
        content=f"Tu es un expert technique Bosch Professional."
            "CONNAISSANCES DE BASE : \n "
            "- GSR = Perceuse-visseuse (Grip, Screw, Rotary)\n"
            "- GWS = Meuleuse angulaire (Grinder, Wheel, Sanding)\n"
            "- GBH = Perforateur (Hammer)\n\n"
            "RÈGLE : Si les appareils ne sont pas de la meme catégorie, ils ne peuvent pas être comparé\n"
            "Une meuleuse ne peut pas percé des murs et une perceuse ne coupe pas\n"
            "Chaque catégorie à sa fonction"
            "elles sont complémentaires. L'une perce, l'autre coupe/meule."
            "Réponds toujours au format JSON suivant :\n"
            f"{format_instructions}\n"
            )
    # Called agent with Instructions + history
    response = model.invoke([system_instruction] + messages)
    
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes for the agent and tools
workflow.add_node("agent", agent_node)
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Define edges
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

# Set entry point
workflow.set_entry_point("agent")

# Compile the graph
graph = workflow.compile()

query ="J'aimerai scier du bois pour une cuisine pourrais-tu me dis quel modele serais adapté?"
query2 = "J'aimera que que me dise quel sera la plus utile pour un petit bricoleur et pour percer."


queries = [query]
# Initialize the state
state = AgentState(
    messages=[
        SystemMessage(
            content=(
                "Tu es un expert technique Bosch Professional. "
                "Utilise l'outil 'get_product_specs' pour répondre\
                      aux questions sur les outils."
                "Si tu ne trouves pas l'info, ne l'invente pas."
                "Sois précis dans tes réponses"
            )
        ),
        HumanMessage(content=queries),
    ]
)

print(f"[Agent] is thinking ...")
for event in graph.stream(state):
    for key, value in event.items():
        if key == "tools":
            print(f"\n[Outils appelés: {value[0].tool}]")
            print(f"Entrée: {value[0].tool_input}")
        elif key == "agent":
            print(f"\n[Agent]: {value['messages'][-1].content}")
            final_response = value["messages"][-1].content
        elif key == "__end__":
            final_response = value["messages"][-1].content
            print("\n--- Final Response ---")
            print(value["messages"][-1].content)

try:
    parsed_response = parser.parse(final_response)
    print("\n--- Reasoning Responsee ---")
    print(parsed_response["reasoning"])
except Exception as e:
    print(f"\nPArsinf Error : {e}. Unstructure Response : {final_response}")