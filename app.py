import streamlit as st
import logging
import os
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from agentic_ai import TechnicalValidation, AgentState, agent_node
from utils.data_loader import PdfExtractor
from utils.vector_store import VectorStoreManager
from dotenv import load_dotenv
from utils.config import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI


logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Bosch Tech Bot", page_icon="🛠️")
load_dotenv()

@st.cache_resource
def load_resources():
    pdf = PdfExtractor('./data').load_and_split()
    vector_chroma = VectorStoreManager()
    store = vector_chroma.init_or_load(pdf)

    parser = JsonOutputParser(pydantic_object=TechnicalValidation)
    
    # Définition de l'outil avec accès au store
    @tool
    def get_product_specs(query: str):
        """Recherche technique spécifique et compatibilité entre les outils Bosch."""
        docs = store.similarity_search(query, k=3)
        return "\n\n".join([f"Source: {doc.metadata.get('source')}\n{doc.page_content}" for doc in docs])
        
    tools = [get_product_specs]
    
    # Initialisation de Mistral via LangChain
    model_with_tools = ChatMistralAI(
        model="mistral-small-latest", 
        api_key=MISTRAL_API_KEY,
        temperature=0
    ).bind_tools(tools)
    
    # Définition du Node Agent
    def wrapped_agent_node(state):
        # On passe le parser pour que l'agent_node puisse l'utiliser dans son prompt
        return agent_node(state, model=model_with_tools, parser=parser)
    
    # Construction du Graphe
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", wrapped_agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile(), parser

# Initialisation
graph, parser = load_resources()

# --- INTERFACE STREAMLIT ---
st.title("🛠️ Bosch Tech Advisor")
st.caption("Expert en outillage professionnel et compatibilité")

# Gestion de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LOGIQUE DE CHAT ---
if prompt := st.chat_input("Posez votre question technique..."):
    # Affichage message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    format_instructions = parser.get_format_instructions()

    # Préparation de l'état initial pour LangGraph
    state = AgentState(
        messages=[
            SystemMessage(
                content=(
                    "Tu es un expert technique Bosch Professional. "
                    "Utilise l'outil 'get_product_specs' pour les recherches.\n"
                    "CONNAISSANCES références: GSR=Perceuse, GWS=Meuleuse, GBH=Perforateur, GKS=Scie circulaire, GMP=Humidimètre.\n"
                    "Si tu ne connais pas la réponse avec tes connaissances n'invente rien"
                    "Tu fourniras toujours la référence de tes connaissances par exemple GSR pour une perceuse"
                    f"Réponds toujours au format JSON suivant :\n{format_instructions}"
                )
            ),
            HumanMessage(content=prompt),
        ]
    )  

    # Exécution et streaming
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        final_response_content = ""
        
        # On parcourt les étapes du graphe
        logging.info(f"Agent is thinking ...")
        for event in graph.stream(state):
            for key, value in event.items():
                if key == "tools":
                    status_placeholder.info("🔍 Recherche dans la base documentaire Bosch...")
                elif key == "agent":
                    logging.info(f"\n[Agent]: {value['messages'][-1].content}")
                    status_placeholder.empty()
                    final_response_content = value["messages"][-1].content

        # Parsing final et affichage
        try:
            parsed_response = parser.parse(final_response_content)
            # On récupère le champ 'reasoning' ou 'answer' selon ta classe TechnicalValidation
            clean_answer = parsed_response.get("reasoning", final_response_content)
            st.markdown(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})
        except Exception as e:
            # Fallback si le JSON est mal formé
            st.markdown(final_response_content)
            st.session_state.messages.append({"role": "assistant", "content": final_response_content})

with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Vider la conversation"):
        st.session_state.messages = []
        st.rerun()