from typing import TypedDict, Dict, Any, List
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from agents.billing_agents import process_billing_query
from agents.service_agents import process_recommendation_query
from agents.knowledge_agents import process_knowledge_query
from agents.network_agents import  process_network_query
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

# Define the state structure
class TelecomAssistantState(TypedDict):
    query: str                            # The user's original query
    customer_info: Dict[str, Any]         # Customer information if available
    classification: str                   # Query classification
    intermediate_responses: Dict[str, Any] # Responses from different nodes
    final_response: str                   # Final formatted response
    chat_history: List[Dict[str, str]]    # Conversation history
 
# Classification node - determines query type
def classify_query(state: TelecomAssistantState) -> TelecomAssistantState:
    """Classify the query into different categories"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    promptTemplate = ChatPromptTemplate(
        [
            ("system", """Classify the following customer telecom query into exactly one category:

        CATEGORIES:
        - billing_account: Billing issues, payments, charges, account information
        - network_troubleshooting: Connection problems, signal issues, data speeds, outages
        - service_recommendation: Plan selection, upgrades, new services, family plans
        - knowledge_retrieval: Technical information, setup instructions, device compatibility.Return ONLY the category name and nothing else (e.g., "billing_account")
             If there is nothing related to the above categories, go to fallback_handler function.
             """
        
             ),
            ("human", "{query}")
        ]
    )
    parser = StrOutputParser()
    
    
    query = state["query"]
    chain = promptTemplate | llm | parser
    classification = chain.invoke({"query":query})
    print(classification)


    return {**state, "classification": classification}
    
# Routing function - determines next node based on classification
def route_query(state: TelecomAssistantState) -> str:
    """Route the query to the appropriate node based on classification"""
    classification = state["classification"]
    print(f"Routing based on classification: {classification}")
    
    if classification == "billing_account":
        return "crew_ai_node"
    elif classification == "network_troubleshooting":
        return "autogen_node"
    elif classification == "service_recommendation":
        return "langchain_node"
    elif classification == "knowledge_retrieval":
        return "llamaindex_node"
    else:
        return "fallback_handler"  # Default fallback for unrecognized classifications
 
# Node function templates for each framework
def crew_ai_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle billing and account queries using CrewAI"""
    # Extract customer email from customer_info if available
    customer_email = state.get("customer_info", {}).get("email", "")
    query = state["query"]
    
    response = process_billing_query(customer_email, query)
    return {**state, "intermediate_responses": {"crew_ai": response}}
    
def autogen_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle network troubleshooting using AutoGen"""
    # This would be replaced with actual AutoGen implementation
    query = state["query"]
    customer_email = state.get("customer_info", {}).get("email", "")
    response = process_network_query(query, customer_email)
    return {**state, "intermediate_responses": {"autogen": response}}
 
def langchain_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle service recommendations using LangChain"""
    query = state["query"]
    response = process_recommendation_query(query)
    return {**state, "intermediate_responses": {"langchain": response}}
 
def llamaindex_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle knowledge retrieval using LlamaIndex"""
    # This would be replaced with actual LlamaIndex implementation
    query = state["query"]
    response = process_knowledge_query(query)
    return {**state, "intermediate_responses": {"llamaindex": response}}
 
def fallback_handler(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle queries that don't fit other categories"""
    response = "I'm not sure how to help with that specific question. Could you try rephrasing or ask about our services, billing, network issues, or technical support?"
    return {**state, "intermediate_responses": {"fallback": response}}
 
def formulate_response(state: TelecomAssistantState) -> TelecomAssistantState:
    """Create final response from intermediate results"""
    # Get the response from whichever node was called
    intermediate_responses = state["intermediate_responses"]
   
    # Extract the first response we find (in a real implementation, you'd format this better)
    response_value = next(iter(intermediate_responses.values()))
   
    return {**state, "final_response": response_value}
 
def create_graph():
    """Create and return the workflow graph"""
    # Build the graph
    workflow = StateGraph(TelecomAssistantState)
   
    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("crew_ai_node", crew_ai_node)
    workflow.add_node("autogen_node", autogen_node)
    workflow.add_node("langchain_node", langchain_node)
    workflow.add_node("llamaindex_node", llamaindex_node)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("formulate_response", formulate_response)
   
    # Add conditional edges from classification to appropriate node
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "crew_ai_node": "crew_ai_node",
            "autogen_node": "autogen_node",
            "langchain_node": "langchain_node",
            "llamaindex_node": "llamaindex_node",
            "fallback_handler": "fallback_handler"
        }
    )
   
    # Add edges from each processing node to response formulation
    workflow.add_edge("crew_ai_node", "formulate_response")
    workflow.add_edge("autogen_node", "formulate_response")
    workflow.add_edge("langchain_node", "formulate_response")
    workflow.add_edge("llamaindex_node", "formulate_response")
    workflow.add_edge("fallback_handler", "formulate_response")
    workflow.add_edge("formulate_response", END)
   
    # Set the entry point
    workflow.set_entry_point("classify_query")
   
    # Compile the graph
    return workflow.compile()