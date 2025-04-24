import autogen
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
import chromadb
import asyncio
import os

# Import Langchain components for database tools
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_chroma import Chroma


def create_network_agents(orig_query):
    """Create and return an AutoGen group chat for network troubleshooting"""

    llm_config = {
        "config_list": [{"model": "gpt-4o-mini","temperature": 0.2,"api_key": os.environ["OPENAI_API_KEY"]}],
    }

    def search_docs(query):
        results = []
        for item in retriever.get_relevant_documents(query):
            results.append(item)
        return results
    
    db_uri = "sqlite:///telecom.db"
    db = SQLDatabase.from_uri(db_uri)
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_tools = []
    for tool in toolkit.get_tools():
        db_tools.append(LangChainToolAdapter(tool))

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    chroma_client = chromadb.PersistentClient(path="data/chroma")
    chroma_collection = chroma_client.get_or_create_collection("informationDocuments")

    vectorstore = Chroma(
        persist_directory="data/chroma"
    )

    retriever = vectorstore.as_retriever()

    retrieval_tool = Tool(
        name = "Vector Store Retrieval",
        description="Searches technical documentation for network troubleshooting guides.",
        func=search_docs
    )

    network_desc = """
    You are a network diagnostics expert who analyzes connectivity issues.
    Your responsibilities:
    1. Check for known outages or incidents in the customer's area
    2. Analyze network performance metrics
    3. Identify patterns that indicate specific network problems
    4. Determine if the issue is widespread or localized to the customer
    
    Always begin by checking the network status database for outages in the 
    customer's region before suggesting device-specific solutions.
    """

    network_diagnostics_agent = AssistantAgent(
        name="NetworkDiagnosticsAgent",
        description="Expert in network diagnostics and outage detection",
        model_client=model_client,
        reflect_on_tool_use=True,
        tools=db_tools,
        system_message=network_desc
    )
    
    # TODO: Create the DeviceExpertAgent
    # - Focus on device-specific troubleshooting
    # - Give access to technical documentation
    # Example system message:
    device_desc = """
    You are a device troubleshooting expert who knows how to resolve 
    connectivity issues on different phones and devices.
    Your responsibilities:
    1. Suggest device-specific settings to check
    2. Provide step-by-step instructions for configuration
    3. Explain how to diagnose hardware vs. software issues
    4. Recommend specific actions based on the device type
    
    Always ask for the device model if it's not specified, as troubleshooting
    steps differ between iOS, Android, and other devices.
    """
    device_expert_agent = AssistantAgent(
        name="DeviceExpertAgent",
        description="Expert in device-specific network configurations and troubleshooting",
        model_client=model_client,
        reflect_on_tool_use=True,
        tools=db_tools,
        system_message=device_desc
    )

    
    # TODO: Create the SolutionIntegratorAgent
    # - Focus on combining insights into a coherent action plan
    # - Prioritize solutions by likelihood of success and ease of implementation
    # Example system message:
    sol_desc = """
    You are a solution integrator who combines technical analysis into
    actionable plans for customers.
    Your responsibilities:
    1. Synthesize information from the network and device experts
    2. Create a prioritized list of troubleshooting steps
    3. Present solutions in order from simplest to most complex
    4. Estimate which solution is most likely to resolve the issue
    
    Your final answer should always be a numbered list of actions the customer
    can take, starting with the simplest and most likely to succeed.
    """

    solution_integration_agent = AssistantAgent(
        name="SolutionIntegrationAgent",
        description="Expert in synthesizing technical analysis into formulable plans.",
        model_client=model_client,
        reflect_on_tool_use=True,
        tools=db_tools,
        system_message=sol_desc
    )

    
    # TODO: Set up the GroupChat
    # - Include all agents in the group
    # - Configure the chat flow for collaborative problem-solving
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=30)
    termination = text_mention_termination | max_messages_termination

    # Define selector prompt

    team = SelectorGroupChat(
        [network_diagnostics_agent, device_expert_agent, solution_integration_agent],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row
    )
    
    return team


async def _process_network_query(query,customer_details):
    """Process a network troubleshooting query using AutoGen agents"""
    # TODO: Create network agents using create_network_agents function
    team = create_network_agents(query)
    task = f"""Network Troubleshooting Request:
    {query}
    {customer_details}


    With the available tools, check if there're any ongoing outages that could cause disruption.
    Please analyze this network issue, identify possible causes, and provide a step-by-step
    troubleshooting plan that the customer can follow. Your response should be comprehensive
    but easy to understand for non-technical users.

    """
    # TODO: Initiate the troubleshooting process
    # - Start the conversation with the user's query
    # - Let the agents collaborate to analyze and solve the problem
    print("started")
    res = await team.run(task=task)
    all_messages = res.messages
    print("ended")
    final_response = ""

    #Find the solution integrator's final response
    for message in reversed(all_messages):
        if message.source == "SolutionIntegrationAgent" and "TROUBLESHOOTING COMPLETE" in message.content:
            # Clean up the response (remove the TROUBLESHOOTING COMPLETE marker)
            final_response = message.content.replace("TROUBLESHOOTING COMPLETE", "").strip()
            break

    # If we couldn't find a clean final response, use the last message from the solution integrator
    if not final_response:
        for message in reversed(all_messages):
            if message.source == "SolutionIntegrationAgent":
                final_response = message.content
                break

    # Format the final response
    formatted_response = f"""
## Network Troubleshooting Plan

{final_response}

---
*This troubleshooting plan was created by our Network Support Team based on the information provided.*
"""

    return formatted_response
    # return all_messages

def process_network_query(query,customer_details):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop=loop)
    try:
        result = loop.run_until_complete(_process_network_query(query, customer_details))
        return result
    finally:
        loop.close()

