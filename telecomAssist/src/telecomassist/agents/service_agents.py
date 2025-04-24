from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.tools.render import render_text_description
from langchain.schema import SystemMessage
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
load_dotenv()

# Define a prompt template for service recommendations
SERVICE_RECOMMENDATION_TEMPLATE = """You are a telecom service advisor who helps customers find the best plan for their needs.
 
    When recommending plans, consider:
    1. The customer's usage patterns (data, voice, SMS)
    2. Number of people/devices that will use the plan
    3. Special requirements (international calling, streaming, etc.)
    4. Budget constraints
 
    Always explain WHY a particular plan is a good fit for their needs.
 
    You have access to the following tools:
 
    {tools}
 
    Use the following format:
 
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
 
    Question : {input}
    Thought : {agent_scratchpad}
"""

def create_service_agent():
    """Create and return a LangChain agent for service recommendations"""
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )
   
    # Create database and tools
    db = SQLDatabase.from_uri("sqlite:///telecom.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
   
    # Extract the actual tools from the toolkit
    tools = toolkit.get_tools()
   
    # Add Python REPL tool
    tools.append(PythonREPLTool())
   
   
    prompt = PromptTemplate.from_template(template=SERVICE_RECOMMENDATION_TEMPLATE)
      
   
    # Create the ReAct agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
   
    # Create the AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=30
    )
   
    return agent_executor
 
def process_recommendation_query(query):
    print(query)
    """Process a service recommendation query using the LangChain agent"""
    # Create the service agent
    agent = create_service_agent()
    
    # Process the query and handle any errors
    try:
        result = agent.invoke({"input": query})
        agent_response = result.get("output") 
    except Exception as e:
        return f"Error processing recommendation: {str(e)}"
    
    # Return the raw agent response (or format as needed)
    return agent_response




