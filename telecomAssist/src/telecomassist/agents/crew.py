from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from crewai.tools.base_tool import Tool 
from crewai.tools.structured_tool import CrewStructuredTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

db = SQLDatabase.from_uri("sqlite:///telecom.db")

dbtoolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o-mini", api_key=""))
dbtools = dbtoolkit.get_tools()

crewai_dbtools = []
for tool in dbtools:
    crewaitool = Tool.from_langchain(
        CrewStructuredTool.from_function(
            name=tool.name, func=tool._run, args_schema=tool.args_schema 
        )
    )
    crewai_dbtools.append(crewaitool)

@CrewBase
class Tel():
    """Tel crew"""
    @agent
    def billing_specialist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['billing_specialist_agent'],
            tools=crewai_dbtools,
            verbose=True
        )

    @agent
    def service_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['service_advisor_agent'],
            tools=crewai_dbtools,
            verbose=True
        )
    
    @task
    def billing_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['billing_analysis_task'],
        )

    @task
    def usage_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['usage_analysis_task'],
            output_file='report.md'
        )
    
    @task
    def comprehensive_response_task(self) -> Task:
        return Task(
            config=self.tasks_config['comprehensive_response_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Tel crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
