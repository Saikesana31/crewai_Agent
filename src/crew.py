from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import CSVSearchTool,FileReadTool
from typing import List, Dict, Any
from dotenv import load_dotenv

# CrewBase is a base class for creating crews
@CrewBase
class MatchtoJobsCrew():
    """Crew for matching CVs to job opportunities."""
    agents: List[BaseAgent]
    tasks: List[Task]

    # Path to the agents and tasks configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Tools to use in the crew
    tools = [CSVSearchTool(), FileReadTool()]


    # Returns an Agent object
    @agent
    def cv_reader(self) -> Agent:
        return Agent(
            config=self.agents_config['cv_reader'],
            tools = [self.tools[1]],  # FileReadTool is at index 1
            verbose = True, # For debugging
            allow_delegation = False 
        )


    @agent
    def matcher(self) -> Agent:
        return Agent(
            config = self.agents_config['matcher'],
            tools = [self.tools[0]],  # CSVSearchTool is at index 0
            verbose = True,
            allow_delegation = False,
        )

    # Returns a Task object
    @task
    def read_cv_task(self) -> Task:
        return Task(
            config = self.tasks_config['read_cv_task'],
            agent = self.cv_reader(),
            
        )

    @task
    def match_cv_task(self) -> Task:
        return Task(
            config = self.tasks_config['match_cv_task'],
            agent = self.matcher(),
            
        )

    # Returns a Crew object
    @crew
    def crew(self) -> Crew:
        """ Crew for matching CVs to job opportunities. """
        return Crew(
            agents = self.agents, # automatically load the agents from the @agent decorators
            tasks = self.tasks, # automatically load the tasks from the @task decorators
            process = Process.sequential,
            verbose = True
        )