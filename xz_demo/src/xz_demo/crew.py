from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from langchain.chat_models import ChatOpenAI
import subprocess
import requests
import sys
import os

# import custom tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "tools")))
from custom_tool import (
    SaveCodeFileToolSWE,
    SaveCodeFileToolQA,
    ManageCodeFileTool,
    ReadFileTool,
    RunCommandTool,
    APITestTool,
)

# set up environ variables
os.environ["SERPER_API_KEY"] = "8c6bdede2899e29ceaabfb8f3ca98776b11f077a"

# kill -9 $(lsof -ti:3000) $(lsof -ti:5000)


VERBOSE = True

# Define different OpenAI models
gpt_4o_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
o1_mini = ChatOpenAI(model_name="o1-mini", temperature=0.9)


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
@CrewBase
class XzDemo:
    """XZ Demo crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def product_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["product_manager"],
            verbose=VERBOSE,
            memory=True,
            max_iter=20,
            llm=gpt_4o_mini,
            tools=[SerperDevTool(), WebsiteSearchTool()],  # able to use websearch tools
        )

    @agent
    def ux_ui_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["ux_ui_designer"],
            verbose=VERBOSE,
            memory=True,
            max_iter=3,
            llm=gpt_4o_mini,
        )

    @agent
    def software_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["software_engineer"],
            verbose=VERBOSE,
            memory=True,
            max_iter=100,
            llm=gpt_4o_mini,
            tools=[
                SaveCodeFileToolSWE(),
                ManageCodeFileTool(),
            ],
        )

    @agent
    def qa_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_engineer"],
            verbose=VERBOSE,
            memory=True,
            max_iter=50,
            llm=gpt_4o_mini,
            tools=[
                ReadFileTool(),
                RunCommandTool(),
                APITestTool(),
            ],
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def define_vision_task(self) -> Task:
        return Task(
            config=self.tasks_config["define_vision_task"],
            output_file="outputs/vision_and_roadmap.md",
            timeout=100,
        )

    @task
    def mvp_feature_task(self) -> Task:
        return Task(
            config=self.tasks_config["mvp_feature_task"],
            output_file="outputs/mvp_feature_list.md",
            timeout=100,
            tools=[
                ReadFileTool(filepath="outputs/vision_and_roadmap.md"),
                SaveCodeFileToolSWE(),
            ],
        )

    # @task
    # def user_flow_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["user_flow_task"],
    #         output_file="outputs/user_flow.md",
    #         timeout=100,
    #         tools=[ReadFileTool(filepath="outputs/mvp_feature_list.md")],
    #     )

    # @task
    # def mvp_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["mvp_task"],
    #         tools=[
    #             SaveCodeFileToolSWE(),
    #             ReadFileTool(filepath="outputs/mvp_feature_list.md"),
    #         ],
    #     )

    @task
    def mvp_think_and_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["mvp_think_and_plan_task"],
            tools=[ReadFileTool(filepath="outputs/mvp_feature_list.md")],
            output_file="outputs/mvp_build_plan.md",
        )

    @task
    def backend_build_task(self) -> Task:
        return Task(
            config=self.tasks_config["backend_build_task"],
            tools=[ReadFileTool(), SaveCodeFileToolSWE(), ManageCodeFileTool()],
        )

    @task
    def frontend_build_task(self) -> Task:
        return Task(
            config=self.tasks_config["frontend_build_task"],
            tools=[ReadFileTool(), SaveCodeFileToolSWE(), ManageCodeFileTool()],
        )

    @task
    def compile_and_document_task(self) -> Task:
        return Task(
            config=self.tasks_config["compile_and_document_task"],
            tools=[ReadFileTool(), SaveCodeFileToolSWE()],
        )

    @task
    def qa_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["qa_review_task"],
            output_file="outputs/qa_review.md",
            timeout=300,  # Stop the task after 5 minutes
        )

    # @task
    # def improve_mvp_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["improve_mvp_task"],
    #         tools=[
    #             ReadFileTool(filepath="outputs/qa_review.md"),
    #             SaveCodeFileToolSWE(),
    #             ManageCodeFileTool(),
    #         ],
    #         output_file="outputs/mvp_improvement_log.md",
    #     )

    # @task
    # def database_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["database_task"],
    #         output_file="outputs/database_schema.sql",
    #     )

    # @task
    # def deployment_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["deployment_task"],
    #         output_file="outputs/deployment_guide.md",
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the XzDemo crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
