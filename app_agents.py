from dotenv import load_dotenv
from langchain import hub
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import (
    create_python_agent,
    create_csv_agent,
)
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI
import openai
import os

load_dotenv()


def main():
    print("Start...")

    python_agent_executor = create_python_agent(
        llm=AzureChatOpenAI(
            openai_api_key="",
            azure_endpoint="",
            openai_api_type="azure",
            azure_deployment="",
            openai_api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    python_agent_executor.invoke(
        """generate and save in current working directory 15 QRcodes you have qrcode package installed already"""
    )

    csv_agent = create_csv_agent(
        llm=AzureChatOpenAI(
            openai_api_key="",
            azure_endpoint="",
            openai_api_type="azure",
            azure_deployment="",
            openai_api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )
    csv_agent.invoke(
      input={
        "input": "how many columns are there in file episode_info.csv"
      }
    )
    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main()
