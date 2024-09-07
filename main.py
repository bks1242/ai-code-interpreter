from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_core.tools import Tool
import openai
import os


load_dotenv()

# Retrieve Azure OpenAI specific configuration from environment variables
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Set the OpenAI library configuration using the retrieved environment variables
openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY


def main():
    print("start..")
    
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        tools=tools,
        llm=AzureChatOpenAI(
           
        ),
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)
    python_agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRcodes
                                that point to https://localhost:4203, you have qrcode package installed already"""
        }
    )
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=AzureChatOpenAI(
            openai_api_key="b2049d8a815e48928884676cf12aae68",
            azure_endpoint="https://testtcmodel.openai.azure.com/",
            openai_api_type="azure",
            azure_deployment="GPT4TestCaseCreationModel",
            openai_api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    ############# CSV AGENT ##############################
    # csv_agent_executor: AgentExecutor = create_csv_agent(
    #     llm=AzureChatOpenAI(
    #        
    #     ),
    #     path="episode_info.csv",
    #     verbose=True,
    #     allow_dangerous_code=True,
    # )
    
    
    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=AzureChatOpenAI(
            openai_api_key="b2049d8a815e48928884676cf12aae68",
            azure_endpoint="https://testtcmodel.openai.azure.com/",
            openai_api_type="azure",
            azure_deployment="GPT4TestCaseCreationModel",
            openai_api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ),
        allow_dangerous_code=True,
        path="episode_info.csv",
        verbose=True,
    )

    ################################### Router Agent ###########################
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, any]:
        return python_agent_executor.invoke({"input": original_prompt})
      
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                          returning the results of the code execution
                          DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                         takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
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
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)
    # prompt = base_prompt.partial(instructions="")
    # grand_agent = create_react_agent(
    #     prompt=prompt,
    #     tools=grand_tools,
    #     llm=AzureChatOpenAI(
    #     ),
    # )

    print(
        grand_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )
    
    print(
        grand_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `https://localhost:4203`",
            }
        )
    )


if __name__ == "__main__":
    main()
