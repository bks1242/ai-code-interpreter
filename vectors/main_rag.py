from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from pinecone import Pinecone

import os

load_dotenv()
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["PINECONE_API_KEY"] = ""

if __name__ == "__main__":
    print("Retrieving...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="",
        openai_api_key="",
        azure_endpoint="",
        openai_api_type="azure",
        openai_api_version="2024-05-01-preview",
    )
    llm = AzureChatOpenAI(
        openai_api_key="",
        azure_endpoint="",
        openai_api_type="azure",
        azure_deployment="",
        openai_api_version="2024-05-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    print("Creating Query....")
    query = "What is pincone in machine learning?"
    print("Getting normal answer from azure chat open ai ....")
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)
    
    pc = Pinecone(api_key="")
    print("initialising vector store ....")
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name="sample-embeddings-index")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    print("combining docs chain ....")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    print("retrieval docs chain ....")
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    
    result = retrieval_chain.invoke(input={"input": query})
    
    print(result)
    
