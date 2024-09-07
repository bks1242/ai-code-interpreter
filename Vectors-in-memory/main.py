import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


if __name__ == "__main__":
    print("Hii....")
    pdf_path = "Vectors-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

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

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    
    new_vectorstore = FAISS.load_local(
      "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
      llm, retrieval_qa_chat_prompt
    )
    
    retrieval_chain = create_retrieval_chain(
      new_vectorstore.as_retriever(), combine_docs_chain
    )
    
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    
    print(res["answer"])
