import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()


# index = pc.Index("sample-embeddings-index")
# index.upsert(
#     vectors=[
#         {
#             "id": "vec1", 
#             "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
#             "metadata": {"genre": "drama"}
#         }, {
#             "id": "vec2", 
#             "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
#             "metadata": {"genre": "action"}
#         }, {
#             "id": "vec3", 
#             "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
#             "metadata": {"genre": "drama"}
#         }, {
#             "id": "vec4", 
#             "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 
#             "metadata": {"genre": "action"}
#         }
#     ],
#     namespace= "ns1"
# )
# pc = Pinecone(api_key="")

def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="",
        openai_api_key="",
        azure_endpoint="",
        openai_api_type="azure",
        openai_api_version="2024-05-01-preview",
    )
    # text = "this is a test document"
    # query_result = embeddings.embed_query(text)
    # print(query_result)
    # doc_result = embeddings.embed_documents([text])
    # print(doc_result[0][:5])
    print("loading the data....")
    loader = TextLoader("vectors\medium-blog.txt")
    document = loader.load()
    
    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    print("Real Ingestion...")
    pc = Pinecone(api_key="")
    PineconeVectorStore.from_documents(texts, embeddings, index_name="sample-embeddings-index")
    print("Finish")

if __name__ == "__main__":
    print("Ingesting....")
    main()
