import os

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from backend.bigquery_vector import create_vectors as bq_create_vectors

PROJECT_ID = "threatexplainer"
REGION = "us-central1"
DATASET = "test_dataset"
TABLE = "helper_documents"



def ingest_docs2(context="langchain-docs" , truncate_all=False):
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url, "context": context, "length": len(doc.page_content)})

    print(f"Going to add {len(documents)} to BigQueryVectorStore")

    metadata = [d.metadata for d in documents]

    bq_create_vectors(
        project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE,
        metadata=metadata, texts=[d.page_content for d in documents], truncate=truncate_all
    )

    print("****Added details to Bigquery vectorstore done ***")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to BigQueryVectorStore")

    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )

    store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET,
        table_name=TABLE,
        location=REGION,
        embedding=embedding,
    )

    store.add_documents(documents)

    print("****Added details to Bigquery vectorstore done ***")

if __name__ == '__main__':
    ingest_docs2()