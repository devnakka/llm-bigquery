from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from typing import List, Any, Optional, Dict
from langchain_core.documents import Document

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from backend.bigquery_vector_search import search_by_text as bq_search_by_text


PROJECT_ID = "threatexplainer"
REGION = "us-central1"
DATASET = "test_dataset"
TABLE = "helper_documents"

LLM = ChatVertexAI(model="gemini-1.5-flash-001",verbose=True, temperature=0)

class BigQueryRetriever(BaseRetriever):
    project_id: str
    region: str
    dataset: str
    table: str
    filter: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.project_id = kwargs['project_id']
        self.region = kwargs['region']
        self.dataset = kwargs['dataset']
        self.table = kwargs['table']
        self.filter = kwargs['filter']

    def get_relevant_documents(self, query: str) -> List[Document]:
        return bq_search_by_text(
            project_id=self.project_id,
            region=self.region,
            dataset=self.dataset,
            table=self.table,
            filter=self.filter,
            query=query
        )

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    filter = {"context": "langchain-docs"}
    retriever = BigQueryRetriever(project_id=PROJECT_ID, region=REGION, dataset=DATASET, table=TABLE, filter=filter)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(LLM, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=LLM, retriever=retriever, prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embedding = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project=PROJECT_ID)
    store = BigQueryVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET,
        table_name=TABLE,
        location=REGION,
        embedding=embedding,
    )
    chat = ChatVertexAI(model_name="gpt-4o", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": store.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | store.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])