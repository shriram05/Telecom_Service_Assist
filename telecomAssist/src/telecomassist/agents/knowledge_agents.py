from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer, SQLDatabase
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.vector_stores.chroma import ChromaVectorStore
import pandas as pd
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()

def create_knowledge_engine():
    """Create and return a LlamaIndex query engine for knowledge retrieval"""
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    chroma_client = chromadb.PersistentClient(path="C:/Users/shriramkumar.an/Desktop/Telecom Service Assistant/telecomAssist/src/telecomassist/data/chroma")
    chroma_collection = chroma_client.get_or_create_collection("informationDocuments")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
    vector_query_engine = vector_index.as_query_engine(
            similarity_top_k=3,
            storage_context=storage_context,
            response_mode="compact"  # For concise answers
        )

    db_path = "C:/Users/shriramkumar.an/Desktop/Telecom Service Assistant/telecomAssist/telecom.db"
    sqlite_uri = f"sqlite:///{db_path}"
    sql_database = SQLDatabase.from_uri(sqlite_uri)
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
    )
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=("""You are an expert in converting natural language questions about telecom services"
                      " into SQL queries. The database contains tables for coverage areas, device compatibility,"
                      " and technical specifications."
                      "When writing SQL: 1. Use coverage_quality table for location-based questions"
                      " 2. Use device_compatibility for phone-specific inquiries 3. Use common_network_issues for network technology questions"
                      " Write focused queries that only retrieve the columns needed to answer the question.""")
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            f"Useful for answering semantic questions from the documents related to company policies and frequently asked questions"
        ),
    )
    # RouterQueryEngine to handle routing based on question type
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[sql_tool, vector_tool]
    )
    
    return router_query_engine  # Return the query engine

def process_knowledge_query(query):
    """Process a knowledge retrieval query using the LlamaIndex query engine"""
    agent = create_knowledge_engine()  # Get the query engine

    # Perform the query using the agent
    response = agent.query(query)
    return response.response  # Return the response from the query

