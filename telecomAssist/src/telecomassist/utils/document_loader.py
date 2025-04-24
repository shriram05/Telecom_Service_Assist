from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core import Settings


# embed_model  = OpenAIEmbedding()
# Settings.embed_model = embed_model

import chromadb


class DocumentLoader:
    def __init__(self):
        chroma_client = chromadb.PersistentClient(path="data/chroma")
        chroma_collection = chroma_client.get_or_create_collection("informationDocuments")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        pipeline = IngestionPipeline(
        transformations=[
                SentenceSplitter(chunk_size=500, chunk_overlap=50),
                TitleExtractor(),
                ]
            )
        documents = SimpleDirectoryReader(input_dir="data/documents",recursive=True).load_data(num_workers=0)
        # run the pipeline
        nodes = pipeline.run(documents=documents)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        index.storage_context.persist()


    def process_uploads(self, temp_dir):
        chroma_client = chromadb.PersistentClient(path="data/chroma")
        chroma_collection = chroma_client.get_or_create_collection("informationDocuments")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        pipeline = IngestionPipeline(
        transformations=[
                SentenceSplitter(chunk_size=500, chunk_overlap=50),
                TitleExtractor(),
                ]
            )
        documents = SimpleDirectoryReader(input_dir="D:/Telecom Service Assistant/telecomAssist/src/telecomassist/data/temp_dir",recursive=True).load_data(num_workers=0)
        # run the pipeline
        nodes = pipeline.run(documents=documents)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        index.storage_context.persist()
        












