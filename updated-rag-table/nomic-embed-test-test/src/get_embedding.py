import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings