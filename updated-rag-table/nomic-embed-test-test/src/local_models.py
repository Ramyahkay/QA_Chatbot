import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

#MODEL = "mixtral:8x7b"
MODEL = "llama3"

model = Ollama(model= MODEL)
embeddings =  OllamaEmbeddings(model = MODEL)

#print(model.invoke("Tell me a joke"))