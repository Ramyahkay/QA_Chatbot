import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding import get_embedding_function

CHROMA_PATH = "/Users/hari/Desktop/updated-rag-table/nomic-embed-test-test/src/chroma"

PROMPT_TEMPLATE = """
Please provide a response using only the exact terms and technologies mentioned in the context below and make sure to cross check if all the information you provide is in the document. If not do not state any other information outside the document else answer the question based on the context with information that is only present in the document. However if the question is to analyze the context then feel free to analyze and give a response based only of the document:

{context}

Question: {question}
"""

print("Loading model...")
MODEL = Ollama(model="llama3")

print("Warming up the model...")
MODEL.invoke("Hello, world!")

def main():
    print("Preparing DB...")
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    print("Ask a question (type 'exit' to quit):")
    while True:
        query_text = input(
        if query_text.lower() == "exit":
            break
        response_text = query_rag(query_text, db)
        print(f"Response: {response_text}\n")
        print("Ask another question (type 'exit' to quit):")

def query_rag(query_text: str, db):
    print("Searching DB...")
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Invoking model...")
    response_text = MODEL.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
