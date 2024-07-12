from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from local_models import model, embeddings
from operator import itemgetter

loader = PyPDFLoader("/Users/hari/Downloads/DassaultSys_document_qa_chatbot/data/NVIDIAAn.pdf")
pages = loader.load_and_split()
# print(pages)



template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = prompt | model 

print(chain.invoke({"context": "My parents named me Santiago", "question": "What's your name'?"}))



vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()
retriever.invoke("revenue")



chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
   
)

questions = [
    "What was NVIDIA's total revenue for the first quarter of Fiscal 2024?",
    "How much did NVIDIA's revenue increase from the previous quarter?",
    "What was the Data Center revenue for the first quarter?",
    "According to Jensen Huang, what major transition is expected within global data center infrastructure?"
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print()