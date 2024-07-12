import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from pydantic_settings import BaseSettings
import pdfplumber


CHROMA_PATH = "/Users/hari/Desktop/updated-rag-table/nomic-embed-test-test/chroma"
DATA_PATH = "/Users/hari/Desktop/updated-rag-table/nomic-embed-test-test/data/"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    documents = []
    with pdfplumber.open(DATA_PATH + 'NVIDIAAn_full.pdf') as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"source": "NVIDIAAn_full.pdf", "page": page.page_number}))
            for table in page.extract_tables():
                for row in table:
                    cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                    documents.append(Document(page_content=' | '.join(cleaned_row), metadata={"source": "NVIDIAAn_full.pdf", "page": page.page_number}))
    save_text_to_file(documents)  
    return documents

def save_text_to_file(documents):
    with open('extracted_texts_and_tables.txt', 'w') as file:
        for doc in documents:
            file.write(f"Document: {doc.metadata.get('source', 'Unknown')} Page: {doc.metadata.get('page', 'Unknown')}\n")
            file.write(doc.page_content + "\n\n")


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()