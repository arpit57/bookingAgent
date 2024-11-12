# document_embedder.py
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file_path: str) -> str:
    """Load document content from file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_embeddings(file_path: str, persist_directory: str = "./chroma_db"):
    """Create and persist embeddings for the document"""
    # Load document
    text = load_document(file_path)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create Document objects
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "shopSphere.txt", "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Remove existing vector store if it exists
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
    
    vector_store = Chroma(
        collection_name="ecommerce_docs",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Add documents
    vector_store.add_documents(documents)
    
    print(f"Created {len(documents)} chunks and stored them in the vector store")
    
    # Print first few chunks for verification
    print("\nFirst few chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(chunk)
        print("-" * 50)

if __name__ == "__main__":
    file_path = "shopSphere.txt"  # Update with your file path
    create_embeddings(file_path)