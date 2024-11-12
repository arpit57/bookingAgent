# policy_agent.py
import os
from typing import Dict, Any, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DocumentSearchAgent:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the search agent with existing Chroma vector store"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "chroma_db")
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(
            collection_name="ecommerce_docs",
            embedding_function=self.embeddings,
            persist_directory=db_path
        )
        
        # Initialize LLM and prompt
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = PromptTemplate(
            template="""Given the following context about our e-commerce policies and the user's question, 
            provide a brief, direct answer using only the information provided in the context. 
            Keep the response concise and to the point.

            Context:
            {context}

            Question: {question}

            Provide a concise response focusing only on the relevant information:""",
            input_variables=["context", "question"]
        )
        
        # Create the runnable chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def search_documents(self, query: str, k: int = 2) -> List[Document]:
        """Search documents based on the query"""
        results = self.vector_store.similarity_search(
            query,
            k=k
        )
        return results

    def generate_response(self, question: str, context: str) -> str:
        """Generate a concise response using LLM"""
        try:
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response from context."

def search_docs(user_input: str) -> str:
    """Main function to process search requests"""
    try:
        search_agent = DocumentSearchAgent()
        
        # Get relevant chunks
        results = search_agent.search_documents(
            query=user_input,
            k=2
        )
        
        if not results:
            return "No relevant information found."
            
        # Combine chunks for context
        context = "\n".join(doc.page_content for doc in results)
        
        # Generate concise response
        response = search_agent.generate_response(user_input, context)
        return response
        
    except Exception as e:
        print(f"Error in search_docs: {e}")
        return "Error retrieving information from company policies."

if __name__ == "__main__":
    # Test queries
    queries = [
        "What payment methods do you accept?",
        "Tell me about your shipping options",
        "What is your return policy?",
        "Do you offer EMI?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = search_docs(query)
        print(f"Result: {result}")