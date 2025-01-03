import os
import pandas as pd
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ProductSearchAgent:
    def __init__(self, csv_path: str = "products.csv"):
        """Initialize the search agent with products CSV data"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(script_dir, csv_path)
        
        # Load products data
        self.products_df = pd.read_csv(self.csv_path)
        
        # Initialize LLM and prompt
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = PromptTemplate(
            template="""Given the following product data and the user's query, provide a concise response 
            with only the relevant information. Format the response in a clear, easy-to-read manner.
            Only include attributes that are relevant to the user's query.

            Product Data:
            {product_data}

            User Query: {question}

            Guidelines:
            - Only include relevant product attributes
            - make sure to extract the product id
            - Format prices with currency symbol
            - Show stock status only if relevant
            - Keep descriptions brief and relevant
            - If multiple products, use clear separation
            - Only use information present in the product data

            Provide a concise response:""",
            input_variables=["product_data", "question"]
        )
        
        # Create the runnable chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def search_products(self, query: str) -> pd.DataFrame:
        """Search products based on the query"""
        # Convert query and product data to lowercase for better matching
        query = query.lower()
        
        # Create masks for different search criteria
        category_mask = self.products_df['Category'].str.lower().str.contains(query, na=False)
        subcategory_mask = self.products_df['Subcategory'].str.lower().str.contains(query, na=False)
        name_mask = self.products_df['ProductName'].str.lower().str.contains(query, na=False)
        brand_mask = self.products_df['Brand'].str.lower().str.contains(query, na=False)
        description_mask = self.products_df['Description'].str.lower().str.contains(query, na=False)
        
        # Combine all masks with OR operation
        combined_mask = (
            category_mask | 
            subcategory_mask | 
            name_mask | 
            brand_mask | 
            description_mask
        )
        
        # Return matching products
        results = self.products_df[combined_mask]
        
        # If no exact matches, try fuzzy matching or broader criteria
        if len(results) == 0:
            # Example: Check if any words from the query match
            query_words = query.split()
            for word in query_words:
                word_mask = (
                    self.products_df['Category'].str.lower().str.contains(word, na=False) |
                    self.products_df['Subcategory'].str.lower().str.contains(word, na=False) |
                    self.products_df['ProductName'].str.lower().str.contains(word, na=False) |
                    self.products_df['Brand'].str.lower().str.contains(word, na=False)
                )
                results = self.products_df[word_mask]
                if len(results) > 0:
                    break
        
        return results.head(5)  # Limit to top 5 matches

    def generate_response(self, question: str, products_data: pd.DataFrame) -> str:
        """Generate a concise response using LLM"""
        try:
            if products_data.empty:
                return "No matching products found."
                
            # Convert products data to string format
            products_str = products_data.to_string()
            
            response = self.chain.invoke({
                "product_data": products_str,
                "question": question
            })
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response about products."

def search_products(user_input: str) -> str:
    """Main function to process product search requests"""
    try:
        search_agent = ProductSearchAgent()
        
        # Get relevant products
        results = search_agent.search_products(user_input)
        
        if results.empty:
            return "No matching products found."
            
        # Generate concise response
        response = search_agent.generate_response(user_input, results)
        return response
        
    except Exception as e:
        print(f"Error in search_products: {e}")
        return "Error retrieving product information."

if __name__ == "__main__":
    # Test queries
    queries = [
        "what all smartphones are available?",
        "any cookwares in stock?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = search_products(query)
        print(f"Result: {result}")