from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from enum import Enum
import json

from specialized_agents.policy_agent import search_docs
from specialized_agents.booking_agent import book_appointment
from specialized_agents.products_agent import search_products

# Define the possible actions
class AgentAction(str, Enum):
    POLICY_QUERY = "policy_query"
    BOOKING = "booking"
    DB_QUERY = "db_query"
    FINAL_RESPONSE = "final_response"

# Define state
class GraphState(TypedDict):
    input: str
    action: str
    response: str
    context: dict
    final_response: str

# Initialize OpenAI client
client = OpenAI()

def primary_agent(state: GraphState):
    """
    Primary agent that decides the next action based on user input
    """
    prompt = f"""Given the user input, determine the most appropriate action:
    - If the query is about shop sphere(an e-commerce company) or anything that could be related to it, respond with '{AgentAction.POLICY_QUERY}'
    - If the query is about scheduling or booking appointments, respond with '{AgentAction.BOOKING}'
    - If the query is about a product information, respond with '{AgentAction.DB_QUERY}'
    - If it's a general query that can be answered directly, respond with '{AgentAction.FINAL_RESPONSE}'
    
    User Input: {state['input']}
    
    Respond with only one of the above action values.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    action = response.choices[0].message.content.strip()
    print(f"### Action: {action}")
    state['action'] = action
    return state

def policy_agent(state: GraphState):
    """
    Agent that handles company policy related queries using RAG
    """
    # Placeholder for actual RAG implementation
    policy_response = search_docs(state['input'])
    state['context']['policy_info'] = policy_response
    print(f"### state at end of policy agent: {state}")
    return state

def booking_agent(state: GraphState):
    """
    Agent that handles appointment booking
    """
    # Placeholder for actual booking implementation
    booking_response = book_appointment(state['input'])
    state['context']['booking_info'] = booking_response
    return state

def products_agent(state: GraphState):
    """
    Agent that handles product information
    """
    products_response = search_products(state['input'])
    state['context']['products_info'] = products_response
    return state

def response_generator(state: GraphState):
    """
    Generate the final response based on the context and previous agent outputs
    """
    # print(f"### state at start of response generator: {state}")
    context = json.dumps(state['context'])
    
    prompt = f"""Generate a helpful and coherent response to the user's query using the available context.
    Context come from agents:
    policy agent: no need to mofify the context of this agent. 
    booking agent: performs google calendar booking. it gives booking info in json. make sense of the actions performed and respond in natural language.
    products agent: make sense of the product from info and help user understand the product.
    
    User Input: {state['input']}
    Context: {context}
    
    
    
    add product ids at the end of the repsonse seperated by a pipe symbol like this: |product_id1,product_id2,product_id3 if the product ids are not present in the context, add |null
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    state['final_response'] = response.choices[0].message.content.strip()
    return state

def route_to_specialist(state: GraphState) -> str:
    """
    Route to appropriate specialist based on action
    """
    if state['action'] == AgentAction.POLICY_QUERY:
        return "policy_agent"
    elif state['action'] == AgentAction.BOOKING:
        return "booking_agent"
    elif state['action'] == AgentAction.DB_QUERY:
        return "products_agent"
    return "response_generator"

# Create the graph
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("primary_agent", primary_agent)
builder.add_node("policy_agent", policy_agent)
builder.add_node("booking_agent", booking_agent)
builder.add_node("products_agent", products_agent)
builder.add_node("response_generator", response_generator)

# Add edges
builder.add_edge(START, "primary_agent")

# Add conditional edges from primary agent
builder.add_conditional_edges(
    "primary_agent",
    route_to_specialist,
    {
        "policy_agent": "policy_agent",
        "booking_agent": "booking_agent",
        "products_agent": "products_agent",
        "response_generator": "response_generator"
    }
)

# Add edges from specialist agents to response generator
builder.add_edge("policy_agent", "response_generator")
builder.add_edge("booking_agent", "response_generator")
builder.add_edge("products_agent", "response_generator")
builder.add_edge("response_generator", END)

# Compile the graph
graph = builder.compile()

# Draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
except Exception:
    pass

def run_workflow(user_input: str) -> str:
    """
    Run the workflow with the given user input
    """
    inputs = {
        "input": user_input,
        "action": "",
        "response": "",
        "context": {},
        "final_response": ""
    }
    
    result = graph.invoke(inputs)
    return result["final_response"]

# Interactive loop
if __name__ == "__main__":
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        response = run_workflow(user_input)
        print(f"Assistant: {response}")