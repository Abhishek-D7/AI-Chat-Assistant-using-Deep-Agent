from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from app.state import DeepAgentState
from app.utils import get_llm
from app.memory import PineconeMemory

class Plan(BaseModel):
    """Plan to follow to answer the user request"""
    steps: List[str] = Field(description="List of sequential steps to follow")

async def planner_node(state: DeepAgentState):
    print("---PLANNER---")
    messages = state['messages']
    user_id = state['user_id']
    user_message = messages[-1].content
    
    # Search memory
    memory = PineconeMemory()
    try:
        context = await memory.search_memory(user_id, user_message)
        context_str = "\n".join(context)
    except Exception as e:
        print(f"Memory search failed: {e}")
        context_str = "No memory available."

    llm = get_llm()
    structured_llm = llm.with_structured_output(Plan)
    
    system_prompt = """You are a Planner for a B2B AI Chat Assistant. 
    Your job is to break down the user's request into a sequential list of steps.
    
    Consider the user's history/context:
    {context}
    
    Available Workers:
    1. BookingAgent: Can check availability, book meetings.
    2. SupportAgent: Can search FAQs, answer general questions.
    
    Create a concise plan.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ])
    
    chain = prompt | structured_llm
    plan = chain.invoke({"context": context_str, "input": user_message})
    
    return {
        "plan": plan.steps, 
        "current_step_index": 0,
        "task_complete": False,
        "scratchpad": {}
    }
