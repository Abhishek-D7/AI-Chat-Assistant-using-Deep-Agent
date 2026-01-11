from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.state import DeepAgentState
from app.utils import get_llm
from typing import Literal

class Router(BaseModel):
    """Worker to route to"""
    next_worker: Literal["BookingAgent", "SupportAgent", "CrisisAgent"] = Field(description="The worker to route to next")

def orchestrator_node(state: DeepAgentState):
    print("---ORCHESTRATOR---")
    plan = state['plan']
    current_step_index = state['current_step_index']
    
    # Check if we are done
    if current_step_index >= len(plan):
        return {"next_worker": "FINISH"}
    
    current_step = plan[current_step_index]
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(Router)
    
    system_prompt = """You are an Orchestrator.
    Your job is to decide which worker should perform the current step of the plan.
    
    Current Plan:
    {plan}
    
    Current Step:
    {current_step}
    
    Workers:
    - BookingAgent: For checking availability, booking slots, confirming emails related to meetings.
    - SupportAgent: For answering FAQs, general questions, issues.
    - CrisisAgent: For human handoff, high severity issues, or when user demands to talk to a person.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"plan": "\n".join(plan), "current_step": current_step})
    
    return {"next_worker": result.next_worker}
