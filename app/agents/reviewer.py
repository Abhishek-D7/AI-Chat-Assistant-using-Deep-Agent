from app.state import DeepAgentState
from langchain_core.messages import AIMessage

def reviewer_node(state: DeepAgentState):
    print("---REVIEWER---")
    task_complete = state['task_complete']
    current_step_index = state['current_step_index']
    plan = state['plan']
    
    # Check if the last message is a question from the worker
    last_message = state['messages'][-1]
    is_question = "?" in last_message.content if hasattr(last_message, 'content') else False
    
    if task_complete:
        # Success, move to next step
        return {"current_step_index": current_step_index + 1}
    
    if not task_complete and is_question:
        # Need user input, so we finish the run to wait for user
        # In LangGraph, we can return a specific key or just let the ConditionalEdge handle it
        # But here we are a specific NODE.
        # If we return, we update state.
        
        # We need to signal to the Graph that we are "pausing".
        # Typically this is done by routing to END.
        return {} # No state update needed, just pass through
        
    if not task_complete and not is_question:
        # Failure/Retrying
        # Add critique
        return {"scratchpad": {**state['scratchpad'], "critique": "Previous attempt failed. Retry."}}

def reviewer_conditional(state: DeepAgentState):
    task_complete = state['task_complete']
    current_step_index = state['current_step_index']
    plan = state['plan']
    
    # Check if the last message is a question
    last_message = state['messages'][-1]
    is_question = "?" in last_message.content
    
    if task_complete:
        # If we finished all steps
        if current_step_index >= len(plan):
            return "END"
        # Otherwise go back to Orchestrator for next step
        return "Orchestrator"
    
    if is_question:
        # Wait for user input
        return "END"
        
    # Otherwise retry
    return "Orchestrator"
