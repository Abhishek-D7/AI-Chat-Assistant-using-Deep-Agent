from langchain_core.messages import AIMessage
from app.state import DeepAgentState
from app.utils import get_llm
from app.tools.booking_tool import booking_agent_tool

def booking_node(state: DeepAgentState):
    print("---BOOKING WORKER---")
    plan = state['plan']
    current_step_index = state['current_step_index']
    scratchpad = state['scratchpad']
    
    # Handle edge case where index out of bounds (though orchestrator should prevent)
    if current_step_index >= len(plan):
        return {"task_complete": True}

    current_step = plan[current_step_index]
    
    tools = [booking_agent_tool]
    llm = get_llm().bind_tools(tools)
    
    system_prompt = """You are a Booking Agent.
    Your goal is to complete the current step of the plan, which involves checking availability or booking meetings.
    
    Current Step:
    {current_step}
    
    Scratchpad (History of actions in this turn):
    {scratchpad}
    """
    
    messages = [
        ("system", system_prompt.format(current_step=current_step, scratchpad=scratchpad)),
    ] + state['messages']
    
    # Invoke LLM
    result = llm.invoke(messages)
    
    task_complete = False
    
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        # Execute tool
        # We need to ensure we run custom async tools properly if they are async.
        # booking_agent_tool is async def, so we should await it if we were in async context.
        # However, LangGraph nodes are sync functions in this setup? 
        # Wait, if we are in main.py: await graph.invoke(), then nodes can be async?
        # But this function 'booking_node' is def, not async def.
        # So we should run the tool synchronously?
        # But booking_agent_tool is async.
        # We need to bridge this.
        # For now, let's use a helper to run async tool in sync context or make this node async.
        # The graph supports async nodes. Let's make this async.
        pass

# Redefining as async to handle async tool
async def booking_node(state: DeepAgentState):
    print("---BOOKING WORKER---")
    plan = state['plan']
    current_step_index = state['current_step_index']
    scratchpad = state['scratchpad']
    
    if current_step_index >= len(plan):
        return {"task_complete": True}

    current_step = plan[current_step_index]
    
    tools = [booking_agent_tool]
    llm = get_llm().bind_tools(tools)
    
    system_prompt = """You are a Booking Agent.
    Your goal is to complete the current step of the plan, which involves checking availability or booking meetings.
    
    Current Step:
    {current_step}
    
    Scratchpad:
    {scratchpad}
    """
    
    messages = [
        ("system", system_prompt.format(current_step=current_step, scratchpad=scratchpad)),
    ] + state['messages']
    
    result = await llm.ainvoke(messages)
    
    task_complete = False
    
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        print(f"Tool Request: {tool_call['name']}")
        
        # Invoke the async tool
        tool_output = await booking_agent_tool.ainvoke(tool_call['args'])
        
        print(f"Tool executed: booking_agent_tool -> {tool_output}")
        response_message = AIMessage(content=str(tool_output))
        task_complete = True
    else:
        response_message = result
        if "?" in result.content:
            task_complete = False
        else:
            task_complete = True

    return {
        "messages": [response_message],
        "task_complete": task_complete
    }
