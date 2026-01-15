from langchain_core.messages import AIMessage
from app.state import DeepAgentState
from app.utils import get_llm
from app.tools.human_handoff_tool import human_handoff_tool

def crisis_node(state: DeepAgentState):
    print("---CRISIS WORKER---")
    plan = state['plan']
    current_step_index = state['current_step_index']
    scratchpad = state['scratchpad']
    current_step = plan[current_step_index]
    
    tools = [human_handoff_tool]
    llm = get_llm().bind_tools(tools)
    
    system_prompt = """You are a Crisis Management Agent.
    Your goal is to handle the current step of the plan, which involves escalating to a human or handling a sensitive issue.
    
    Current Step:
    {current_step}
    
    Scratchpad:
    {scratchpad}
    """
    
    # We should include the last user message to gauge emotion/sentiment if not explicitly in step
    # But usually the planner/orchestrator has passed context.
    
    messages = [
        ("system", system_prompt.format(current_step=current_step, scratchpad=scratchpad)),
    ] + state['messages']
    
    result = llm.invoke(messages)
    
    task_complete = False
    
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        tool_output = human_handoff_tool.invoke(tool_call['args'])
        print(f"Tool executed: human_handoff_tool -> {tool_output}")
        response_message = AIMessage(content=str(tool_output))
        task_complete = True
    else:
        response_message = result
        # If no tool call, maybe it's asking for more info?
        if "?" in result.content:
            task_complete = False
        else:
             # Assuming if it responds without tool, it might be done or failed.
             # Ideally CrisisAgent SHOULD call the tool.
            task_complete = True

    return {
        "messages": [response_message],
        "task_complete": task_complete
    }
