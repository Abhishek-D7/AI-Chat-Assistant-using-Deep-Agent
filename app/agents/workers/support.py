from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from app.state import DeepAgentState
from app.utils import get_llm


from app.tools.faq_tool import faq_agent_tool

# Mock removed, using real tool

def support_node(state: DeepAgentState):
    print("---SUPPORT WORKER---")
    plan = state['plan']
    current_step_index = state['current_step_index']
    scratchpad = state['scratchpad']
    current_step = plan[current_step_index]
    
    tools = [faq_agent_tool]
    llm = get_llm().bind_tools(tools)
    
    system_prompt = """You are a Support Agent. 
    Your goal is to complete the current step of the plan.
    
    Current Step:
    {current_step}
    
    Scratchpad:
    {scratchpad}
    """
    
    messages = [
        ("system", system_prompt.format(current_step=current_step, scratchpad=scratchpad)),
    ] + state['messages']
    
    result = llm.invoke(messages)
    
    task_complete = False
    
    if result.tool_calls:
        tool_call = result.tool_calls[0]
        # We need to map tool call arguments correctly
        # faq_agent_tool expects 'user_message'
        # The LLM might generate a different argument name depending on schema.
        # But since we bound it, it should be correct.
        tool_output = faq_agent_tool.invoke(tool_call['args'])
        print(f"Tool executed: faq_agent_tool -> {tool_output}")
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
