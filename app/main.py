from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.graph import create_graph
from langchain_core.messages import HumanMessage
from app.memory import PineconeMemory
import uuid

app = FastAPI(title="Deep Agent API")
graph = create_graph()
memory_client = PineconeMemory()

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Check if this is a new thread or existing
    # LangGraph MemorySaver handles persistence based on thread_id
    
    # We should add the user message to the state
    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        # Initialize other keys if they don't exist in state (handled by graph but good to be explicit for first run)
        # Note: If resuming, these might be overwritten by state history, which is what we want.
    }
    
    # Run the graph
    # We use stream to get the final state
    try:
        final_state = None
        # We need to run until it pauses (waiting for input) or finishes
        # graph.invoke runs until end.
        final_state = await graph.ainvoke(inputs, config=config)
        
        # Extract response
        messages = final_state['messages']
        last_message_obj = messages[-1]
        
        if isinstance(last_message_obj, HumanMessage):
            # This means the agent produced NO response.
            # We should probably return a generic error or log it.
            print("WARNING: Agent produced no response, preventing echo.")
            last_message = "I apologize, but I encountered an internal issue and couldn't process your request. Please try again."
        else:
            last_message = last_message_obj.content
        
        # Save to memory (long-term) if it's a significant info?
        # For this demo, let's autosave user messages.
        # But maybe better done inside agent.
        # Let's save the user message here for simplicity of prompt reqs:
        # "add_memory... Embed the text... Integration: The Planner should call..."
        # So we add memory here? Or in the planner?
        # The prompt says: "add_memory(..., text)".
        # Let's add the Human input to memory so future plans know about it.
        await memory_client.add_memory(request.user_id, request.message)
        
        return {
            "response": last_message,
            "plan": final_state.get('plan', []),
            "current_step": final_state.get('current_step_index', 0),
            "task_complete": final_state.get('task_complete', False)
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        with open("server_error.log", "w") as f:
            f.write(error_msg)
        print(f"ERROR CAUGHT: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
