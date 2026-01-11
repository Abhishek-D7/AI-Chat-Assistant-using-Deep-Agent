from langgraph.graph import StateGraph, START, END
from app.state import DeepAgentState
from app.agents.planner import planner_node
from app.agents.orchestrator import orchestrator_node 
from app.agents.workers.booking import booking_node
from app.agents.workers.support import support_node
from app.agents.workers.crisis import crisis_node
from app.agents.reviewer import reviewer_node, reviewer_conditional
from langgraph.checkpoint.memory import MemorySaver

def create_graph():
    builder = StateGraph(DeepAgentState)
    
    # Add Nodes
    builder.add_node("Planner", planner_node)
    builder.add_node("Orchestrator", orchestrator_node)
    builder.add_node("BookingAgent", booking_node)
    builder.add_node("SupportAgent", support_node)
    builder.add_node("CrisisAgent", crisis_node) 
    builder.add_node("Reviewer", reviewer_node)
      
    # Edges
    # START -> Planner
    # Ensure Planner is only called if Plan is empty? 
    # Or purely START -> Planner.
    # The prompt says: "Flow: START -> Planner -> Orchestrator..."
    # If we resume from a question, we don't want to go back to Planner.
    # We'll handle that conditional at START if needed, or rely on persistent state.
    # For a fresh request, START -> Planner. 
    # If we are resuming, we probably have a plan. 
    # builder.add_conditional_edges(START, start_conditional)
    builder.add_edge(START, "Planner")
    
    builder.add_edge("Planner", "Orchestrator")
    
    # Orchestrator Conditional
    # Returns "BookingAgent", "SupportAgent", or "FINISH" (if index > len)
    # But orchestrator_node returns the name.
    
    def orchestrator_routing(state: DeepAgentState):
        worker_name = state.get("next_worker")
        if worker_name == "FINISH":
            return END
        return worker_name
        
    builder.add_conditional_edges("Orchestrator", orchestrator_routing, ["BookingAgent", "SupportAgent", "CrisisAgent", END])
    
    builder.add_edge("BookingAgent", "Reviewer")
    builder.add_edge("SupportAgent", "Reviewer")
    builder.add_edge("CrisisAgent", "Reviewer")
    
    # Reviewer Conditional
    builder.add_conditional_edges("Reviewer", reviewer_conditional, {
        "Orchestrator": "Orchestrator",
        "END": END
    })
    
    # Compile
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
